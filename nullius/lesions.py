"""Synthetic dermoscopy studies, written as real DICOM files.

The images are procedurally generated: a skin background with vignetting and
sensor noise, a lesion whose boundary is a radial Fourier series, optional
internal blotches, and hair strokes. Appearance is driven by a single latent
`severity` in [0, 1] that continuously controls the features the ABCD rule names
- asymmetry, border irregularity, colour variegation and diameter.

The two classes draw severity from OVERLAPPING distributions (benign mean 0.30,
suspicious mean 0.70, both sd 0.15), and 6% of studies are deliberately
atypical: the label says one thing and the morphology says the other. This is
not a flaw in the generator, it is the point of it. An earlier version drew the
two classes from disjoint parameter ranges, which made them linearly separable
and produced an AUROC of 1.000 - a number that measured the generator, not the
model, and left the calibration and confidence gates untestable because no score
ever landed in the indeterminate band. Real dermoscopy has ambiguous lesions and
labels that come from histopathology rather than from appearance. A benchmark
you cannot fail teaches you nothing.

This is a generator, not a dataset. It exists so the pipeline can be exercised
end to end, and every accuracy number produced from it measures pipeline
integrity, not clinical validity. ADR-009 states that in full, and the
dashboard repeats it next to the metrics rather than in a footnote. Studies
also carry deliberately realistic PHI so the de-identification stage has
something real to remove.
"""

from __future__ import annotations

import math
import random
import struct
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from . import dicom

NATIVE_SIZE = 160
PIXEL_SPACING_MM = 0.06
# Class-conditional severity distributions. These overlap by design - see the
# module docstring and ADR-012.
SEVERITY_MEAN_BENIGN = 0.30
SEVERITY_MEAN_SUSPICIOUS = 0.70
SEVERITY_SD = 0.15
# Fraction of studies whose morphology contradicts their label.
LABEL_NOISE = 0.06
BODY_SITES = ("BACK", "SHOULDER", "FOREARM", "CALF", "CHEST", "NECK")
INSTITUTIONS = ("ST ANNES DERMATOLOGY", "NORTHFIELD SKIN CLINIC")


@dataclass
class Study:
    study_id: str
    patient_id: str
    patient_name: str
    path: str
    label: int  # 1 = suspicious by construction, 0 = benign by construction
    split: str
    body_site: str
    truth_note: str
    severity: float = 0.0
    atypical: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clamp16(value: float) -> int:
    return max(0, min(65535, int(round(value))))


def draw_severity(rng: random.Random, suspicious: bool) -> tuple[float, bool]:
    """Draw the latent that drives appearance, plus whether this case is atypical.

    The class-conditional distributions overlap heavily on purpose, and a
    fraction of cases are drawn from the *other* class's distribution entirely.
    Those are the lesions that look benign and are not, and vice versa - exactly
    the cases a triage model must be uncertain about rather than confident on.
    """
    atypical = rng.random() < LABEL_NOISE
    looks_suspicious = (not suspicious) if atypical else suspicious
    mean = SEVERITY_MEAN_SUSPICIOUS if looks_suspicious else SEVERITY_MEAN_BENIGN
    severity = min(0.98, max(0.02, rng.gauss(mean, SEVERITY_SD)))
    return severity, atypical


def _render_lesion(rng: random.Random, severity: float, size: int = NATIVE_SIZE) -> list[int]:
    """Return flat 16-bit samples for one synthetic dermoscopy frame.

    Every morphological parameter is a continuous function of `severity`, so the
    feature distributions of the two classes overlap rather than separating.
    """
    centre = size / 2.0
    skin_level = rng.uniform(3000, 3600)
    lesion_level = rng.uniform(1150, 1500)

    # Boundary: r(theta) = base * (1 + sum a_k cos(k theta + phase_k))
    radius_fraction = 0.20 + 0.18 * severity + rng.gauss(0, 0.012)
    base_radius = min(0.42, max(0.17, radius_fraction)) * size
    harmonics: list[tuple[int, float, float]] = []
    for k in (2, 3, 5, 7):
        amplitude = 0.005 + 0.185 * (severity ** 1.3) + rng.gauss(0, 0.010)
        harmonics.append((k, min(0.23, max(0.003, amplitude)), rng.uniform(0, 2 * math.pi)))

    # Centroid offset scales with severity instead of switching on the label.
    offset_scale = (0.008 + 0.062 * severity) * size
    offset_r = rng.gauss(0, 0.55) * offset_scale
    offset_c = rng.gauss(0, 0.55) * offset_scale

    # Blotches become more likely and stronger as severity rises, so a benign
    # lesion can have them and a suspicious one can lack them.
    blotches: list[tuple[float, float, float, float]] = []
    if rng.random() < 0.10 + 0.85 * severity:
        for _ in range(rng.randint(1, 4)):
            blotches.append(
                (
                    centre + rng.uniform(-0.5, 0.5) * base_radius,
                    centre + rng.uniform(-0.5, 0.5) * base_radius,
                    rng.uniform(0.15, 0.35) * base_radius,
                    rng.uniform(-700, 700) * (0.35 + 0.65 * severity),
                )
            )

    pixels: list[int] = []
    for r in range(size):
        for c in range(size):
            dr, dc = r - centre, c - centre
            radial = math.hypot(dr, dc) / centre
            # Vignetting plus a slow illumination gradient: real devices do this.
            value = skin_level * (1.0 - 0.18 * radial * radial) + 90.0 * (c - centre) / size

            lr, lc = r - centre - offset_r, c - centre - offset_c
            distance = math.hypot(lr, lc)
            theta = math.atan2(lr, lc)
            boundary = base_radius * (1.0 + sum(a * math.cos(k * theta + p) for k, a, p in harmonics))
            if distance < boundary:
                value = lesion_level
                for br, bc, brad, delta in blotches:
                    d = math.hypot(r - br, c - bc)
                    if d < brad:
                        value += delta * (1.0 - d / brad)
                # Soften the rim so the border-sharpness feature is meaningful.
                edge = boundary - distance
                if edge < 3.0:
                    value += (skin_level - lesion_level) * (1.0 - edge / 3.0) * 0.45
            pixels.append(_clamp16(value + rng.gauss(0, 45)))

    # Hair strokes: thin dark lines the median filter must remove.
    for _ in range(rng.randint(0, 5)):
        angle = rng.uniform(0, math.pi)
        r0, c0 = rng.uniform(0, size), rng.uniform(0, size)
        for step in range(int(size * 0.9)):
            rr = int(r0 + step * math.sin(angle))
            cc = int(c0 + step * math.cos(angle))
            if 0 <= rr < size and 0 <= cc < size:
                pixels[rr * size + cc] = _clamp16(pixels[rr * size + cc] * 0.45)
    return pixels


def build_study(
    rng: random.Random,
    index: int,
    patient_id: str,
    patient_name: str,
    suspicious: bool,
    split: str,
    out_dir: Path,
) -> Study:
    severity, atypical = draw_severity(rng, suspicious)
    pixels = _render_lesion(rng, severity)
    study_id = f"stu-{index:03d}"
    body_site = rng.choice(BODY_SITES)
    elements: dict[str, Any] = {
        "SOPClassUID": dicom.SECONDARY_CAPTURE_STORAGE,
        "SOPInstanceUID": dicom.make_uid("sop", study_id),
        "StudyInstanceUID": dicom.make_uid("study", patient_id, study_id),
        "SeriesInstanceUID": dicom.make_uid("series", patient_id, study_id),
        "StudyDate": "20260714",
        "StudyTime": "101500",
        "AccessionNumber": f"ACC{index:06d}",
        "Modality": "XC",  # external-camera photography
        "Manufacturer": "NULLIUS Synthetic Dermoscope",
        "InstitutionName": rng.choice(INSTITUTIONS),
        "ReferringPhysicianName": "HALSTEAD^J",
        "StudyDescription": "Dermoscopy, pigmented lesion",
        "SeriesDescription": f"Polarised dermoscopy, {body_site.lower()}",
        "PatientName": patient_name,
        "PatientID": patient_id.upper(),
        "PatientBirthDate": "19570311",
        "PatientSex": "O",
        "BodyPartExamined": body_site,
        "SeriesNumber": 1,
        "InstanceNumber": 1,
        "SamplesPerPixel": 1,
        "PhotometricInterpretation": "MONOCHROME2",
        "Rows": NATIVE_SIZE,
        "Columns": NATIVE_SIZE,
        "PixelSpacing": [PIXEL_SPACING_MM, PIXEL_SPACING_MM],
        "BitsAllocated": 16,
        "BitsStored": 16,
        "HighBit": 15,
        "PixelRepresentation": 0,
        "WindowCenter": 2048,
        "WindowWidth": 4096,
        "PixelData": struct.pack(f"<{len(pixels)}H", *pixels),
    }
    path = out_dir / f"{study_id}.dcm"
    dicom.write_file(path, elements)
    return Study(
        study_id=study_id,
        patient_id=patient_id,
        patient_name=patient_name,
        path=str(path),
        label=1 if suspicious else 0,
        split=split,
        body_site=body_site,
        severity=round(severity, 4),
        atypical=atypical,
        truth_note=(
            (
                "ATYPICAL: labelled suspicious but generated with benign morphology - "
                "the model is expected to get this wrong or abstain"
                if suspicious
                else "ATYPICAL: labelled benign but generated with suspicious morphology - "
                "a false positive here is the correct cautious behaviour"
            )
            if atypical
            else f"severity {severity:.2f} drawn from the "
            f"{'suspicious' if suspicious else 'benign'} distribution"
        ),
    )


def generate_cohort(out_dir: str | Path, count: int = 80, seed: int = 20260731) -> list[Study]:
    """Half suspicious, half benign, split 50/50 into train and test.

    The split is by index parity on a shuffled list, so it is deterministic and
    a study can never appear in both halves.
    """
    rng = random.Random(seed)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    for stale in target.glob("*.dcm"):
        stale.unlink()

    names = [
        "ALVAREZ^MARTA", "OKONKWO^DANIEL", "BERGSTROM^INGRID", "NAKAMURA^KEN",
        "HAIDARI^SORAYA", "MULLER^JONAS", "ROSSI^ELENA", "DUBOIS^CLAIRE",
        "KOWALSKI^PIOTR", "SANTOS^BEATRIZ", "AHMED^YUSUF", "LINDQVIST^ANNA",
    ]
    labels = [i < count // 2 for i in range(count)]
    rng.shuffle(labels)

    studies: list[Study] = []
    for index, suspicious in enumerate(labels):
        patient_number = (index % 12) + 1
        studies.append(
            build_study(
                rng=rng,
                index=index,
                patient_id=f"pat-{patient_number:03d}",
                patient_name=names[patient_number - 1],
                suspicious=suspicious,
                split="train" if index % 2 == 0 else "test",
                out_dir=target,
            )
        )
    return studies
