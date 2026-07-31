"""DICOM Part 10: encode, decode and de-identify - in pure Python.

Why hand-rolled rather than pydicom: this repo must run with `python3` and no
install step (ADR-002), and the sandbox has no network. But the deeper reason is
that "we support DICOM" should mean the format is understood, not that a library
was imported. Everything here is written against PS3.10 (file format) and PS3.5
(value representations).

Scope, stated honestly:
  * Explicit VR Little Endian only (1.2.840.10008.1.2.1). Every conformant
    implementation must support it, so it is the correct single choice.
  * No sequences (SQ), no compressed pixel data, no multi-frame.
  * Single-sample MONOCHROME2, 16 bits allocated.
Anything outside that raises rather than guessing, because silently
misinterpreting pixel data is exactly the failure mode that hurts people.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EXPLICIT_VR_LITTLE_ENDIAN = "1.2.840.10008.1.2.1"
SECONDARY_CAPTURE_STORAGE = "1.2.840.10008.5.1.4.1.1.7"
# Root is a real, syntactically valid UID root reserved for this project's demo data.
UID_ROOT = "1.2.826.0.1.3680043.10.9999"
IMPLEMENTATION_CLASS_UID = UID_ROOT + ".1"

# VRs whose length field is 4 bytes preceded by 2 reserved bytes (PS3.5 7.1.2).
EXTENDED_LENGTH_VRS = {"OB", "OW", "OF", "SQ", "UT", "UN"}

# Tag -> (keyword, VR). Deliberately small: only what this pipeline reads or writes.
DICTIONARY: dict[tuple[int, int], tuple[str, str]] = {
    (0x0002, 0x0000): ("FileMetaInformationGroupLength", "UL"),
    (0x0002, 0x0001): ("FileMetaInformationVersion", "OB"),
    (0x0002, 0x0002): ("MediaStorageSOPClassUID", "UI"),
    (0x0002, 0x0003): ("MediaStorageSOPInstanceUID", "UI"),
    (0x0002, 0x0010): ("TransferSyntaxUID", "UI"),
    (0x0002, 0x0012): ("ImplementationClassUID", "UI"),
    (0x0008, 0x0016): ("SOPClassUID", "UI"),
    (0x0008, 0x0018): ("SOPInstanceUID", "UI"),
    (0x0008, 0x0020): ("StudyDate", "DA"),
    (0x0008, 0x0030): ("StudyTime", "TM"),
    (0x0008, 0x0050): ("AccessionNumber", "SH"),
    (0x0008, 0x0060): ("Modality", "CS"),
    (0x0008, 0x0070): ("Manufacturer", "LO"),
    (0x0008, 0x0080): ("InstitutionName", "LO"),
    (0x0008, 0x0090): ("ReferringPhysicianName", "PN"),
    (0x0008, 0x1030): ("StudyDescription", "LO"),
    (0x0008, 0x103E): ("SeriesDescription", "LO"),
    (0x0010, 0x0010): ("PatientName", "PN"),
    (0x0010, 0x0020): ("PatientID", "LO"),
    (0x0010, 0x0030): ("PatientBirthDate", "DA"),
    (0x0010, 0x0040): ("PatientSex", "CS"),
    (0x0012, 0x0062): ("PatientIdentityRemoved", "CS"),
    (0x0012, 0x0063): ("DeidentificationMethod", "LO"),
    (0x0018, 0x0015): ("BodyPartExamined", "CS"),
    (0x0020, 0x000D): ("StudyInstanceUID", "UI"),
    (0x0020, 0x000E): ("SeriesInstanceUID", "UI"),
    (0x0020, 0x0011): ("SeriesNumber", "IS"),
    (0x0020, 0x0013): ("InstanceNumber", "IS"),
    (0x0028, 0x0002): ("SamplesPerPixel", "US"),
    (0x0028, 0x0004): ("PhotometricInterpretation", "CS"),
    (0x0028, 0x0010): ("Rows", "US"),
    (0x0028, 0x0011): ("Columns", "US"),
    (0x0028, 0x0030): ("PixelSpacing", "DS"),
    (0x0028, 0x0100): ("BitsAllocated", "US"),
    (0x0028, 0x0101): ("BitsStored", "US"),
    (0x0028, 0x0102): ("HighBit", "US"),
    (0x0028, 0x0103): ("PixelRepresentation", "US"),
    (0x0028, 0x1050): ("WindowCenter", "DS"),
    (0x0028, 0x1051): ("WindowWidth", "DS"),
    (0x7FE0, 0x0010): ("PixelData", "OW"),
}
BY_KEYWORD: dict[str, tuple[tuple[int, int], str]] = {
    keyword: (tag, vr) for tag, (keyword, vr) in DICTIONARY.items()
}

# Direct identifiers this pipeline refuses to carry into the model path.
PHI_KEYWORDS = (
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "AccessionNumber",
    "InstitutionName",
    "ReferringPhysicianName",
    "StudyTime",
)


class DicomError(ValueError):
    """Raised instead of guessing when a file is outside the supported profile."""


@dataclass
class Dataset:
    elements: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def __contains__(self, keyword: str) -> bool:
        return keyword in self.elements

    def __getitem__(self, keyword: str) -> Any:
        return self.elements[keyword]

    def get(self, keyword: str, default: Any = None) -> Any:
        return self.elements.get(keyword, default)

    @property
    def rows(self) -> int:
        return int(self.elements["Rows"])

    @property
    def columns(self) -> int:
        return int(self.elements["Columns"])

    @property
    def pixel_spacing_mm(self) -> float:
        spacing = self.elements.get("PixelSpacing", 1.0)
        return float(spacing[0] if isinstance(spacing, (list, tuple)) else spacing)

    def pixels(self) -> list[int]:
        """Flat row-major unsigned 16-bit samples."""
        if int(self.elements.get("SamplesPerPixel", 1)) != 1:
            raise DicomError("only single-sample (monochrome) pixel data is supported")
        if int(self.elements.get("BitsAllocated", 16)) != 16:
            raise DicomError("only 16 bits allocated is supported")
        raw = self.elements["PixelData"]
        expected = self.rows * self.columns * 2
        if len(raw) != expected:
            raise DicomError(f"pixel data is {len(raw)} bytes, header declares {expected}")
        return list(struct.unpack(f"<{self.rows * self.columns}H", raw))

    def summary(self) -> dict[str, Any]:
        """The header fields a reviewer actually wants to see, PixelData excluded."""
        return {k: v for k, v in self.elements.items() if k != "PixelData"}


# --------------------------------------------------------------------- encoding
def _encode_value(vr: str, value: Any) -> bytes:
    if vr == "US":
        values = value if isinstance(value, (list, tuple)) else [value]
        return b"".join(struct.pack("<H", int(v)) for v in values)
    if vr == "UL":
        return struct.pack("<I", int(value))
    if vr in ("OB", "OW"):
        data = bytes(value)
        return data + (b"\x00" if len(data) % 2 else b"")
    text = "\\".join(str(v) for v in value) if isinstance(value, (list, tuple)) else str(value)
    data = text.encode("ascii", "replace")
    if len(data) % 2:
        data += b"\x00" if vr == "UI" else b" "
    return data


def _encode_element(tag: tuple[int, int], vr: str, value: Any) -> bytes:
    data = _encode_value(vr, value)
    head = struct.pack("<HH", tag[0], tag[1]) + vr.encode("ascii")
    if vr in EXTENDED_LENGTH_VRS:
        head += b"\x00\x00" + struct.pack("<I", len(data))
    else:
        head += struct.pack("<H", len(data))
    return head + data


def encode(elements: dict[str, Any]) -> bytes:
    """Serialise a dataset to a DICOM Part 10 byte stream."""
    for required in ("SOPClassUID", "SOPInstanceUID", "Rows", "Columns", "PixelData"):
        if required not in elements:
            raise DicomError(f"missing required element {required}")
    unknown = sorted(set(elements) - set(BY_KEYWORD))
    if unknown:
        raise DicomError(f"unknown keywords: {unknown}")

    body = b""
    for keyword in sorted(elements, key=lambda k: BY_KEYWORD[k][0]):
        tag, vr = BY_KEYWORD[keyword]
        body += _encode_element(tag, vr, elements[keyword])

    meta_body = b"".join(
        [
            _encode_element((0x0002, 0x0001), "OB", b"\x00\x01"),
            _encode_element((0x0002, 0x0002), "UI", elements["SOPClassUID"]),
            _encode_element((0x0002, 0x0003), "UI", elements["SOPInstanceUID"]),
            _encode_element((0x0002, 0x0010), "UI", EXPLICIT_VR_LITTLE_ENDIAN),
            _encode_element((0x0002, 0x0012), "UI", IMPLEMENTATION_CLASS_UID),
        ]
    )
    meta = _encode_element((0x0002, 0x0000), "UL", len(meta_body)) + meta_body
    return b"\x00" * 128 + b"DICM" + meta + body


# --------------------------------------------------------------------- decoding
def _decode_value(vr: str, raw: bytes) -> Any:
    if vr == "US":
        values = list(struct.unpack(f"<{len(raw) // 2}H", raw))
        return values[0] if len(values) == 1 else values
    if vr == "UL":
        return struct.unpack("<I", raw)[0]
    if vr in ("OB", "OW"):
        return raw
    text = raw.decode("ascii", "replace").rstrip("\x00 ")
    if vr == "DS":
        parts = [float(p) for p in text.split("\\") if p != ""]
        return parts[0] if len(parts) == 1 else parts
    if vr == "IS":
        parts = [int(float(p)) for p in text.split("\\") if p != ""]
        return parts[0] if len(parts) == 1 else parts
    return text.split("\\") if "\\" in text else text


def decode(blob: bytes) -> Dataset:
    """Parse a DICOM Part 10 byte stream. Raises DicomError on anything unsupported."""
    if len(blob) < 132 or blob[128:132] != b"DICM":
        raise DicomError("not a DICOM Part 10 stream: 'DICM' magic missing at offset 128")
    offset = 132
    elements: dict[str, Any] = {}
    meta: dict[str, Any] = {}
    while offset < len(blob):
        if offset + 8 > len(blob):
            raise DicomError(f"truncated element header at byte {offset}")
        group, element = struct.unpack_from("<HH", blob, offset)
        offset += 4
        vr = blob[offset : offset + 2].decode("ascii", "replace")
        offset += 2
        if vr in EXTENDED_LENGTH_VRS:
            offset += 2
            (length,) = struct.unpack_from("<I", blob, offset)
            offset += 4
        else:
            (length,) = struct.unpack_from("<H", blob, offset)
            offset += 2
        if length == 0xFFFFFFFF:
            raise DicomError("undefined-length elements (sequences) are not supported")
        raw = blob[offset : offset + length]
        if len(raw) != length:
            raise DicomError(f"truncated value for tag ({group:04X},{element:04X})")
        offset += length
        keyword, _ = DICTIONARY.get((group, element), (f"Unknown{group:04X}{element:04X}", vr))
        (meta if group == 0x0002 else elements)[keyword] = _decode_value(vr, raw)

    syntax = meta.get("TransferSyntaxUID")
    if syntax and syntax != EXPLICIT_VR_LITTLE_ENDIAN:
        raise DicomError(f"unsupported transfer syntax {syntax}; only Explicit VR Little Endian")
    return Dataset(elements=elements, meta=meta)


def write_file(path: str | Path, elements: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(encode(elements))
    return target


def read_file(path: str | Path) -> Dataset:
    return decode(Path(path).read_bytes())


# ---------------------------------------------------------------- de-identification
def pseudonym(value: str, salt: str = "nullius") -> str:
    """Stable pseudonym. Salted so the mapping cannot be rebuilt by dictionary attack."""
    return hashlib.sha256(f"{salt}|{value}".encode()).hexdigest()[:16].upper()


def make_uid(*parts: Any) -> str:
    """Deterministic, syntactically valid UID derived from the given parts."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    suffix = str(int(digest[:24], 16))[:24]
    return f"{UID_ROOT}.{suffix}"


def deidentify(dataset: Dataset, salt: str = "nullius") -> tuple[Dataset, dict[str, str]]:
    """Apply a basic de-identification profile before anything reaches the model.

    Pseudonymises direct identifiers, truncates dates to the year, drops
    institution and physician, and records that it happened in (0012,0062) /
    (0012,0063) so a downstream system can verify rather than trust.
    """
    out = dict(dataset.elements)
    mapping: dict[str, str] = {}
    for keyword in ("PatientName", "PatientID"):
        if keyword in out:
            token = pseudonym(str(out[keyword]), salt)
            mapping[str(out[keyword])] = token
            out[keyword] = token
    for keyword in ("PatientBirthDate", "StudyDate"):
        if keyword in out:
            out[keyword] = f"{str(out[keyword])[:4]}0101"
    for keyword in ("AccessionNumber", "InstitutionName", "ReferringPhysicianName", "StudyTime"):
        out.pop(keyword, None)
    out["PatientIdentityRemoved"] = "YES"
    out["DeidentificationMethod"] = (
        "NULLIUS basic profile: identifiers pseudonymised SHA-256, dates to year, institution removed"
    )
    return Dataset(elements=out, meta=dict(dataset.meta)), mapping


def residual_identifiers(dataset: Dataset) -> list[str]:
    """Keywords that still look like raw PHI. Empty list is the only acceptable result.

    A pseudonym is 16 uppercase hex characters; anything else in an identifier
    field is treated as un-scrubbed.
    """
    offenders: list[str] = []
    if dataset.get("PatientIdentityRemoved") != "YES":
        offenders.append("PatientIdentityRemoved")
    for keyword in ("PatientName", "PatientID"):
        value = str(dataset.get(keyword, ""))
        if value and not (len(value) == 16 and all(c in "0123456789ABCDEF" for c in value)):
            offenders.append(keyword)
    for keyword in ("AccessionNumber", "InstitutionName", "ReferringPhysicianName", "StudyTime"):
        if keyword in dataset:
            offenders.append(keyword)
    birth = str(dataset.get("PatientBirthDate", ""))
    if birth and not birth.endswith("0101"):
        offenders.append("PatientBirthDate")
    return offenders
