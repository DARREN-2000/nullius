import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nullius.app import load_vision
from scripts.build_site import collect_studies, imaging

def main():
    pipeline, catalogue = load_vision(ROOT / "models")
    if pipeline is None:
        print("run scripts/train_lesion_model.py first", file=sys.stderr)
        return 1
    
    # Fix paths
    import re
    for row in catalogue:
        match = re.search(r'(data/dicom/stu-\d+\.dcm)', row["path"].replace("\\", "/"))
        if match:
            row["path"] = str(ROOT / match.group(1))

    studies = collect_studies(pipeline, catalogue)
    bundle = pipeline.bundle
    payload = {
        "studies": studies,
        "featureLabels": imaging.FEATURE_LABELS,
        "featureMean": dict(zip(bundle.feature_names, bundle.mean)),
        "operatingPoint": bundle.operating_point,
    }
    
    out_dir = ROOT / "ui" / "public"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "payload.json").write_text(json.dumps(payload), encoding="utf-8")
    
    report = json.loads((ROOT / "out" / "report.json").read_text("utf-8"))
    (out_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    
    print(f"wrote payload to {out_dir / 'payload.json'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
