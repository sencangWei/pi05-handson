"""Rebuild local probe inputs from pinned parquet rows without loading a model."""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / ".local/postgrasp-slow-v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest_path = Path(__file__).with_name("fixed-points.json")
    manifest = json.loads(manifest_path.read_text())
    DEST.mkdir(parents=True, exist_ok=True)
    for point in manifest["points"]:
        source = ROOT / point["source_file"]
        if sha(source) != point["source_sha256"]:
            raise ValueError(f"Parquet checksum mismatch: {source}")
        rows = pq.read_table(source).slice(point["frame"], 10).to_pylist()
        if len(rows) != 10:
            raise ValueError(f"Incomplete window: {point['id']}")
        values = {"state": np.asarray(rows[0]["state"], dtype=np.float32),
                  "gt": np.asarray([r["actions"] for r in rows], dtype=np.float32)}
        for key in ("image", "wrist_image"):
            with Image.open(io.BytesIO(rows[0][key]["bytes"])) as im:
                values[key] = np.asarray(im.convert("RGB"))
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **values)
        payload = buffer.getvalue()
        if hashlib.sha256(payload).hexdigest() != point["bundle_sha256"]:
            raise ValueError(f"Bundle checksum mismatch: {point['id']}")
        target = DEST / point["bundle"]
        if target.exists():
            if sha(target) != point["bundle_sha256"]:
                raise ValueError(f"Existing bundle differs: {target}")
        else:
            with target.open("xb") as f:
                f.write(payload)
    target_manifest = DEST / "fixed-points.json"
    if target_manifest.exists():
        if sha(target_manifest) != sha(manifest_path):
            raise ValueError("Existing points manifest differs")
    else:
        with target_manifest.open("xb") as f:
            f.write(manifest_path.read_bytes())
    print(f"Verified {len(manifest['points'])} identical pinned probe bundles")


if __name__ == "__main__":
    main()
