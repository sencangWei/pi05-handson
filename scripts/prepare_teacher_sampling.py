"""Pin the same provenance/phase mixture for original-speed and fast teacher controls."""
import argparse
import json
from pathlib import Path

import numpy as np

from correction_data import digest, write_json
from prepare_corrections import safe_name
from prepare_phase_tempo import ranges
from prepare_tracking_teacher import load_base, validate_export
from recovery_sampling import prepare_sampling

ROOT = Path(__file__).resolve().parents[1]


def phase_groups(manifest, teacher_references):
    """Every repeated start keeps its native ten-step, within-episode future window."""
    public = manifest["original_train_frames"]
    offset = manifest["original_train_frames"] + manifest["correction_train_frames"]
    human_end = offset
    carry = set()
    if len(teacher_references) != 2 or len(manifest["teachers"]) != 2:
        raise ValueError("Expected the two matched tracking teachers")
    for teacher, (metadata, reference) in zip(manifest["teachers"], teacher_references, strict=True):
        count = teacher["frame_count"]
        reference = np.asarray(reference)
        if reference.shape != (count,) or metadata["frame_count"] != count:
            raise ValueError("Teacher reference length differs")
        start, stop = metadata["reference_carry_interval"]
        protected = metadata["protected_source_steps"]
        if protected != 10 or stop - start <= 2 * protected:
            raise ValueError("Expected ten protected source steps at both carry boundaries")
        for i in range(count - 9):
            window = reference[i:i + 10]
            if np.all((window >= start + protected) & (window < stop - protected)):
                carry.add(offset + i)
        offset += count
    if offset != manifest["train_frames"] or not carry:
        raise ValueError("Teacher totals or protected carry starts differ")
    generated = set(range(human_end, offset))
    return [
        {"name": "public_complete_demos", "ranges": [[0, public]], "virtual_count": 60000},
        {"name": "all_approved_human", "ranges": [[public, human_end]], "virtual_count": 20000},
        {"name": "validated_teacher_normal_carry", "ranges": ranges(carry), "virtual_count": 15000},
        {"name": "validated_teacher_contact_and_other", "ranges": ranges(generated - carry), "virtual_count": 5000},
    ]


def prepare(root, manifest_path, name):
    root = Path(root).resolve()
    manifest_path = Path(manifest_path).resolve()
    if manifest_path.parent != root / "evidence":
        raise ValueError("Expected an exported evidence manifest")
    name = safe_name(name)
    manifest = json.loads(manifest_path.read_text())
    if "training_sampling_profile" in manifest:
        raise ValueError("Expected the plain tracking teacher export")
    base = load_base(root)
    teachers = validate_export(root, manifest, base)
    references = [(metadata, np.load(directory / "reference-index.npy", allow_pickle=False))
                  for directory, metadata, _ in teachers]
    profile_path = root / "evidence" / f"teacher-sampling-{name}-profile.json"
    alias_path = root / "evidence" / f"tracking-teacher-{name}-manifest.json"
    if profile_path.exists() or alias_path.exists():
        raise FileExistsError("Refusing to overwrite a sampling contract")
    profile = {"schema_version": 1, "seed": manifest["seed"], "repo_id": manifest["repo_id"],
               "dataset_manifest": str(manifest_path.relative_to(root)),
               "dataset_manifest_sha256": digest(manifest_path), "virtual_frames": 100000,
               "groups": phase_groups(manifest, references)}
    write_json(profile_path, profile)
    alias = {**manifest, "training_sampling_profile": {
        "path": str(profile_path.relative_to(root)), "sha256": digest(profile_path)}}
    _, record = prepare_sampling(root, alias)
    write_json(alias_path, alias)
    return alias_path, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--name", type=safe_name, required=True)
    args = parser.parse_args()
    path, record = prepare(ROOT, args.manifest, args.name)
    print(json.dumps({"manifest": str(path), "sampling": record}, indent=2))


if __name__ == "__main__":
    main()
