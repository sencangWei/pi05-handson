"""Prepare phase sampling over unchanged, approved reviewed19_v4 trajectories."""
import argparse
import json
from pathlib import Path

from correction_data import digest
from prepare_corrections import contained, verify_records
from recovery_sampling import prepare_sampling

ROOT = Path(__file__).resolve().parents[1]


def ranges(indices):
    result = []
    for value in sorted(indices):
        if result and result[-1][1] == value:
            result[-1][1] += 1
        else:
            result.append([value, value + 1])
    return result


def phase_groups(manifest, selection):
    """Only lower starts whose entire 10-action window lies inside reviewed transport."""
    clips = manifest["selected_source_ranges"]
    first_new = selection["retained_old_clips"]
    if first_new != 5 or len(clips) != 19 or manifest["original_train_episodes"] != 37:
        raise ValueError("Expected the approved five old plus fourteen new clips")
    offsets, offset = [], manifest["original_train_frames"]
    for clip in clips:
        offsets.append(offset)
        offset += clip["stop"] - clip["start"]
    if offset != manifest["train_frames"]:
        raise ValueError("Selected clip frame totals differ")
    low, seen = set(), set()
    for item in selection["clips"]:
        idx = item["clip_index"]
        if type(idx) is not int or idx in seen or not first_new <= idx < len(clips):
            raise ValueError("Invalid or duplicate new clip index")
        seen.add(idx)
        clip = clips[idx]
        if (item["source_directory"] != clip["source_directory"]
                or item["metadata_sha256"] != clip["metadata_sha256"]):
            raise ValueError("Selected transport source identity differs")
        previous_stop = clip["start"]
        for start, stop in item["raw_transport_ranges"]:
            if (type(start) is not int or type(stop) is not int
                    or not previous_stop <= start < stop <= clip["stop"]):
                raise ValueError("Transport ranges must be ordered, disjoint and inside the approved clip")
            previous_stop = stop
            low.update(range(offsets[idx] + start - clip["start"],
                             offsets[idx] + max(start, stop - 9) - clip["start"]))
    if seen != set(range(first_new, len(clips))) or not low:
        raise ValueError("Transport review must cover every new clip and contain transport")
    new = set(range(offsets[first_new], offset))
    groups = [
        {"name": "public_complete_demos", "ranges": [[0, manifest["original_train_frames"]]], "virtual_count": 140000},
        {"name": "previous_approved_corrections", "ranges": [[offsets[0], offsets[first_new]]], "virtual_count": 20000},
        {"name": "new_contact_and_other", "ranges": ranges(new - low), "virtual_count": 38000},
        {"name": "new_normal_transport", "ranges": ranges(low), "virtual_count": 2000},
    ]
    return groups


def prepare(root, selection_path):
    selection_path = Path(selection_path).resolve()
    if selection_path.parent != root / "evidence":
        raise ValueError("Selection must be an evidence file in this project")
    selection = json.loads(selection_path.read_text())
    base_path = contained(root, selection["dataset_manifest"])
    if digest(base_path) != selection["dataset_manifest_sha256"]:
        raise ValueError("Base manifest checksum differs")
    base = json.loads(base_path.read_text())
    verify_records(root, base["files"] + base["inputs"] + [base["normalization"]])
    for clip in selection["clips"]:
        if digest(contained(root, clip["source_directory"]) / "metadata.json") != clip["metadata_sha256"]:
            raise ValueError("Reviewed raw metadata changed")
    profile_path = root / "evidence/phase-tempo-v1-profile.json"
    alias_path = root / "evidence/corrections-public37_tempo_v1-manifest.json"
    if profile_path.exists() or alias_path.exists():
        raise FileExistsError("Refusing to overwrite an existing tempo profile or manifest")
    profile = {"schema_version": 1, "purpose": "phase_tempo_v1", "seed": base["seed"],
               "repo_id": base["repo_id"], "dataset_manifest": str(base_path.relative_to(root)),
               "dataset_manifest_sha256": digest(base_path), "virtual_frames": 200000,
               "selection": {"path": str(selection_path.relative_to(root)), "sha256": digest(selection_path)},
               "groups": phase_groups(base, selection)}
    with profile_path.open("x") as f:
        json.dump(profile, f, indent=2)
        f.write("\n")
    alias = {**base, "training_sampling_profile": {
        "path": str(profile_path.relative_to(root)), "sha256": digest(profile_path)}}
    _, record = prepare_sampling(root, alias)
    with alias_path.open("x") as f:
        json.dump(alias, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({"manifest": str(alias_path), "sampling": record}, indent=2))
    return alias_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=ROOT / "evidence/phase-tempo-v1-selection.json")
    args = parser.parse_args()
    prepare(ROOT, args.selection)


if __name__ == "__main__":
    main()
