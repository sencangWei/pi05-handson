"""Export approved human corrections plus the pinned public training episodes."""
import argparse
import io
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from correction_data import APPROVAL, approval_checks, digest, load_frames, write_json

ROOT = Path(__file__).resolve().parents[1]
ASSET_ID = "course/libero_black_bowl"


def safe_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
        raise ValueError("Name must be 1-80 ASCII letters, digits, underscores or hyphens")
    return value


def contained(root, relative):
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts or path.is_symlink():
        raise ValueError("Unsafe artifact path")
    path.resolve().relative_to(root.resolve())
    return path


def file_record(root, path):
    path.resolve().relative_to(root.resolve())
    if path.is_symlink():
        raise ValueError("Symbolic source artifacts are not supported")
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": digest(path)}


def verify_records(root, records):
    paths = [item["path"] for item in records]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate manifest artifacts")
    for item in records:
        path = contained(root, item["path"])
        if not path.is_file() or path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError(f"Artifact differs from manifest: {item['path']}")


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path, values):
    path.write_text("".join(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n" for value in values))


SELECTION_RULE = "one contiguous [start,stop) slice per source; no stitching or frame dropping"
SELECTION_RULE_V2 = "each clip is a continuous [start,stop) slice; separate episodes, no stitching or frame dropping"


def load_selection(root, selection, task):
    """Read a local, immutable contiguous-clip plan without changing source recordings."""
    path = Path(selection)
    if path.is_absolute():
        path = path.relative_to(root)
    path = contained(root, path)
    if path.parent != root / "evidence" or path.suffix != ".json":
        raise ValueError("Correction selection must be a JSON file in this project's evidence directory")
    value = json.loads(path.read_text())
    clips = value.get("clips")
    version = value.get("schema_version")
    if (type(version) is not int or version not in (1, 2)
            or value.get("task") != task or value.get("rule") != {1: SELECTION_RULE, 2: SELECTION_RULE_V2}.get(version)
            or not isinstance(clips, list) or not clips):
        raise ValueError("Correction selection schema, task or contiguous-slice rule differs")
    sources = {}
    fields = {"source_directory", "metadata_sha256", "source_frame_count", "start", "stop", "reason"}
    if version == 2:
        fields.add("kind")
    for clip in clips:
        if not isinstance(clip, dict) or set(clip) != fields:
            raise ValueError("Correction selection clip fields differ")
        source = clip["source_directory"]
        if not isinstance(source, str):
            raise ValueError("Invalid correction selection source directory")
        directory = contained(root, source)
        if (directory.parent != root / "corrections" or not directory.name.startswith("human_")
                or source != str(directory.relative_to(root))):
            raise ValueError("Selection must reference an original human correction directory")
        if version == 1 and source in sources:
            raise ValueError("Only one contiguous selection per correction source is allowed")
        if version == 2 and clip["kind"] not in ("placement_release", "lift_clearance", "regrasp_lift"):
            raise ValueError("Unknown curated recovery clip kind")
        start, stop, count = clip["start"], clip["stop"], clip["source_frame_count"]
        if (any(type(number) is not int for number in (start, stop, count))
                or not 0 <= start < stop <= count or stop - start < 10
                or not isinstance(clip["metadata_sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", clip["metadata_sha256"])
                or not isinstance(clip["reason"], str) or not clip["reason"].strip()):
            raise ValueError("Correction selection bounds, checksum or reason differ")
        if any(start < old_stop and old_start < stop for old_start, old_stop in sources.get(source, [])):
            raise ValueError("Correction selection intervals overlap within a source")
        sources.setdefault(source, []).append((start, stop))
    return path, clips


def selected_frames(root, clip, metadata_path, metadata, frames):
    if (str(metadata_path.parent.relative_to(root)) != clip["source_directory"]
            or metadata.get("status") != "approved" or metadata.get("split") != "train"
            or metadata.get("approval_type") != APPROVAL or metadata.get("human_actions_only") is not True
            or digest(metadata_path) != clip["metadata_sha256"]
            or metadata.get("frame_count") != clip["source_frame_count"]
            or len(frames) != clip["source_frame_count"]):
        raise ValueError("Selected correction source approval, metadata checksum or frame count differs")
    selected = frames[clip["start"]:clip["stop"]]
    if "kind" in clip:
        from curated_checks import curated_clip_checks
        curated_clip_checks(clip["kind"], selected)
    else:
        approval_checks(selected)
    return selected


def prepare(root, name, base_manifest=None, selection=None):
    root, name = Path(root).resolve(), safe_name(name)
    repo_id = f"course/libero_corrections_{name}"
    output = root / "data/lerobot" / repo_id
    manifest_path = root / "evidence" / f"corrections-{name}-manifest.json"
    if output.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite an exported dataset or manifest")
    pinned_path = root / "evidence/data_manifest.json"
    pinned = json.loads(pinned_path.read_text())
    if (pinned.get("ready") is not True or pinned.get("train_episodes") != 16
            or set(pinned["train_source_episode_ids"]) & set(pinned["holdout_source_episode_ids"])):
        raise ValueError("Expected the pinned, disjoint 16-episode public training dataset")
    if selection is not None and base_manifest is None:
        raise ValueError("Correction selection requires an expanded public base")
    base = None
    inputs = [file_record(root, pinned_path)]
    if base_manifest is not None:
        from train_corrections import validate_expanded_public
        base_path = Path(base_manifest)
        if not base_path.is_absolute():
            base_path = root / base_path
        base_path = base_path.resolve()
        if base_path.parent != root / "evidence":
            raise ValueError("Public base must be an exported evidence manifest in this project")
        base = json.loads(base_path.read_text())
        if base.get("dataset_kind") != "expanded_public_v1" or base.get("ready") is not True or base.get("project_root") != str(root):
            raise ValueError("Expected a ready expanded public-only base manifest")
        verify_records(root, base["files"] + base["inputs"])
        validate_expanded_public(root, base)
        original = contained(root, base["dataset_dir"])
        source_files = base["files"]
        inputs += [file_record(root, base_path)] + base["files"] + base["inputs"]
        inputs = list({item["path"]: item for item in inputs}.values())
    else:
        original = contained(root, pinned["destination"])
        source_files = pinned["files"]
    public_episodes = base["train_episodes"] if base else 16
    public_frames = base["train_frames"] if base else pinned["train_frames"]
    original_files = [item for item in source_files if Path(item["path"]).is_relative_to(original.relative_to(root))]
    if base is None:
        inputs += original_files
    verify_records(root, original_files)
    if {item["path"] for item in original_files} != {str(p.relative_to(root)) for p in original.rglob("*") if p.is_file()}:
        raise ValueError("Pinned public dataset inventory differs")
    info = json.loads((original / "meta/info.json").read_text())
    if info["fps"] != 10 or info["total_episodes"] != public_episodes:
        raise ValueError("Public dataset contract differs")
    selection_path, clips = load_selection(root, selection, pinned["task"]) if selection is not None else (None, None)
    if selection_path is not None:
        inputs.append(file_record(root, selection_path))
        inputs = list({item["path"]: item for item in inputs}.values())
    corrections, scenarios, raw_files = [], {"train": set(), "holdout": set()}, []
    for metadata_path in sorted((root / "corrections").glob("*/metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("status") != "approved":
            continue
        frames = load_frames(metadata_path.parent, metadata)
        if (metadata.get("approval_type") != APPROVAL or metadata.get("human_actions_only") is not True
                or metadata.get("task") != pinned["task"]):
            raise ValueError("Correction lacks explicit matching task or human approval")
        approval_checks(frames)
        scenarios[metadata["split"]].add(metadata["scenario_key"])
        raw_files.extend(file_record(root, p) for p in sorted(metadata_path.parent.rglob("*")) if p.is_file())
        corrections.append((metadata_path, metadata, frames))
    if scenarios["train"] & scenarios["holdout"]:
        raise ValueError("The same scenario_key occurs in train and holdout corrections")
    train = [record for record in corrections if record[1]["split"] == "train"]
    if clips is not None:
        sources = {str(path.parent.relative_to(root)): (path, metadata, frames) for path, metadata, frames in corrections}
        train = []
        for clip in clips:
            if clip["source_directory"] not in sources:
                raise ValueError("Selected correction source is missing or unapproved")
            path, metadata, frames = sources[clip["source_directory"]]
            train.append((path, metadata, selected_frames(root, clip, path, metadata, frames)))
    if not train:
        raise ValueError("No approved train correction episodes")
    # Arrow, PNG processing and statistics are needed only after input validation.
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image
    from prepare_public_data import feature_stats

    norm = root / "assets/pi05_handson_lora" / ASSET_ID / "norm_stats.json"
    norm_record = file_record(root, norm)
    if base is not None and (base["normalization"]["sha256"] != norm_record["sha256"]
                             or base["normalization"].get("recomputed") is not False):
        raise ValueError("Expanded public base normalization differs")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".corrections-stage-", dir=output.parent) as temporary:
        staging = Path(temporary) / "dataset"
        shutil.copytree(original, staging)
        episodes = jsonl(staging / "meta/episodes.jsonl")
        stats = jsonl(staging / "meta/episodes_stats.jsonl")
        if len(episodes) != public_episodes or len(stats) != public_episodes or sum(ep["length"] for ep in episodes) != public_frames:
            raise ValueError("Original episode metadata differs from pinned counts")
        schema = pq.read_schema(original / "data/chunk-000/episode_000000.parquet")
        if set(schema.names) != {"image", "wrist_image", "state", "actions", "timestamp", "frame_index", "episode_index", "index", "task_index"}:
            raise ValueError("Unexpected original Arrow fields")
        offset = public_frames
        for episode_id, (_, metadata, frames) in enumerate(train, start=public_episodes):
            rows = []
            for frame_id, frame in enumerate(frames):
                row = {"state": frame["state"].tolist(), "actions": frame["action"].tolist(),
                       "timestamp": frame_id / info["fps"], "frame_index": frame_id,
                       "episode_index": episode_id, "index": offset + frame_id, "task_index": 0}
                for key in ("image", "wrist_image"):
                    stream = io.BytesIO()
                    Image.fromarray(frame[key]).save(stream, format="PNG")
                    row[key] = {"bytes": stream.getvalue(), "path": None}
                rows.append(row)
            table = pa.Table.from_pylist(rows, schema=schema)
            destination = staging / info["data_path"].format(episode_chunk=episode_id // info["chunks_size"], episode_index=episode_id)
            destination.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, destination)
            if not pq.read_table(destination).equals(table, check_metadata=True):
                raise ValueError("Correction Arrow roundtrip changed content or metadata")
            episodes.append({"episode_index": episode_id, "tasks": [pinned["task"]], "length": len(frames)})
            stats.append({"episode_index": episode_id, "stats": feature_stats(table, info["features"])})
            offset += len(frames)
        info.update(total_episodes=len(episodes), total_frames=offset,
                    total_chunks=(len(episodes) - 1) // info["chunks_size"] + 1,
                    splits={"train": f"0:{len(episodes)}"})
        write_json(staging / "meta/info.json", info)
        write_jsonl(staging / "meta/episodes.jsonl", episodes)
        write_jsonl(staging / "meta/episodes_stats.jsonl", stats)
        # Public episode parquet files are copied byte-for-byte; no original is rewritten.
        for item in original_files:
            relative = Path(item["path"]).relative_to(original.relative_to(root))
            if relative.parts[0] == "data" and digest(staging / relative) != item["sha256"]:
                raise ValueError("Original episode changed in export")
        dataset_files = [{**file_record(staging, p), "path": str((output / p.relative_to(staging)).relative_to(root))}
                         for p in sorted(staging.rglob("*")) if p.is_file()]
        manifest = {
            "ready": True, "project_root": str(root), "repo_id": repo_id,
            "dataset_dir": str(output.relative_to(root)), "task": pinned["task"],
            "original_train_episodes": public_episodes, "original_train_frames": public_frames,
            "correction_train_episodes": len(train), "correction_train_frames": offset - public_frames,
            "correction_holdout_episodes_excluded": sum(metadata["split"] == "holdout" for _, metadata, _ in corrections),
            "old_holdout_source_episode_ids_excluded": pinned["holdout_source_episode_ids"],
            "train_source_episode_ids": base["train_source_episode_ids"] if base else pinned["train_source_episode_ids"],
            "approval_type": APPROVAL, "official_success_metric": False,
            "sampling": "uniform over combined public and correction frames",
            "normalization": {"asset_id": ASSET_ID, "recomputed": False, **norm_record},
            "timing": {"nominal_fps": info["fps"], "nominal_dt": 1 / info["fps"],
                       "pauses_excluded": True, "one_frame_per_human_control_action": True,
                       "physical_control_dt_by_episode": {str(p.parent.relative_to(root)): m["control_dt"] for p, m, _ in train},
                       "caveat": "Nominal timestamps follow the public 10 Hz convention; physical correction control_dt is recorded separately and may differ."},
            "scenarios": {key: sorted(values) for key, values in scenarios.items()},
            "files": dataset_files + raw_files,
            "inputs": inputs,
        }
        if base is not None:
            manifest.update(
                dataset_kind="expanded_public_corrections_v1", human_corrections_included=True,
                public_base_manifest=str(base_path.relative_to(root)),
                train_episodes=len(episodes), train_frames=offset,
                holdout_episodes=base["holdout_episodes"], holdout_frames=base["holdout_frames"],
                holdout_source_episode_ids_excluded=base["holdout_source_episode_ids_excluded"],
                source_repo=base["source_repo"], source_revision=base["source_revision"], seed=base["seed"],
                correction_source_directories=[str(p.parent.relative_to(root)) for p, _, _ in train],
                curation="Complete user-approved trajectories; endpoint gates verified, not a per-action expert-quality certification",
            )
        if clips is not None:
            manifest.update(curation_selection=str(selection_path.relative_to(root)), selected_source_ranges=clips,
                            curation=("Selected continuous phase clips in separate episodes; partial recovery phases do not imply task success; unchanged actions and observations, no stitching or frame dropping"
                                      if any("kind" in clip for clip in clips) else
                                      "Selected contiguous source clips; unchanged actions and observations, no stitching or frame dropping; endpoint gates verified"))
        # Recheck source snapshots immediately before publishing.
        verify_records(root, manifest["inputs"] + raw_files + [norm_record])
        staged_manifest = Path(temporary) / "manifest.json"
        write_json(staged_manifest, manifest)
        staging.rename(output)
        # Atomic, exclusive publication: readers never see a partially written ready manifest.
        os.link(staged_manifest, manifest_path)
    return manifest_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--name", required=True, type=safe_name)
    parser.add_argument("--base-manifest", type=Path, help="Validated expanded public-only export to combine with approved human corrections")
    parser.add_argument("--selection", type=Path, help="Evidence JSON selecting continuous phase clips as separate episodes (requires expanded public base)")
    args = parser.parse_args(argv)
    print(prepare(args.project_root, args.name, args.base_manifest, args.selection))


if __name__ == "__main__":
    main()
