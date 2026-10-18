"""Export all 47 pinned same-task demonstrations into isolated 37/10 datasets."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from prepare_corrections import ASSET_ID, contained, file_record, jsonl, safe_name, verify_records
from prepare_public_data import REVISION, SOURCE, TASK, feature_stats, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
META_FILES = ("info.json", "tasks.jsonl", "episodes.jsonl")
FIELDS = {"image", "wrist_image", "state", "actions", "timestamp", "frame_index", "episode_index", "index", "task_index"}


def source_path(root, info, episode_id):
    return contained(root / "data/raw_public", info["data_path"].format(
        episode_chunk=episode_id // info["chunks_size"], episode_index=episode_id))


def plan(root):
    """Read cached metadata only; never download, train, or modify old artifacts."""
    root = Path(root).resolve()
    pinned_path = root / "evidence/data_manifest.json"
    pinned = json.loads(pinned_path.read_text())
    if (pinned.get("ready") is not True or pinned.get("source_repo") != SOURCE
            or pinned.get("source_revision") != REVISION or pinned.get("task") != TASK):
        raise ValueError("Original manifest does not identify the pinned public source")
    old_train, old_holdout = pinned["train_source_episode_ids"], pinned["holdout_source_episode_ids"]
    if (len(old_train) != 16 or len(set(old_train)) != 16 or len(old_holdout) != 4
            or len(set(old_holdout)) != 4 or set(old_train) & set(old_holdout)):
        raise ValueError("Original split must contain disjoint 16 train and 4 holdout episodes")
    records = {item["path"]: item for item in pinned["files"]}
    if len(records) != len(pinned["files"]):
        raise ValueError("Original manifest contains duplicate source paths")
    meta_paths = [root / "data/raw_public/meta" / filename for filename in META_FILES]
    try:
        verify_records(root, [records[str(path.relative_to(root))] for path in meta_paths])
    except KeyError as error:
        raise ValueError("Original manifest lacks pinned source metadata hashes") from error
    info = json.loads(meta_paths[0].read_text())
    episodes, tasks = jsonl(meta_paths[2]), jsonl(meta_paths[1])
    if (info.get("fps") != 10 or info.get("codebase_version") != "v2.0"
            or len(episodes) != info["total_episodes"] or len(tasks) != info["total_tasks"]
            or sum(ep["length"] for ep in episodes) != info["total_frames"]
            or len({ep["episode_index"] for ep in episodes}) != len(episodes)):
        raise ValueError("Public source metadata counts or timing differ")
    features = info["features"]
    expected_shapes = {"image": [256, 256, 3], "wrist_image": [256, 256, 3], "state": [8], "actions": [7]}
    expected_dtypes = {key: "int64" for key in FIELDS}
    expected_dtypes.update(image="image", wrist_image="image", state="float32", actions="float32", timestamp="float32")
    if (set(features) != FIELDS
            or any(features[key].get("shape") != expected_shapes.get(key, [1]) for key in FIELDS)
            or any(features[key].get("dtype") != value for key, value in expected_dtypes.items())):
        raise ValueError("Public feature schema differs")
    matching_tasks = [task for task in tasks if task["task"] == TASK]
    selected = sorted((ep for ep in episodes if ep["tasks"] == [TASK]), key=lambda ep: ep["episode_index"])
    ids = [ep["episode_index"] for ep in selected]
    if (len(matching_tasks) != 1 or len(selected) != 47 or sorted(old_train) != ids[:16]
            or sorted(old_holdout) != ids[-4:] or any(ep["length"] <= 0 for ep in selected)):
        raise ValueError("Expected exact-task 47 episodes preserving the original split")
    old_paths = [source_path(root, info, idx) for idx in old_train + old_holdout]
    try:
        verify_records(root, [records[str(path.relative_to(root))] for path in old_paths])
    except KeyError as error:
        raise ValueError("Original manifest lacks pinned raw episode hashes") from error
    remaining = set(ids) - set(old_train) - set(old_holdout)
    additional_holdout = sorted(sorted(remaining, key=lambda idx: hashlib.sha256(f"42:{idx}".encode()).hexdigest())[:6])
    holdout = sorted(old_holdout + additional_holdout)
    train = sorted(set(ids) - set(holdout))
    lengths = {ep["episode_index"]: ep["length"] for ep in selected}
    return {
        "source_repo": SOURCE, "source_revision": REVISION, "task": TASK, "seed": 42,
        "split_rule": "preserve original 16/4; rank remaining by sha256('42:episode_id'); first six holdout",
        "train_source_episode_ids": train, "holdout_source_episode_ids_excluded": holdout,
        "preserved_train_source_episode_ids": sorted(old_train),
        "old_holdout_source_episode_ids_excluded": sorted(old_holdout),
        "additional_holdout_source_episode_ids": additional_holdout,
        "train_episodes": len(train), "holdout_episodes": len(holdout),
        "train_frames": sum(lengths[idx] for idx in train),
        "holdout_frames": sum(lengths[idx] for idx in holdout),
        "missing_source_episode_ids": [idx for idx in ids if not source_path(root, info, idx).is_file()],
        "_info": info, "_lengths": lengths, "_task_index": matching_tasks[0]["task_index"],
    }


def validate_table(table, episode_id, count, task_index, schema):
    if len(table) != count or set(table.schema.names) != FIELDS or table.schema != schema:
        raise ValueError(f"Episode {episode_id}: source count or Arrow schema differs")
    for key, width in (("state", 8), ("actions", 7)):
        array = np.asarray(table[key].to_pylist(), dtype=np.float64)
        if array.shape != (count, width) or not np.isfinite(array).all():
            raise ValueError(f"Episode {episode_id}: invalid {key}")
    for key, expected in (("frame_index", np.arange(count)), ("episode_index", np.full(count, episode_id)),
                          ("task_index", np.full(count, task_index))):
        if not np.array_equal(np.asarray(table[key].to_pylist()), expected):
            raise ValueError(f"Episode {episode_id}: invalid {key}")
    indices = np.asarray(table["index"].to_pylist())
    timestamps = np.asarray(table["timestamp"].to_pylist())
    if (not np.array_equal(np.diff(indices), np.ones(count - 1)) or indices[0] < 0
            or not np.allclose(timestamps, np.arange(count) / 10, atol=1e-4, rtol=0)):
        raise ValueError(f"Episode {episode_id}: invalid source timestamps or global indices")
    for key in ("image", "wrist_image"):
        for value in table[key].to_pylist():
            if not isinstance(value, dict) or not isinstance(value.get("bytes"), bytes):
                raise ValueError(f"Episode {episode_id}: missing embedded {key}")
            with Image.open(io.BytesIO(value["bytes"])) as image:
                image.load()
                if image.mode != "RGB" or image.size != (256, 256):
                    raise ValueError(f"Episode {episode_id}: invalid RGB {key}")


def export_split(root, staging, ids, specification, schema, split):
    info, lengths = specification["_info"], specification["_lengths"]
    episodes, statistics, offset = [], [], 0
    for new_id, source_id in enumerate(ids):
        source = pq.read_table(source_path(root, info, source_id))
        count = lengths[source_id]
        validate_table(source, source_id, count, specification["_task_index"], schema)
        table = source
        for key, values in {"episode_index": [new_id] * count, "index": range(offset, offset + count),
                            "frame_index": range(count), "task_index": [0] * count}.items():
            position = table.schema.get_field_index(key)
            field = table.schema.field(position)
            table = table.set_column(position, field, pa.array(values, type=field.type))
        destination = staging / f"data/chunk-000/episode_{new_id:06d}.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, destination)
        reread = pq.read_table(destination)
        if not reread.equals(table, check_metadata=True):
            raise ValueError("Export Parquet roundtrip changed content")
        for key in ("image", "wrist_image", "state", "actions", "timestamp"):
            if not reread[key].equals(source[key]):
                raise ValueError(f"Export changed source {key}")
        episodes.append({"episode_index": new_id, "tasks": [TASK], "length": count})
        statistics.append({"episode_index": new_id, "stats": feature_stats(table, info["features"])})
        offset += count
        print(f"Prepared {split} {new_id + 1}/{len(ids)}: source {source_id}, {count} frames", flush=True)
    exported_info = dict(info, codebase_version="v2.1", total_episodes=len(ids), total_frames=offset,
                         total_tasks=1, total_videos=0, total_chunks=1, chunks_size=1000,
                         splits={"train": f"0:{len(ids)}"})
    write_json(staging / "meta/info.json", exported_info)
    write_jsonl(staging / "meta/tasks.jsonl", [{"task_index": 0, "task": TASK}])
    write_jsonl(staging / "meta/episodes.jsonl", episodes)
    write_jsonl(staging / "meta/episodes_stats.jsonl", statistics)


def prepare(root, name="public37_v1"):
    root, name = Path(root).resolve(), safe_name(name)
    repo_id = f"course/libero_corrections_{name}"
    outputs = [root / "data/lerobot" / repo_id, root / "data/lerobot" / f"{repo_id}_holdout"]
    manifest_path = root / "evidence" / f"corrections-{name}-manifest.json"
    if any(path.exists() for path in outputs + [manifest_path]):
        raise FileExistsError("Refusing to overwrite exported datasets or manifest")
    specification = plan(root)
    info = specification["_info"]
    if specification["missing_source_episode_ids"]:
        from huggingface_hub import hf_hub_download
        for idx in specification["missing_source_episode_ids"]:
            relative = source_path(root, info, idx).relative_to(root / "data/raw_public")
            print(f"Downloading pinned public episode {idx}", flush=True)
            hf_hub_download(SOURCE, str(relative), repo_type="dataset", revision=REVISION, local_dir=root / "data/raw_public")
    # Old inputs remain immutable, including their original cached checksums.
    plan(root)
    all_ids = sorted(specification["train_source_episode_ids"] + specification["holdout_source_episode_ids_excluded"])
    inputs = [file_record(root, root / "evidence/data_manifest.json")]
    inputs += [file_record(root, root / "data/raw_public/meta" / filename) for filename in META_FILES]
    inputs += [file_record(root, source_path(root, info, idx)) for idx in all_ids]
    norm = file_record(root, root / "assets/pi05_handson_lora" / ASSET_ID / "norm_stats.json")
    schema = pq.read_schema(source_path(root, info, all_ids[0]))
    if (set(schema.names) != FIELDS or schema.field("state").type != pa.list_(pa.float32(), 8)
            or schema.field("actions").type != pa.list_(pa.float32(), 7)):
        raise ValueError("Unexpected source Arrow feature schema")
    outputs[0].parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".expanded-public-stage-", dir=outputs[0].parent) as temporary:
        staged = [Path(temporary) / "train", Path(temporary) / "holdout"]
        for destination, ids, split in zip(staged, (specification["train_source_episode_ids"], specification["holdout_source_episode_ids_excluded"]), ("train", "holdout"), strict=True):
            export_split(root, destination, ids, specification, schema, split)
        manifest = {key: value for key, value in specification.items() if not key.startswith("_")}
        manifest.pop("missing_source_episode_ids")
        manifest.update({
            "dataset_kind": "expanded_public_v1", "ready": True, "project_root": str(root),
            "repo_id": repo_id, "dataset_dir": str(outputs[0].relative_to(root)),
            "holdout_repo_id": f"{repo_id}_holdout", "holdout_dataset_dir": str(outputs[1].relative_to(root)),
            "original_train_episodes": 37, "original_train_frames": specification["train_frames"],
            "correction_train_episodes": 0, "correction_train_frames": 0,
            "correction_holdout_episodes_excluded": 0, "human_corrections_included": False,
            "normalization": {"asset_id": ASSET_ID, "recomputed": False, **norm},
            "sampling": "uniform over public training frames", "scenarios": {"train": [], "holdout": []},
            "timing": {"nominal_fps": 10, "nominal_dt": .1, "pauses_excluded": True},
            "files": [{**file_record(stage, path), "path": str((output / path.relative_to(stage)).relative_to(root))}
                      for stage, output in zip(staged, outputs, strict=True) for path in sorted(stage.rglob("*")) if path.is_file()],
            "inputs": inputs,
        })
        verify_records(root, inputs + [norm])
        staged_manifest = Path(temporary) / "manifest.json"
        write_json(staged_manifest, manifest)
        # Reserve empty destinations exclusively; only our own reservations are replaced.
        reserved = []
        try:
            for output in outputs:
                output.mkdir()
                reserved.append(output)
            for stage, output in zip(staged, outputs, strict=True):
                stage.rename(output)
            os.link(staged_manifest, manifest_path)
        except BaseException:
            import shutil
            for output in reserved:
                shutil.rmtree(output)
            raise
    return manifest_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--name", default="public37_v1", type=safe_name)
    parser.add_argument("--plan", "--metadata-only", dest="plan_only", action="store_true")
    args = parser.parse_args(argv)
    if args.plan_only:
        print(json.dumps({key: value for key, value in plan(args.project_root).items() if not key.startswith("_")}, indent=2))
    else:
        print(prepare(args.project_root, args.name), flush=True)


if __name__ == "__main__":
    main()
