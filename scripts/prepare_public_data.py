#!/usr/bin/env python3
"""Prepare a pinned public LIBERO subset; never upload or read company data."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

SOURCE = "physical-intelligence/libero"
REVISION = "a4336d589d589045d1c56423ffdf3b88a0e19b1f"
DESTINATION = "course/libero_black_bowl"
TASK = "pick up the black bowl next to the plate and place it on the plate"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values))


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def feature_stats(table, features):
    """LeRobot v2.1 stats schema: numeric (D,), images (C,1,1), count (1,)."""
    result = {}
    for key, feature in features.items():
        if feature["dtype"] == "string":
            continue
        values = table[key].to_pylist()
        if feature["dtype"] == "image":
            minimum, maximum = np.full(3, np.inf), np.full(3, -np.inf)
            sums, squares = np.zeros(3), np.zeros(3)
            pixels = 0
            for value in values:
                if not isinstance(value, dict) or value.get("bytes") is None:
                    raise ValueError(f"Expected embedded image bytes in {key}")
                with Image.open(io.BytesIO(value["bytes"])) as image:
                    array = np.asarray(image.convert("RGB"), dtype=np.float64).reshape(-1, 3) / 255
                minimum = np.minimum(minimum, array.min(axis=0))
                maximum = np.maximum(maximum, array.max(axis=0))
                sums += array.sum(axis=0)
                squares += np.square(array).sum(axis=0)
                pixels += len(array)
            mean = sums / pixels
            stats = dict(min=minimum, max=maximum, mean=mean,
                         std=np.sqrt(np.maximum(0, squares / pixels - mean * mean)))
            stats = {k: v.reshape(3, 1, 1).tolist() for k, v in stats.items()}
        else:
            array = np.asarray(values, dtype=np.float64)
            if not np.isfinite(array).all():
                raise ValueError(f"Nonfinite values in {key}")
            stats = {name: np.atleast_1d(value).tolist() for name, value in {
                "min": array.min(axis=0), "max": array.max(axis=0),
                "mean": array.mean(axis=0), "std": array.std(axis=0),
            }.items()}
        stats["count"] = [len(values)]
        result[key] = stats
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    raw = root / "data/raw_public"
    raw.mkdir(parents=True, exist_ok=True)
    for filename in ("info.json", "tasks.jsonl", "episodes.jsonl"):
        hf_hub_download(SOURCE, "meta/" + filename, repo_type="dataset",
                        revision=REVISION, local_dir=raw)
    info = json.loads((raw / "meta/info.json").read_text())
    episodes = [json.loads(line) for line in (raw / "meta/episodes.jsonl").read_text().splitlines()]
    selected = sorted((ep for ep in episodes if ep["tasks"] == [TASK]),
                      key=lambda ep: ep["episode_index"])
    if len(selected) < 20:
        raise ValueError(f"Only {len(selected)} episodes found for exact task")
    # First 16 train; last four from the matching task are held out completely.
    train, holdout = selected[:16], selected[-4:]
    manifest = dict(source_repo=SOURCE, source_revision=REVISION, source_format=info["codebase_version"],
                    destination_repo=DESTINATION, destination_format="v2.1", task=TASK,
                    fps=info["fps"], train_source_episode_ids=[e["episode_index"] for e in train],
                    holdout_source_episode_ids=[e["episode_index"] for e in holdout],
                    train_episodes=len(train), holdout_episodes=len(holdout),
                    train_frames=sum(e["length"] for e in train),
                    holdout_frames=sum(e["length"] for e in holdout),
                    image_stats="all embedded RGB pixels, population std, [0,1] scale",
                    ready=False)
    print(json.dumps(manifest, indent=2), flush=True)
    if args.metadata_only:
        return
    output = root / "data/lerobot" / DESTINATION
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing dataset: {output}")
    for episode in train + holdout:
        idx = episode["episode_index"]
        filename = info["data_path"].format(episode_chunk=idx // info["chunks_size"], episode_index=idx)
        hf_hub_download(SOURCE, filename, repo_type="dataset", revision=REVISION, local_dir=raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".libero-stage-", dir=output.parent) as temporary:
        staging = Path(temporary) / "dataset"
        episode_records, stats_records = [], []
        offset = 0
        for new_id, episode in enumerate(train):
            old_id = episode["episode_index"]
            source_path = raw / info["data_path"].format(
                episode_chunk=old_id // info["chunks_size"], episode_index=old_id)
            table = pq.read_table(source_path)
            count = len(table)
            if count != episode["length"]:
                raise ValueError(f"Episode {old_id}: frame count disagrees with metadata")
            for key, values in {"episode_index": [new_id] * count, "index": range(offset, offset + count),
                                "frame_index": range(count), "task_index": [0] * count}.items():
                position = table.schema.get_field_index(key)
                if position < 0:
                    raise ValueError(f"Missing index field {key}")
                field = table.schema.field(position)
                table = table.set_column(position, field, pa.array(values, type=field.type))
            timestamps = np.asarray(table["timestamp"].to_pylist()).reshape(-1)
            if not np.allclose(np.diff(timestamps), 1 / info["fps"], atol=1e-4, rtol=0):
                raise ValueError(f"Episode {old_id}: inconsistent timestamps")
            relative = f"data/chunk-000/episode_{new_id:06d}.parquet"
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, destination)
            # Arrow index replacement leaves RGB bytes, state and actions untouched.
            reread = pq.read_table(destination)
            for key in ("image", "wrist_image", "state", "actions"):
                if not reread[key].equals(table[key]):
                    raise ValueError(f"Parquet roundtrip changed {key}")
            stats_records.append(dict(episode_index=new_id, stats=feature_stats(table, info["features"])))
            episode_records.append(dict(episode_index=new_id, tasks=[TASK], length=count))
            offset += count
            print(f"Prepared {new_id + 1}/16: source episode {old_id}, {count} frames", flush=True)
        new_info = dict(info)
        new_info.update(codebase_version="v2.1", total_episodes=16, total_frames=offset,
                        total_tasks=1, total_videos=0, total_chunks=1, chunks_size=1000,
                        splits={"train": "0:16"})
        write_json(staging / "meta/info.json", new_info)
        write_jsonl(staging / "meta/tasks.jsonl", [dict(task_index=0, task=TASK)])
        write_jsonl(staging / "meta/episodes.jsonl", episode_records)
        write_jsonl(staging / "meta/episodes_stats.jsonl", stats_records)
        staging.rename(output)
    manifest.update(ready=True, destination=str(output.relative_to(root)))
    manifest["files"] = [dict(path=str(path.relative_to(root)), bytes=path.stat().st_size,
                              sha256=digest(path))
                         for directory in (raw, output) for path in sorted(directory.rglob("*"))
                         if path.is_file() and ".cache" not in path.parts]
    write_json(root / "evidence/data_manifest.json", manifest)
    print(f"READY: {output}; manifest: {root / 'evidence/data_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
