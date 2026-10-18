"""Append two physically validated tracking teachers without rewriting originals."""
import argparse
import io
import json
import os
from pathlib import Path
import shutil
import tempfile

from correction_data import MAX_FRAMES, digest, load_frames, validate_frame, write_json
from prepare_corrections import contained, file_record, jsonl, safe_name, verify_records, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
BASE_MANIFEST = "evidence/corrections-public37_reviewed19_v4-manifest.json"
KIND = "tracking_teacher_v1"
SAMPLING = "uniform over unchanged public/human and validated tracking teacher frames"


def assert_base(base):
    if (base.get("ready") is not True or base.get("dataset_kind") != "expanded_public_corrections_v1"
            or base.get("original_train_episodes") != 37 or base.get("correction_train_episodes") != 19
            or base.get("train_episodes") != 56 or base.get("holdout_episodes") != 10
            or len(set(base.get("correction_source_directories", []))) != 17
            or "training_sampling_profile" in base):
        raise ValueError("Expected the unchanged public37/reviewed19 base and ten excluded holdouts")


def load_base(root):
    from train_corrections import validate_expanded_corrections
    path = root / BASE_MANIFEST
    base = json.loads(path.read_text())
    assert_base(base)
    if base.get("project_root") != str(root):
        raise ValueError("Base project root differs")
    verify_records(root, base["files"] + base["inputs"] + [base["normalization"]])
    validate_expanded_corrections(root, base)
    return base


def relative_path(root, value):
    path = Path(value)
    return contained(root, str(path.relative_to(root)) if path.is_absolute() else str(path))


def load_teachers(root, directories, base):
    """Validate provenance and actual frame arrays; never grant human approval."""
    import numpy as np
    from generate_tracking_teacher import GATE_NAMES, quality_gates, reference_schedule
    if len(directories) != 2:
        raise ValueError("Both validated tracking teacher episodes are required")
    sources = {clip["source_directory"]: clip["metadata_sha256"] for clip in base["selected_source_ranges"]}
    result, seen, rates = [], set(), set()
    for value in directories:
        directory = relative_path(root, value)
        metadata_path = directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        if (metadata.get("generator_version") != 2 or metadata.get("feedback_scope") != "audited_carry_only"
                or metadata.get("noncarry_action_policy") != "exact_source_action"):
            raise ValueError("Teacher requires generator v2 with audited-carry-only feedback and exact noncarry actions")
        gates = metadata.get("quality_gates", {})
        rate = metadata.get("reference_rate")
        if (type(rate) is not int or rate not in (1, 2) or metadata.get("requested_reference_rate") != rate
                or metadata.get("role") != {1: "carry_rate1_control", 2: "carry_rate2_treatment"}.get(rate)):
            raise ValueError("Teacher reference rate or role differs")
        rates.add(rate)
        if (metadata.get("schema_version") != 1 or metadata.get("source_kind") != "offline_tracking_teacher"
                or metadata.get("human_actions_only") is not False or metadata.get("approval_type") is not None
                or metadata.get("ready") is not True or metadata.get("status") != "validated"
                or metadata.get("split") != "train" or metadata.get("task") != base["task"]
                or metadata.get("control_dt") != .05 or set(gates) != set(GATE_NAMES)
                or any(passed is not True for passed in gates.values())):
            raise ValueError("Teacher must be ready, validated, non-human, with passing quality gates")
        try:
            recomputed = quality_gates(metadata["quality_metrics"], rate)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Teacher quality metrics are missing or invalid") from error
        if recomputed != gates:
            raise ValueError("Teacher quality metrics contradict the declared gates")
        code_hashes = metadata.get("source_code_sha256", {})
        if (not code_hashes or any(not isinstance(v, str) or len(v) != 64 for v in code_hashes.values())):
            raise ValueError("Teacher generator source hashes are missing")
        source = relative_path(root, metadata["source_directory"])
        source_name = str(source.relative_to(root))
        if source_name in seen or sources.get(source_name) != metadata.get("source_metadata_sha256"):
            raise ValueError("Teacher source is duplicated or outside the retained training lineage")
        seen.add(source_name)
        original = json.loads((source / "metadata.json").read_text())
        if (digest(source / "metadata.json") != sources[source_name] or original.get("split") != "train"
                or original.get("status") != "approved" or metadata.get("scenario_key") != original["scenario_key"]
                or metadata.get("scenario") != original["scenario"]
                or metadata.get("source_frame_count") != original["frame_count"]
                or metadata.get("source_initial_sim_state_sha256") != original["initial_sim_state_sha256"]
                or metadata.get("initial_sim_state_sha256") != original["initial_sim_state_sha256"]):
            raise ValueError("Teacher source identity, scenario or split differs")
        original_frames = load_frames(source, original)
        count, records = metadata.get("frame_count"), metadata.get("files", [])
        if type(count) is not int or not 10 <= count <= MAX_FRAMES or len(records) != count:
            raise ValueError("Teacher frame count differs")
        names = [f"frame-{i:06d}.npz" for i in range(count)]
        if sorted(p.name for p in directory.glob("frame-*.npz")) != names:
            raise ValueError("Teacher frame inventory differs")
        frames = []
        for name, record in zip(names, records, strict=True):
            path = directory / name
            if path.is_symlink() or record.get("path") != name or digest(path) != record.get("sha256"):
                raise ValueError("Teacher frame checksum differs")
            with np.load(path, allow_pickle=False) as archive:
                frame = {key: archive[key] for key in archive.files}
            validate_frame(frame)
            frames.append(frame)
        tail = frames[-10:]
        if not all(bool(f["done"]) and f["action"][6] < 0
                   and np.abs(f["post_state"][6:8]).sum() >= .06
                   and np.linalg.norm(f["post_state"][:3] - f["post_bowl_pos"]) >= .10 for f in tail):
            raise ValueError("Teacher actual ten-step release/retreat gate failed")
        if any(not np.allclose(a["post_state"], b["state"], rtol=0, atol=1e-5)
               for a, b in zip(frames, frames[1:])):
            raise ValueError("Teacher observations are not a continuous physical episode")
        initial = np.load(directory / "initial_sim_state.npy", allow_pickle=False)
        if (initial.ndim != 1 or not np.isfinite(initial).all()
                or digest_bytes(initial.tobytes()) != metadata.get("initial_sim_state_sha256")):
            raise ValueError("Teacher initial simulation state checksum differs")
        reference = np.load(directory / "reference-index.npy", allow_pickle=False)
        schedule = reference_schedule(original["frame_count"], metadata["reference_carry_interval"],
                                      protected=metadata["protected_source_steps"], rate=rate)
        if not np.array_equal(reference, [index for index, _ in schedule]) or len(schedule) != count:
            raise ValueError("Teacher reference progress does not cover one continuous source")
        start, stop = metadata["reference_carry_interval"]
        for frame, index in zip(frames, reference, strict=True):
            if not start <= index < stop:
                expected = np.asarray(original_frames[int(index)]["action"], dtype=np.float32)
                if np.asarray(frame["action"], dtype=np.float32).tobytes() != expected.tobytes():
                    raise ValueError("Teacher noncarry action differs from the exact source action")
        result.append((directory, metadata, frames))
    if len(rates) != 1:
        raise ValueError("Both teacher sources must have the same reference rate and role")
    return result


def digest_bytes(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()


def frame_table(frames, schema, info, episode, offset):
    import pyarrow as pa
    from PIL import Image
    rows = []
    for i, frame in enumerate(frames):
        row = {"state": frame["state"].tolist(), "actions": frame["action"].tolist(),
               "timestamp": i / info["fps"], "frame_index": i, "episode_index": episode,
               "index": offset + i, "task_index": 0}
        for key in ("image", "wrist_image"):
            stream = io.BytesIO()
            Image.fromarray(frame[key]).save(stream, format="PNG")
            row[key] = {"bytes": stream.getvalue(), "path": None}
        rows.append(row)
    return pa.Table.from_pylist(rows, schema=schema)


def records_for(root, directories):
    return [file_record(root, path) for directory in directories
            for path in sorted(directory.rglob("*")) if path.is_file()]


def validate_export(root, manifest, base):
    """Check the complete new export after the caller has validated the human base."""
    import pyarrow.parquet as pq
    from prepare_public_data import feature_stats
    assert_base(base)
    if (manifest.get("dataset_kind") != KIND or manifest.get("ready") is not True
            or manifest.get("project_root") != str(root) or manifest.get("sampling") != SAMPLING
            or manifest.get("human_actions_only") is not False or manifest.get("approval_type") is not None
            or manifest.get("human_base_approval_type") != base["approval_type"]
            or manifest.get("teacher_source_kind") != "offline_tracking_teacher"
            or manifest.get("base_manifest") != file_record(root, root / BASE_MANIFEST)):
        raise ValueError("Tracking teacher export identity or base checksum differs")
    for key in ("task", "normalization", "selected_source_ranges", "correction_source_directories",
                "original_train_episodes", "original_train_frames", "correction_train_episodes",
                "correction_train_frames", "holdout_episodes", "holdout_frames", "seed",
                "train_source_episode_ids", "holdout_source_episode_ids_excluded", "scenarios"):
        if manifest.get(key) != base.get(key):
            raise ValueError(f"Tracking teacher changed inherited base {key}")
    verify_records(root, manifest["files"] + manifest["inputs"])
    teachers = load_teachers(root, [item["directory"] for item in manifest["teachers"]], base)
    inputs = base["files"] + base["inputs"] + [file_record(root, root / BASE_MANIFEST)]
    inputs += records_for(root, [directory for directory, _, _ in teachers])
    if {r["path"]: r for r in manifest["inputs"]} != {r["path"]: r for r in inputs}:
        raise ValueError("Tracking teacher input inventory differs")
    original, dataset = contained(root, base["dataset_dir"]), contained(root, manifest["dataset_dir"])
    actual = records_for(root, [dataset])
    if manifest["files"] != actual:
        raise ValueError("Tracking teacher dataset inventory differs")
    info = json.loads((dataset / "meta/info.json").read_text())
    base_info = json.loads((original / "meta/info.json").read_text())
    episodes, stats = jsonl(original / "meta/episodes.jsonl"), jsonl(original / "meta/episodes_stats.jsonl")
    for path in original.rglob("*"):
        if path.is_file() and path.name not in ("info.json", "episodes.jsonl", "episodes_stats.jsonl"):
            if digest(path) != digest(dataset / path.relative_to(original)):
                raise ValueError("Original public/human training prefix changed")
    schema = pq.read_schema(original / base_info["data_path"].format(episode_chunk=0, episode_index=0))
    offset = base["train_frames"]
    expected_files = {str(path.relative_to(original)) for path in original.rglob("*") if path.is_file()}
    expected_teachers = []
    for i, (directory, metadata, frames) in enumerate(teachers, start=base["train_episodes"]):
        relative = info["data_path"].format(episode_chunk=i // info["chunks_size"], episode_index=i)
        expected_files.add(relative)
        table = frame_table(frames, schema, info, i, offset)
        if not pq.read_table(dataset / relative).equals(table, check_metadata=True):
            raise ValueError("Teacher exported schema/state/action/RGB/indices differ from actual frames")
        episodes.append({"episode_index": i, "tasks": [base["task"]], "length": len(frames)})
        stats.append({"episode_index": i, "stats": feature_stats(table, info["features"])})
        expected_teachers.append({"directory": str(directory.relative_to(root)),
                                  "metadata_sha256": digest(directory / "metadata.json"),
                                  "source_directory": str(relative_path(root, metadata["source_directory"]).relative_to(root)),
                                  "episode_index": i, "frame_count": len(frames)})
        offset += len(frames)
    expected_info = {**base_info, "total_episodes": len(episodes), "total_frames": offset,
                     "total_chunks": (len(episodes) - 1) // base_info["chunks_size"] + 1,
                     "splits": {"train": f"0:{len(episodes)}"}}
    expected_timing = {**base["timing"], "teacher_physical_control_dt": .05,
                       "teacher_timestamp_semantics": "10Hz nominal index grid; each real native control step is20Hz, without resampling"}
    if (info != expected_info or manifest.get("teachers") != expected_teachers
            or manifest.get("train_episodes") != len(episodes) or manifest.get("train_frames") != offset
            or manifest.get("teacher_train_episodes") != 2
            or manifest.get("teacher_train_frames") != offset - base["train_frames"]
            or manifest.get("timing") != expected_timing
            or jsonl(dataset / "meta/episodes.jsonl") != episodes
            or jsonl(dataset / "meta/episodes_stats.jsonl") != stats
            or {str(p.relative_to(dataset)) for p in dataset.rglob("*") if p.is_file()} != expected_files):
        raise ValueError("Tracking teacher export counts, statistics or episode inventory differ")
    return teachers


def prepare(root, name, teacher_directories):
    root, name = Path(root).resolve(), safe_name(name)
    repo = f"course/libero_tracking_teacher_{name}"
    output = root / "data/lerobot" / repo
    manifest_path = root / "evidence" / f"tracking-teacher-{name}-manifest.json"
    if output.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite tracking teacher export")
    base = load_base(root)
    teachers = load_teachers(root, teacher_directories, base)
    import pyarrow.parquet as pq
    from prepare_public_data import feature_stats
    original = contained(root, base["dataset_dir"])
    info = json.loads((original / "meta/info.json").read_text())
    schema = pq.read_schema(original / info["data_path"].format(episode_chunk=0, episode_index=0))
    inputs = base["files"] + base["inputs"] + [file_record(root, root / BASE_MANIFEST)]
    inputs += records_for(root, [directory for directory, _, _ in teachers])
    inputs = list({r["path"]: r for r in inputs}.values())
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tracking-stage-", dir=output.parent) as temporary:
        staging = Path(temporary) / "dataset"
        shutil.copytree(original, staging)
        episodes, stats = jsonl(staging / "meta/episodes.jsonl"), jsonl(staging / "meta/episodes_stats.jsonl")
        offset, identities = base["train_frames"], []
        for i, (directory, metadata, frames) in enumerate(teachers, start=base["train_episodes"]):
            table = frame_table(frames, schema, info, i, offset)
            relative = info["data_path"].format(episode_chunk=i // info["chunks_size"], episode_index=i)
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, destination)
            if not pq.read_table(destination).equals(table, check_metadata=True):
                raise ValueError("Teacher Arrow roundtrip changed content")
            episodes.append({"episode_index": i, "tasks": [base["task"]], "length": len(frames)})
            stats.append({"episode_index": i, "stats": feature_stats(table, info["features"])})
            identities.append({"directory": str(directory.relative_to(root)), "metadata_sha256": digest(directory / "metadata.json"),
                               "source_directory": str(relative_path(root, metadata["source_directory"]).relative_to(root)),
                               "episode_index": i, "frame_count": len(frames)})
            offset += len(frames)
        info.update(total_episodes=len(episodes), total_frames=offset,
                    total_chunks=(len(episodes) - 1) // info["chunks_size"] + 1, splits={"train": f"0:{len(episodes)}"})
        write_json(staging / "meta/info.json", info)
        write_jsonl(staging / "meta/episodes.jsonl", episodes)
        write_jsonl(staging / "meta/episodes_stats.jsonl", stats)
        files = [{**file_record(staging, p), "path": str((output / p.relative_to(staging)).relative_to(root))}
                 for p in sorted(staging.rglob("*")) if p.is_file()]
        manifest = {**base, "dataset_kind": KIND, "repo_id": repo, "dataset_dir": str(output.relative_to(root)),
                    "base_manifest": file_record(root, root / BASE_MANIFEST), "sampling": SAMPLING,
                    "approval_type": None, "human_base_approval_type": base["approval_type"],
                    "human_actions_only": False, "teacher_source_kind": "offline_tracking_teacher",
                    "train_episodes": len(episodes), "train_frames": offset,
                    "teacher_train_episodes": 2, "teacher_train_frames": offset - base["train_frames"],
                    "teachers": identities, "files": files, "inputs": inputs,
                    "timing": {**base["timing"], "teacher_physical_control_dt": .05,
                               "teacher_timestamp_semantics": "10Hz nominal index grid; each real native control step is20Hz, without resampling"}}
        # Recheck immutable inputs before publication; staged prefix is byte-identical.
        verify_records(root, inputs)
        for path in original.rglob("*.parquet"):
            if digest(path) != digest(staging / path.relative_to(original)):
                raise ValueError("Original training episode changed")
        staged_manifest = Path(temporary) / "manifest.json"
        write_json(staged_manifest, manifest)
        staging.rename(output)
        os.link(staged_manifest, manifest_path)
    return manifest_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--name", required=True, type=safe_name)
    parser.add_argument("--teachers", nargs=2, required=True, type=Path)
    args = parser.parse_args(argv)
    print(prepare(args.project_root, args.name, args.teachers))


if __name__ == "__main__":
    main()
