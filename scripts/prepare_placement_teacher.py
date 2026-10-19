"""Append validated, explicitly non-human placement teachers to the unchanged human base."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from correction_data import digest, load_frames, validate_frame, write_json
from prepare_corrections import contained, file_record, jsonl, safe_name, verify_records, write_jsonl
from prepare_tracking_teacher import (BASE_MANIFEST, digest_bytes, frame_table, load_base,
                                      records_for, relative_path)

ROOT = Path(__file__).resolve().parents[1]
KIND = "placement_teacher_v1"
SOURCE_KINDS = {"offline_placement_recovery_teacher", "offline_retargeted_task_teacher"}
REQUIRED_GATES = {"terminal_ten_on_open_gap_retreat", "terminal_sixty_on_open_gap_retreat",
                  "fifty_followup_steps", "followup_all_released", "followup_bowl_motion_le_5mm",
                  "continuous_physics", "no_unexpected_deep_target_contact", "finite_bounded_actions"}
INHERITED = ("task", "normalization", "selected_source_ranges", "correction_source_directories",
             "original_train_episodes", "original_train_frames", "correction_train_episodes",
             "correction_train_frames", "holdout_episodes", "holdout_frames", "seed",
             "train_source_episode_ids", "holdout_source_episode_ids_excluded", "scenarios")


def validate_trajectory(frames):
    if not 60 <= len(frames) <= 800:
        raise ValueError("Teacher requires sixty stable terminal frames within 800 total actions")
    for frame in frames:
        validate_frame(frame)
    if any(not np.allclose(a["post_state"], b["state"], rtol=0, atol=1e-5)
           for a, b in zip(frames, frames[1:])):
        raise ValueError("Teacher observations must form one continuous physical episode")
    if not all(bool(f["done"]) and f["action"][6] < 0
               and np.abs(f["post_state"][6:8]).sum() >= .06
               and np.linalg.norm(f["post_state"][:3] - f["post_bowl_pos"]) >= .10
               for f in frames[-60:]):
        raise ValueError("Teacher requires sixty consecutive On, actual-open, open-command and retreat frames")
    if max(np.linalg.norm(f["post_bowl_pos"] - frames[-51]["post_bowl_pos"])
           for f in frames[-50:]) > .005:
        raise ValueError("Teacher bowl drift exceeds 5mm during fifty-step followup")


def load_teacher(root, value, base):
    directory = relative_path(root, value)
    metadata = json.loads((directory / "metadata.json").read_text())
    if (metadata.get("schema_version") != 1 or metadata.get("source_kind") not in SOURCE_KINDS
            or metadata.get("human_actions_only") is not False or metadata.get("approval_type") is not None
            or metadata.get("ready") is not True or metadata.get("status") != "validated"
            or metadata.get("split") != "train" or metadata.get("task") != base["task"]
            or metadata.get("control_dt") != .05):
        raise ValueError("Expected a ready, validated, explicitly non-human placement teacher")
    gates = metadata.get("quality_gates")
    if (not isinstance(gates, dict) or not REQUIRED_GATES <= gates.keys()
            or not all(value is True for value in gates.values())):
        raise ValueError("Teacher quality gates have not passed")
    source_scene = relative_path(root, metadata["source_scene"])
    scene = json.loads((source_scene / "scene.json").read_text())
    if scene.get("split") not in (None, "train"):
        raise ValueError("Held-out scenes cannot become training teachers")
    if (scene.get("layout_id") != 24 or scene.get("task_language") != base["task"]
            or scene.get("task_metadata") != metadata.get("task_metadata")
            or metadata.get("scenario") != scene):
        raise ValueError("Teacher source scenario/task metadata differs")
    source_files = metadata.get("source_files", {})
    if not source_files or any(digest(contained(source_scene, name)) != sha for name, sha in source_files.items()):
        raise ValueError("Teacher source evidence hashes differ")
    if metadata["source_kind"] == "offline_retargeted_task_teacher":
        reference = relative_path(root, metadata["reference_scene"])
        files = metadata.get("reference_source_files", {})
        if (not {"scene.json", "result.json", "actions.csv"} <= files.keys()
                or any(digest(contained(reference, name)) != sha for name, sha in files.items())
                or json.loads((reference / "result.json").read_text()).get("success") is not True):
            raise ValueError("Full-task teacher successful geometry-reference hashes differ")
    source_state = relative_path(root, metadata["source_state_file"])
    expected = "final_state.npy" if metadata["source_kind"] == "offline_placement_recovery_teacher" else "initial_state.npy"
    if source_state != source_scene / expected or digest(source_state) != metadata.get("source_state_file_sha256"):
        raise ValueError("Teacher physical source state identity differs")
    initial = np.load(directory / "initial_sim_state.npy", allow_pickle=False)
    if (initial.ndim != 1 or not np.isfinite(initial).all()
            or not np.array_equal(initial, np.load(source_state, allow_pickle=False))
            or digest_bytes(initial.tobytes()) != metadata.get("initial_sim_state_sha256")):
        raise ValueError("Teacher initial physical state differs from its declared source")
    hashes = metadata.get("source_code_sha256", {})
    generator = {"offline_placement_recovery_teacher": "generate_placement_recovery_teacher.py",
                 "offline_retargeted_task_teacher": "generate_retargeted_task_teacher.py"}[metadata["source_kind"]]
    if (not {generator, "generate_tracking_teacher.py", "simulation_lab.py"} <= hashes.keys()
            or any(digest(contained(root, f"scripts/{name}")) != sha for name, sha in hashes.items())):
        raise ValueError("Teacher generator source hashes differ")
    frames = load_frames(directory, metadata)
    validate_trajectory(frames)
    verified = metadata.get("first_ten_step_release_retreat_control_steps")
    if type(verified) is not int or verified < 10 or len(frames)-verified != 50:
        raise ValueError("Teacher must include exactly fifty followup steps after the first ten-step release gate")
    streak, first_verified = 0, None
    for i, frame in enumerate(frames, 1):
        release = (bool(frame["done"]) and frame["action"][6] < 0
                   and np.abs(frame["post_state"][6:8]).sum() >= .06
                   and np.linalg.norm(frame["post_state"][:3] - frame["post_bowl_pos"]) >= .10)
        streak = streak + 1 if release else 0
        if streak == 10:
            first_verified = i
            break
    if verified != first_verified:
        raise ValueError("Teacher first ten-step release gate contradicts actual frames")
    metrics = metadata.get("quality_metrics", {})
    if not np.isclose(metrics.get("simulation_duration_s", -1), len(frames)*.05, rtol=0, atol=1e-7):
        raise ValueError("Teacher physical duration differs from native20Hz frame count")
    return directory, metadata, frames


def load_teachers(root, directories, base):
    if not directories or len(set(str(Path(p).resolve()) for p in directories)) != len(directories):
        raise ValueError("Teacher sources must be nonempty and unique")
    return [load_teacher(root, value, base) for value in directories]


def identity(root, directory, metadata, frames, episode):
    return {"directory": str(directory.relative_to(root)), "metadata_sha256": digest(directory / "metadata.json"),
            "source_kind": metadata["source_kind"], "source_scene": metadata["source_scene"],
            "episode_index": episode, "frame_count": len(frames)}


def inputs_for(root, base, teachers):
    inputs = base["files"] + base["inputs"] + [file_record(root, root / BASE_MANIFEST)]
    inputs += records_for(root, [directory for directory, _, _ in teachers])
    for _, metadata, _ in teachers:
        source_scene = relative_path(root, metadata["source_scene"])
        inputs += [file_record(root, source_scene / "scene.json"),
                   file_record(root, relative_path(root, metadata["source_state_file"]))]
        inputs += [file_record(root, contained(source_scene, name)) for name in metadata["source_files"]]
        if metadata["source_kind"] == "offline_retargeted_task_teacher":
            reference = relative_path(root, metadata["reference_scene"])
            inputs += [file_record(root, contained(reference, name)) for name in metadata["reference_source_files"]]
    return list({record["path"]: record for record in inputs}.values())


def validate_export(root, manifest, base):
    import pyarrow.parquet as pq
    from prepare_public_data import feature_stats
    if (manifest.get("dataset_kind") != KIND or manifest.get("ready") is not True
            or manifest.get("project_root") != str(root) or manifest.get("human_actions_only") is not False
            or manifest.get("approval_type") is not None
            or manifest.get("human_base_approval_type") != base["approval_type"]
            or manifest.get("base_manifest") != file_record(root, root / BASE_MANIFEST)
            or any(manifest.get(key) != base.get(key) for key in INHERITED)):
        raise ValueError("Placement export identity or unchanged human/public base differs")
    verify_records(root, manifest["files"] + manifest["inputs"] + [manifest["normalization"]])
    teachers = load_teachers(root, [item["directory"] for item in manifest["teachers"]], base)
    if manifest["inputs"] != inputs_for(root, base, teachers):
        raise ValueError("Placement export input inventory differs")
    original, dataset = contained(root, base["dataset_dir"]), contained(root, manifest["dataset_dir"])
    if manifest["files"] != records_for(root, [dataset]):
        raise ValueError("Placement export file inventory differs")
    info = json.loads((original / "meta/info.json").read_text())
    schema = pq.read_schema(original / info["data_path"].format(episode_chunk=0, episode_index=0))
    episodes, stats = jsonl(original / "meta/episodes.jsonl"), jsonl(original / "meta/episodes_stats.jsonl")
    expected_files = {str(p.relative_to(original)) for p in original.rglob("*") if p.is_file()}
    for path in original.rglob("*"):
        if path.is_file() and path.name not in ("info.json", "episodes.jsonl", "episodes_stats.jsonl"):
            if digest(path) != digest(dataset / path.relative_to(original)):
                raise ValueError("Original public/human training prefix changed")
    offset, identities = base["train_frames"], []
    for i, (directory, metadata, frames) in enumerate(teachers, base["train_episodes"]):
        relative = info["data_path"].format(episode_chunk=i // info["chunks_size"], episode_index=i)
        table = frame_table(frames, schema, info, i, offset)
        if not pq.read_table(dataset / relative).equals(table, check_metadata=True):
            raise ValueError("Teacher exported RGB/state/action/window indices differ from actual frames")
        expected_files.add(relative)
        episodes.append({"episode_index": i, "tasks": [base["task"]], "length": len(frames)})
        stats.append({"episode_index": i, "stats": feature_stats(table, info["features"])})
        identities.append(identity(root, directory, metadata, frames, i))
        offset += len(frames)
    expected_info = {**info, "total_episodes": len(episodes), "total_frames": offset,
                     "total_chunks": (len(episodes)-1)//info["chunks_size"]+1,
                     "splits": {"train": f"0:{len(episodes)}"}}
    if (json.loads((dataset / "meta/info.json").read_text()) != expected_info
            or jsonl(dataset / "meta/episodes.jsonl") != episodes
            or jsonl(dataset / "meta/episodes_stats.jsonl") != stats
            or {str(p.relative_to(dataset)) for p in dataset.rglob("*") if p.is_file()} != expected_files
            or manifest.get("teachers") != identities or manifest.get("train_frames") != offset
            or manifest.get("train_episodes") != len(episodes)
            or manifest.get("teacher_train_episodes") != len(teachers)
            or manifest.get("teacher_train_frames") != offset-base["train_frames"]):
        raise ValueError("Placement export counts, statistics or boundaries differ")
    return teachers


def prepare(root, name, directories):
    import pyarrow.parquet as pq
    from prepare_public_data import feature_stats
    root, name = Path(root).resolve(), safe_name(name)
    repo = f"course/libero_placement_teacher_{name}"
    output = root / "data/lerobot" / repo
    manifest_path = root / "evidence" / f"placement-teacher-{name}-manifest.json"
    if output.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite placement teacher export")
    base = load_base(root)
    teachers = load_teachers(root, directories, base)
    inputs = inputs_for(root, base, teachers)
    original = contained(root, base["dataset_dir"])
    info = json.loads((original / "meta/info.json").read_text())
    schema = pq.read_schema(original / info["data_path"].format(episode_chunk=0, episode_index=0))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".placement-stage-", dir=output.parent) as temporary:
        staging = Path(temporary) / "dataset"
        shutil.copytree(original, staging)
        episodes, stats = jsonl(staging / "meta/episodes.jsonl"), jsonl(staging / "meta/episodes_stats.jsonl")
        offset, identities = base["train_frames"], []
        for i, (directory, metadata, frames) in enumerate(teachers, base["train_episodes"]):
            table = frame_table(frames, schema, info, i, offset)
            relative = info["data_path"].format(episode_chunk=i // info["chunks_size"], episode_index=i)
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, destination)
            if not pq.read_table(destination).equals(table, check_metadata=True):
                raise ValueError("Teacher Arrow roundtrip changed content")
            episodes.append({"episode_index": i, "tasks": [base["task"]], "length": len(frames)})
            stats.append({"episode_index": i, "stats": feature_stats(table, info["features"])})
            identities.append(identity(root, directory, metadata, frames, i))
            offset += len(frames)
        info.update(total_episodes=len(episodes), total_frames=offset,
                    total_chunks=(len(episodes)-1)//info["chunks_size"]+1, splits={"train": f"0:{len(episodes)}"})
        write_json(staging / "meta/info.json", info)
        write_jsonl(staging / "meta/episodes.jsonl", episodes)
        write_jsonl(staging / "meta/episodes_stats.jsonl", stats)
        files = [{**file_record(staging, p), "path": str((output / p.relative_to(staging)).relative_to(root))}
                 for p in sorted(staging.rglob("*")) if p.is_file()]
        manifest = {**base, "dataset_kind": KIND, "repo_id": repo, "dataset_dir": str(output.relative_to(root)),
                    "base_manifest": file_record(root, root / BASE_MANIFEST), "sampling": "uniform over unchanged base and validated placement teachers",
                    "approval_type": None, "human_base_approval_type": base["approval_type"], "human_actions_only": False,
                    "train_episodes": len(episodes), "train_frames": offset, "teacher_train_episodes": len(teachers),
                    "teacher_train_frames": offset-base["train_frames"], "teachers": identities, "files": files, "inputs": inputs,
                    "timing": {**base["timing"], "teacher_physical_control_dt": .05,
                               "teacher_timestamp_semantics": "10Hz nominal index grid; real native control steps20Hz, no resampling"}}
        verify_records(root, inputs)
        for path in original.rglob("*.parquet"):
            if digest(path) != digest(staging / path.relative_to(original)):
                raise ValueError("Original training episode changed")
        staged_manifest = Path(temporary) / "manifest.json"
        write_json(staged_manifest, manifest)
        staging.rename(output)
        os.link(staged_manifest, manifest_path)
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, type=safe_name)
    parser.add_argument("--teachers", nargs="+", required=True, type=Path)
    args = parser.parse_args()
    print(prepare(ROOT, args.name, args.teachers))


if __name__ == "__main__":
    main()
