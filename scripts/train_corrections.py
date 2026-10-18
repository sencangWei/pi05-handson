"""Start a separate course fine-tune from an immutable correction dataset."""
import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess

from correction_data import APPROVAL, approval_checks, digest, load_frames
from prepare_corrections import ASSET_ID, contained, jsonl, load_selection, safe_name, selected_frames, verify_records
from train_release_focus import positive_int, positive_learning_rate

ROOT = Path(__file__).resolve().parents[1]


def validate_expanded_public(root, manifest):
    """Accept only the pinned 37/10 public export, with no human trajectories."""
    source = {"source_repo": "physical-intelligence/libero",
              "source_revision": "a4336d589d589045d1c56423ffdf3b88a0e19b1f",
              "task": "pick up the black bowl next to the plate and place it on the plate"}
    pinned = json.loads((root / "evidence/data_manifest.json").read_text())
    if any(manifest.get(key) != value or pinned.get(key) != value for key, value in source.items()):
        raise ValueError("Expanded public source revision or task differs")
    train, holdout = manifest["train_source_episode_ids"], manifest["holdout_source_episode_ids_excluded"]
    old_train, old_holdout = pinned["train_source_episode_ids"], pinned["holdout_source_episode_ids"]
    if (pinned.get("ready") is not True or len(set(old_train)) != 16 or len(set(old_holdout)) != 4
            or set(old_train) & set(old_holdout) or manifest.get("seed") != 42
            or len(train) != 37 or len(set(train)) != 37 or len(holdout) != 10 or len(set(holdout)) != 10
            or any(type(idx) is not int for idx in train + holdout)
            or train != sorted(train) or holdout != sorted(holdout) or set(train) & set(holdout)
            or manifest.get("preserved_train_source_episode_ids") != sorted(old_train)
            or manifest.get("old_holdout_source_episode_ids_excluded") != sorted(old_holdout)
            or not set(old_train) <= set(train) or not set(old_holdout) <= set(holdout)):
        raise ValueError("Expanded public training and holdout sources overlap or lineage differs")
    if (manifest.get("original_train_episodes") != 37 or manifest.get("train_episodes") != 37
            or manifest.get("holdout_episodes") != 10 or manifest.get("correction_train_episodes") != 0
            or manifest.get("correction_train_frames") != 0 or manifest.get("correction_holdout_episodes_excluded") != 0
            or manifest.get("human_corrections_included") is not False
            or manifest.get("scenarios") != {"train": [], "holdout": []}
            or manifest.get("sampling") != "uniform over public training frames"):
        raise ValueError("Expanded public export must contain 37/10 public episodes and no human corrections")
    raw = root / "data/raw_public"
    info = json.loads((raw / "meta/info.json").read_text())
    source_episodes = {row["episode_index"]: row for row in jsonl(raw / "meta/episodes.jsonl")
                       if row["tasks"] == [source["task"]]}
    all_ids = set(train + holdout)
    remaining = all_ids - set(old_train) - set(old_holdout)
    additional = sorted(sorted(remaining, key=lambda idx: hashlib.sha256(f"42:{idx}".encode()).hexdigest())[:6])
    if (len(source_episodes) != 47 or all_ids != set(source_episodes)
            or manifest.get("additional_holdout_source_episode_ids") != additional
            or set(holdout) != set(old_holdout + additional)):
        raise ValueError("Expanded public split differs from the pinned seed-42 source split")
    inputs = {item["path"]: item for item in manifest["inputs"]}
    expected_inputs = {"evidence/data_manifest.json"}
    for name in ("info.json", "episodes.jsonl", "tasks.jsonl"):
        path = f"data/raw_public/meta/{name}"
        expected_inputs.add(path)
        if inputs.get(path) != next((item for item in pinned["files"] if item["path"] == path), None):
            raise ValueError("Expanded public metadata differs from original pinned inputs")
    for idx in all_ids:
        relative = info["data_path"].format(episode_chunk=idx // info["chunks_size"], episode_index=idx)
        expected_inputs.add(str(Path("data/raw_public") / relative))
    if set(inputs) != expected_inputs:
        raise ValueError("Expanded public input inventory differs")
    expected_files = set()
    for split, ids in (("train", train), ("holdout", holdout)):
        repo = manifest["repo_id"] + ("_holdout" if split == "holdout" else "")
        relative = f"data/lerobot/{repo}"
        if split == "holdout" and (manifest.get("holdout_repo_id") != repo or manifest.get("holdout_dataset_dir") != relative):
            raise ValueError("Expanded holdout must use a separate dataset directory")
        dataset = contained(root, relative)
        records = jsonl(dataset / "meta/episodes.jsonl")
        frames = sum(source_episodes[idx]["length"] for idx in ids)
        exported_info = json.loads((dataset / "meta/info.json").read_text())
        if (manifest.get(f"{split}_frames") != frames or exported_info.get("total_episodes") != len(ids)
                or exported_info.get("total_frames") != frames or exported_info.get("fps") != 10
                or records != [{"episode_index": number, "tasks": [source["task"]],
                                "length": source_episodes[idx]["length"]} for number, idx in enumerate(ids)]
                or jsonl(dataset / "meta/tasks.jsonl") != [{"task_index": 0, "task": source["task"]}]):
            raise ValueError("Expanded public dataset metadata counts or task differ")
        expected_files.update(f"{relative}/meta/{name}" for name in
                              ("info.json", "episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl"))
        expected_files.update(f"{relative}/data/chunk-000/episode_{number:06d}.parquet" for number in range(len(ids)))
        if {str(path.relative_to(root)) for path in dataset.rglob("*") if path.is_file()} != {
                path for path in expected_files if Path(path).is_relative_to(relative)}:
            raise ValueError("Expanded public dataset inventory differs")
    if manifest.get("original_train_frames") != manifest["train_frames"] or {item["path"] for item in manifest["files"]} != expected_files:
        raise ValueError("Expanded public exported file inventory or frame count differs")


def validate_expanded_corrections(root, manifest):
    """Validate a copied public training prefix and appended approved human episodes."""
    base_path = contained(root, manifest["public_base_manifest"])
    if base_path.parent != root / "evidence":
        raise ValueError("Expanded correction base must be an evidence manifest")
    base = json.loads(base_path.read_text())
    if (base.get("ready") is not True or base.get("project_root") != str(root)
            or base.get("dataset_kind") != "expanded_public_v1"):
        raise ValueError("Expanded correction base is not the pinned public export")
    verify_records(root, base["files"] + base["inputs"])
    validate_expanded_public(root, base)
    expected_inputs = {item["path"]: item for item in base["files"] + base["inputs"]}
    expected_inputs[str(base_path.relative_to(root))] = {
        "path": str(base_path.relative_to(root)), "bytes": base_path.stat().st_size, "sha256": digest(base_path)}
    clips = None
    if "curation_selection" in manifest:
        if not isinstance(manifest["curation_selection"], str) or not manifest["curation_selection"]:
            raise ValueError("Missing correction selection path")
        selection_path, clips = load_selection(root, manifest["curation_selection"], base["task"])
        if manifest.get("selected_source_ranges") != clips:
            raise ValueError("Correction selected ranges differ from the pinned selection")
        expected_inputs[str(selection_path.relative_to(root))] = {
            "path": str(selection_path.relative_to(root)), "bytes": selection_path.stat().st_size,
            "sha256": digest(selection_path)}
    elif "selected_source_ranges" in manifest:
        raise ValueError("Selected correction ranges require a pinned selection file")
    if {item["path"]: item for item in manifest["inputs"]} != expected_inputs:
        raise ValueError("Expanded correction public base or selection lineage differs")
    inherited = ("source_repo", "source_revision", "task", "seed", "train_source_episode_ids",
                 "holdout_source_episode_ids_excluded", "old_holdout_source_episode_ids_excluded",
                 "holdout_episodes", "holdout_frames", "original_train_episodes", "original_train_frames",
                 "normalization")
    if any(manifest.get(key) != base.get(key) for key in inherited):
        raise ValueError("Expanded correction public split or normalization differs")
    if (manifest.get("human_corrections_included") is not True or manifest.get("approval_type") != APPROVAL
            or manifest.get("sampling") != "uniform over combined public and correction frames"):
        raise ValueError("Expanded correction sampling or human approval differs")
    files = {item["path"]: item for item in manifest["files"]}
    dataset = contained(root, manifest["dataset_dir"])
    base_dataset = contained(root, base["dataset_dir"])
    expected_files = {str(path.relative_to(root)) for path in dataset.rglob("*") if path.is_file()}
    corrections, scenarios = [], {"train": set(), "holdout": set()}
    clip_sources = {clip["source_directory"] for clip in clips or []}
    raw_sources = {}
    metadata_paths = sorted(path for path in files if Path(path).parts[0] == "corrections"
                            and Path(path).name == "metadata.json")
    for relative in metadata_paths:
        metadata_path = contained(root, relative)
        metadata = json.loads(metadata_path.read_text())
        if (metadata.get("status") != "approved" or metadata.get("approval_type") != APPROVAL
                or metadata.get("human_actions_only") is not True or metadata.get("task") != base["task"]):
            raise ValueError("Mixed dataset correction lacks matching task or human approval")
        frames = load_frames(metadata_path.parent, metadata)
        approval_checks(frames)
        scenarios[metadata["split"]].add(metadata["scenario_key"])
        directory = str(metadata_path.parent.relative_to(root))
        corrections.append((directory, metadata, len(frames)))
        if directory in clip_sources:
            raw_sources[directory] = (metadata_path, metadata, frames)
        expected_files.update(str(path.relative_to(root)) for path in metadata_path.parent.rglob("*") if path.is_file())
    if set(files) != expected_files:
        raise ValueError("Expanded correction raw file inventory differs")
    if scenarios["train"] & scenarios["holdout"] or manifest["scenarios"] != {
            key: sorted(values) for key, values in scenarios.items()}:
        raise ValueError("Expanded correction scenario splits overlap or differ")
    if clips is None:
        train = [(directory, metadata, count) for directory, metadata, count in corrections if metadata["split"] == "train"]
    else:
        if set(raw_sources) != clip_sources:
            raise ValueError("Selected correction source is missing or unapproved")
        selected = []
        train = []
        for clip in clips:
            metadata_path, metadata, frames = raw_sources[clip["source_directory"]]
            selected.append(selected_frames(root, clip, metadata_path, metadata, frames))
            train.append((clip["source_directory"], metadata, len(selected[-1])))
    correction_sources = [directory for directory, _, _ in train]
    control_dts = {directory: metadata["control_dt"] for directory, metadata, _ in train}
    human_frames = sum(count for _, _, count in train)
    if (not train or manifest.get("correction_train_episodes") != len(train)
            or manifest.get("correction_train_frames") != human_frames
            or manifest.get("correction_holdout_episodes_excluded") != sum(metadata["split"] == "holdout" for _, metadata, _ in corrections)):
        raise ValueError("Expanded correction human episode or frame counts differ")
    episodes = jsonl(dataset / "meta/episodes.jsonl")
    base_episodes = jsonl(base_dataset / "meta/episodes.jsonl")
    expected_episodes = base_episodes + [
        {"episode_index": idx, "tasks": [base["task"]], "length": count}
        for idx, (_, _, count) in enumerate(train, start=len(base_episodes))]
    timing = manifest.get("timing", {})
    if (manifest.get("correction_source_directories") != correction_sources
            or timing.get("physical_control_dt_by_episode") != control_dts
            or timing.get("nominal_fps") != 10 or timing.get("nominal_dt") != .1
            or timing.get("pauses_excluded") is not True or timing.get("one_frame_per_human_control_action") is not True):
        raise ValueError("Expanded correction source ordering or timing differs")
    info = json.loads((dataset / "meta/info.json").read_text())
    base_info = json.loads((base_dataset / "meta/info.json").read_text())
    total_frames = sum(row["length"] for row in expected_episodes)
    stats = jsonl(dataset / "meta/episodes_stats.jsonl")
    if (episodes != expected_episodes or [row.get("episode_index") for row in stats] != list(range(len(episodes)))
            or info.get("total_episodes") != len(episodes)
            or info.get("total_frames") != total_frames or manifest.get("train_episodes") != len(episodes)
            or manifest.get("train_frames") != total_frames or info.get("fps") != base_info["fps"]
            or info.get("features") != base_info["features"]
            or jsonl(dataset / "meta/tasks.jsonl") != jsonl(base_dataset / "meta/tasks.jsonl")):
        raise ValueError("Expanded correction exported metadata counts or task differ")
    expected_dataset = {f"meta/{name}" for name in ("info.json", "episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl")}
    import pyarrow.parquet as pq
    for episode in episodes:
        idx = episode["episode_index"]
        relative = info["data_path"].format(episode_chunk=idx // info["chunks_size"], episode_index=idx)
        expected_dataset.add(relative)
        path = contained(root, str((dataset / relative).relative_to(root)))
        if idx < len(base_episodes):
            if digest(path) != digest(base_dataset / relative):
                raise ValueError("Expanded correction public episode prefix changed")
        elif pq.read_metadata(path).num_rows != episode["length"]:
            raise ValueError("Expanded correction parquet frame count differs")
        elif clips is not None:
            # Counts alone cannot prove that the declared interval is the exported one.
            import io
            import numpy as np
            from PIL import Image
            frames = selected[idx - len(base_episodes)]
            table = pq.read_table(path)
            offset = sum(row["length"] for row in episodes[:idx])
            if (table.schema != pq.read_schema(base_dataset / "data/chunk-000/episode_000000.parquet")
                    or table["index"].to_pylist() != list(range(offset, offset + len(frames)))
                    or not np.array_equal(np.asarray(table["timestamp"].to_pylist(), dtype=np.float32),
                                          np.asarray([number / info["fps"] for number in range(len(frames))], dtype=np.float32))
                    or table["episode_index"].to_pylist() != [idx] * len(frames)
                    or table["frame_index"].to_pylist() != list(range(len(frames)))
                    or table["task_index"].to_pylist() != [0] * len(frames)
                    or not np.array_equal(np.asarray(table["state"].to_pylist(), dtype=np.float32), np.stack([frame["state"] for frame in frames]))
                    or not np.array_equal(np.asarray(table["actions"].to_pylist(), dtype=np.float32), np.stack([frame["action"] for frame in frames]))):
                raise ValueError("Selected correction actions, states or episode boundaries differ from raw slice")
            for key in ("image", "wrist_image"):
                for row, frame in zip(table[key].to_pylist(), frames, strict=True):
                    with Image.open(io.BytesIO(row["bytes"])) as image:
                        if not np.array_equal(np.asarray(image), frame[key]):
                            raise ValueError("Selected correction camera payload differs from raw slice")
    if {str(path.relative_to(dataset)) for path in dataset.rglob("*") if path.is_file()} != expected_dataset:
        raise ValueError("Expanded correction dataset inventory differs")


def validate_inputs(manifest_path, parent_checkpoint, experiment, *, root=None):
    """Complete file validation before importing JAX, model configuration, or trainer."""
    root = Path(root or ROOT).resolve()
    experiment = safe_name(experiment)
    manifest_path = Path(manifest_path).resolve()
    if manifest_path.parent != root / "evidence" or not re.fullmatch(r"corrections-[A-Za-z0-9][A-Za-z0-9_-]{0,79}-manifest.json", manifest_path.name):
        raise ValueError("Correction manifest must be an exported evidence manifest in this project")
    manifest = json.loads(manifest_path.read_text())
    lineage_path = manifest_path
    if "training_sampling_profile" in manifest:
        from recovery_sampling import prepare_sampling
        _, sampling_record = prepare_sampling(root, manifest)
        lineage_path = contained(root, sampling_record["base_manifest"])
    if not re.fullmatch(r"corrections-[A-Za-z0-9][A-Za-z0-9_-]{0,79}-manifest.json", lineage_path.name):
        raise ValueError("Recovery sampling base must be an exported correction manifest")
    name = lineage_path.name[len("corrections-"):-len("-manifest.json")]
    expected_repo = f"course/libero_corrections_{name}"
    if (manifest.get("ready") is not True or manifest.get("project_root") != str(root)
            or manifest.get("repo_id") != expected_repo
            or manifest.get("dataset_dir") != f"data/lerobot/{expected_repo}"):
        raise ValueError("Correction dataset manifest is not ready or has incompatible lineage")
    expanded = manifest.get("dataset_kind") == "expanded_public_v1"
    mixed = manifest.get("dataset_kind") == "expanded_public_corrections_v1"
    if not mixed and ("curation_selection" in manifest or "selected_source_ranges" in manifest):
        raise ValueError("Correction selection requires an expanded mixed dataset")
    if not (expanded or mixed) and (manifest.get("dataset_kind") is not None or manifest.get("original_train_episodes") != 16
                         or type(manifest.get("correction_train_episodes")) is not int
                         or manifest["correction_train_episodes"] < 1):
        raise ValueError("Correction dataset manifest is not ready or has incompatible lineage")
    scenarios = manifest["scenarios"]
    if set(scenarios["train"]) & set(scenarios["holdout"]):
        raise ValueError("Correction scenario splits overlap")
    if set(manifest["train_source_episode_ids"]) & set(manifest["old_holdout_source_episode_ids_excluded"]):
        raise ValueError("Original training and holdout sources overlap")
    dataset = contained(root, manifest["dataset_dir"])
    files = manifest["files"]
    dataset_files = {item["path"] for item in files if Path(item["path"]).is_relative_to(Path(manifest["dataset_dir"]))}
    if not dataset_files or dataset_files != {str(p.relative_to(root)) for p in dataset.rglob("*") if p.is_file()}:
        raise ValueError("Exported dataset inventory differs")
    if not manifest.get("inputs") or not any(item["path"] == "evidence/data_manifest.json" for item in manifest["inputs"]):
        raise ValueError("Missing pinned public dataset lineage")
    verify_records(root, files + manifest["inputs"])
    if expanded:
        validate_expanded_public(root, manifest)
    elif mixed:
        validate_expanded_corrections(root, manifest)
    norm = manifest["normalization"]
    if (norm.get("asset_id") != ASSET_ID or norm.get("recomputed") is not False
            or norm.get("path") != f"assets/pi05_handson_lora/{ASSET_ID}/norm_stats.json"):
        raise ValueError("Normalization must reuse the original course assets")
    verify_records(root, [norm])
    parent = Path(parent_checkpoint).resolve()
    relative = parent.relative_to((root / "checkpoints/pi05_handson_lora").resolve())
    if len(relative.parts) != 2 or not re.fullmatch(r"[0-9]+", parent.name):
        raise ValueError("Parent must be a numeric course checkpoint step")
    if not (parent / "params/manifest.ocdbt").is_file():
        raise ValueError("Parent checkpoint params manifest is missing")
    parent_norm = parent / "assets" / ASSET_ID / "norm_stats.json"
    if digest(parent_norm) != norm["sha256"]:
        raise ValueError("Normalization differs from the parent checkpoint")
    checkpoint_dir = root / "checkpoints/pi05_handson_lora" / experiment
    launch_manifest = root / "evidence" / f"{experiment}-launch.json"
    if checkpoint_dir.exists() or launch_manifest.exists():
        raise FileExistsError(f"Refusing to overwrite an existing experiment: {experiment}")
    return manifest, parent, launch_manifest


def make_experiment(repo_id, parent, experiment, steps, batch_size, learning_rate):
    from course_config import make_config
    from openpi.training.config import AssetsConfig
    from openpi.training.weight_loaders import CheckpointWeightLoader

    cfg = make_config(steps, experiment)
    return dataclasses.replace(
        cfg, data=dataclasses.replace(cfg.data, repo_id=repo_id, assets=AssetsConfig(asset_id=ASSET_ID)),
        batch_size=batch_size, weight_loader=CheckpointWeightLoader(str(parent / "params")),
        lr_schedule=dataclasses.replace(cfg.lr_schedule, peak_lr=learning_rate, decay_lr=learning_rate),
        save_interval=min(1000, steps), keep_period=1,
    )


def git_revision(directory):
    return subprocess.check_output(["rtk", "proxy", "git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()


def build_record(cfg, manifest_path, manifest, parent, dry_run):
    source_files = ["correction_data.py", "prepare_corrections.py", "train_corrections.py", "course_config.py"]
    if manifest.get("dataset_kind") in ("expanded_public_v1", "expanded_public_corrections_v1"):
        source_files.append("prepare_expanded_public.py")
    if manifest.get("curation_selection"):
        selection = json.loads(contained(ROOT, manifest["curation_selection"]).read_text())
        if selection["schema_version"] == 2:
            source_files.append("curated_checks.py")
    return {
        "experiment": cfg.exp_name, "new_updates": cfg.num_train_steps, "batch_size": cfg.batch_size,
        "dataset_manifest": str(Path(manifest_path).resolve()), "dataset_manifest_sha256": digest(manifest_path),
        "repo_id": manifest["repo_id"], "sampling": manifest["sampling"],
        "dataset_kind": manifest.get("dataset_kind", "corrections"),
        "parent_checkpoint": str(parent), "parent_params_manifest_sha256": digest(parent / "params/manifest.ocdbt"),
        "lineage": "Initialize ALL model weights from parent; fresh Adam, scheduler, RNG and data cursor; not exact resume",
        "seed": cfg.seed, "action_horizon": cfg.model.action_horizon,
        "normalization": manifest["normalization"], "timing": manifest["timing"],
        "checkpoint_directory": str(cfg.checkpoint_dir), "final_step_index": cfg.num_train_steps - 1,
        "learning_rate": {"warmup_steps": cfg.lr_schedule.warmup_steps, "peak": cfg.lr_schedule.peak_lr, "final": cfg.lr_schedule.decay_lr},
        "optimizer": "AdamW, clip gradient norm 1.0; only LoRA trainable",
        "checkpoint_save_steps": sorted(set(range(cfg.save_interval, cfg.num_train_steps, cfg.save_interval)) | {cfg.num_train_steps - 1}),
        "source_commit": git_revision(ROOT), "openpi_commit": git_revision(ROOT / "openpi"),
        "source_worktree_status": subprocess.check_output(
            ["rtk", "proxy", "git", "-C", str(ROOT), "status", "--porcelain"], text=True).splitlines(),
        "source_sha256": {name: digest(ROOT / "scripts" / name) for name in source_files},
        "input_files": manifest["files"] + manifest["inputs"],
        "wandb_enabled": cfg.wandb_enabled, "dry_run": dry_run,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--parent-checkpoint", required=True, type=Path)
    parser.add_argument("--steps", type=positive_int, default=2000)
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument("--learning-rate", type=positive_learning_rate, default=5e-5)
    parser.add_argument("--experiment", required=True, type=safe_name)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    manifest, parent, launch_manifest = validate_inputs(args.manifest, args.parent_checkpoint, args.experiment)
    from recovery_sampling import prepare_sampling
    sampling = prepare_sampling(ROOT, manifest)
    os.environ["HF_LEROBOT_HOME"] = str(ROOT / "data/lerobot")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cfg = make_experiment(manifest["repo_id"], parent, args.experiment, args.steps, args.batch_size, args.learning_rate)
    record = build_record(cfg, args.manifest, manifest, parent, args.dry_run)
    if sampling is not None:
        if cfg.seed != sampling[1]["seed"]:
            raise ValueError("Training seed differs from recovery sampling profile")
        record["sampling"] = "recovery focus over unchanged observation/action episodes"
        record["recovery_sampling"] = sampling[1]
        for name in ("recovery_sampling.py", "train_release_focus.py"):
            record["source_sha256"][name] = digest(ROOT / "scripts" / name)
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        print("DRY_RUN_PASS: no model weights loaded, no optimizer created, no training started", flush=True)
        return
    with launch_manifest.open("x") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    data_loader, original = None, None
    if sampling is not None:
        import openpi.training.data_loader as data_loader
        from recovery_sampling import install_sampling
        original = install_sampling(data_loader, *sampling)
    try:
        spec = importlib.util.spec_from_file_location("course_correction_trainer", ROOT / "openpi/scripts/train.py")
        trainer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trainer)
        trainer.main(cfg)
    finally:
        if original is not None:
            data_loader.create_torch_dataset = original


if __name__ == "__main__":
    main()
