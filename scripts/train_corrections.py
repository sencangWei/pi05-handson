"""Start a separate course fine-tune from an immutable correction dataset."""
import argparse
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess

from correction_data import digest
from prepare_corrections import ASSET_ID, contained, safe_name, verify_records
from train_release_focus import positive_int, positive_learning_rate

ROOT = Path(__file__).resolve().parents[1]


def validate_inputs(manifest_path, parent_checkpoint, experiment, *, root=None):
    """Complete file validation before importing JAX, model configuration, or trainer."""
    root = Path(root or ROOT).resolve()
    experiment = safe_name(experiment)
    manifest_path = Path(manifest_path).resolve()
    if manifest_path.parent != root / "evidence" or not re.fullmatch(r"corrections-[A-Za-z0-9][A-Za-z0-9_-]{0,79}-manifest.json", manifest_path.name):
        raise ValueError("Correction manifest must be an exported evidence manifest in this project")
    manifest = json.loads(manifest_path.read_text())
    name = manifest_path.name[len("corrections-"):-len("-manifest.json")]
    expected_repo = f"course/libero_corrections_{name}"
    if (manifest.get("ready") is not True or manifest.get("project_root") != str(root)
            or manifest.get("repo_id") != expected_repo
            or manifest.get("dataset_dir") != f"data/lerobot/{expected_repo}"
            or manifest.get("original_train_episodes") != 16
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
    return {
        "experiment": cfg.exp_name, "new_updates": cfg.num_train_steps, "batch_size": cfg.batch_size,
        "dataset_manifest": str(Path(manifest_path).resolve()), "dataset_manifest_sha256": digest(manifest_path),
        "repo_id": manifest["repo_id"], "sampling": manifest["sampling"],
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
        "source_sha256": {name: digest(ROOT / "scripts" / name) for name in
                          ("correction_data.py", "prepare_corrections.py", "train_corrections.py", "course_config.py")},
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
    os.environ["HF_LEROBOT_HOME"] = str(ROOT / "data/lerobot")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cfg = make_experiment(manifest["repo_id"], parent, args.experiment, args.steps, args.batch_size, args.learning_rate)
    record = build_record(cfg, args.manifest, manifest, parent, args.dry_run)
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        print("DRY_RUN_PASS: no model weights loaded, no optimizer created, no training started", flush=True)
        return
    with launch_manifest.open("x") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    spec = importlib.util.spec_from_file_location("course_correction_trainer", ROOT / "openpi/scripts/train.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    trainer.main(cfg)


if __name__ == "__main__":
    main()
