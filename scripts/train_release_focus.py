"""Prepare or run a separate, terminal-phase sampling experiment from a course checkpoint."""
import argparse
from collections import Counter
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
UPDATES = 2000
CONTEXT = 10
MULTIPLIER = 4
PARENT = ROOT / "checkpoints/pi05_handson_lora/my_first_200/9999"


def sampling_indices(episodes, multiplier=MULTIPLIER, context=CONTEXT):
    """Repeat terminal open frames and preceding context, preserving every original frame."""
    if multiplier < 1 or context < 0:
        raise ValueError("Invalid sampling multiplier or context")
    indices, focused, offset = [], [], 0
    for gripper in episodes:
        if not gripper or not all(x in (-1.0, 1.0) for x in gripper):
            raise ValueError("Expected finite binary gripper labels")
        closed = [i for i, x in enumerate(gripper) if x > 0]
        if not closed or gripper[-1] >= 0:
            raise ValueError("Expected a final open segment after grasping")
        start = max(0, closed[-1] + 1 - context)
        indices.extend(range(offset, offset + len(gripper)))
        focused.extend(range(offset + start, offset + len(gripper)))
        offset += len(gripper)
    if not offset:
        raise ValueError("No training frames")
    indices.extend(focused * (multiplier - 1))
    return indices, focused


class RepeatedFrames:
    """Map shuffled sample indices to the unchanged underlying LeRobot dataset."""
    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, indices
        if not indices or min(indices) < 0 or max(indices) >= len(dataset):
            raise ValueError("Sampling index outside dataset")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def terminal_open_start(gripper):
    return max(i for i, value in enumerate(gripper) if value > 0) + 1


def inspect_data(multiplier):
    import pyarrow.parquet as pq

    manifest_path = ROOT / "evidence/data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest["ready"] or set(manifest["train_source_episode_ids"]) & set(manifest["holdout_source_episode_ids"]):
        raise ValueError("Dataset is not ready or splits overlap")
    expected = {item["path"]: item["sha256"] for item in manifest["files"]}
    episodes, sources, offset = [], {}, 0
    dataset = ROOT / manifest["destination"]
    for episode_id in range(manifest["train_episodes"]):
        path = dataset / f"data/chunk-000/episode_{episode_id:06d}.parquet"
        relative = str(path.relative_to(ROOT))
        sources[relative] = digest(path)
        if sources[relative] != expected[relative]:
            raise ValueError(f"Training file differs from pinned manifest: {relative}")
        table = pq.read_table(path, columns=["actions", "index", "episode_index", "frame_index"])
        def flatten(key):
            return [int(row[0]) if isinstance(row, list) else int(row) for row in table[key].to_pylist()]
        if (flatten("index") != list(range(offset, offset + len(table)))
                or flatten("frame_index") != list(range(len(table)))
                or flatten("episode_index") != [episode_id] * len(table)):
            raise ValueError("LeRobot ordering differs from the sampling map")
        episodes.append([float(a[6]) for a in table["actions"].to_pylist()])
        offset += len(table)
    if offset != manifest["train_frames"]:
        raise ValueError("Training frame count differs from manifest")
    indices, focused = sampling_indices(episodes, multiplier)
    counts = Counter(indices)
    return indices, {
        "dataset_manifest_sha256": digest(manifest_path),
        "training_files_sha256": sources,
        "train_source_episode_ids": manifest["train_source_episode_ids"],
        "holdout_source_episode_ids_excluded": manifest["holdout_source_episode_ids"],
        "original_frames": offset, "virtual_frames": len(indices),
        "focus_frames": len(focused), "focus_multiplier": multiplier,
        "focus_sampling_fraction": sum(counts[i] for i in focused) / len(indices),
        "terminal_open_frames": sum(len(ep) - terminal_open_start(ep) for ep in episodes),
        "focus_rule": f"final open segment plus preceding {CONTEXT} frames in each training episode",
        "sampling_indices_sha256": hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
    }


def validate_experiment(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
        raise ValueError("Experiment must be 1-80 ASCII letters, digits, underscores or hyphens; start with a letter or digit")
    return value


def positive_int(value):
    number = int(value)
    if isinstance(value, bool) or str(number) != str(value) or number < 1:
        raise ValueError("Expected a positive integer")
    return number


def positive_learning_rate(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("Learning rate must be finite and positive")
    return number


def validate_checkpoint(value):
    """Accept only an existing numeric step of a course experiment, after resolving symlinks."""
    checkpoint = Path(value).resolve()
    base = (ROOT / "checkpoints/pi05_handson_lora").resolve()
    try:
        relative = checkpoint.relative_to(base)
    except ValueError:
        raise ValueError(f"Checkpoint must be under {base}") from None
    if len(relative.parts) != 2 or not re.fullmatch(r"[0-9]+", checkpoint.name):
        raise ValueError("Checkpoint must be a numeric step directory directly inside a course experiment")
    if not (checkpoint / "params/manifest.ocdbt").is_file():
        raise ValueError(f"Missing checkpoint params manifest: {checkpoint}")
    return checkpoint


def make_experiment(uniform=False, *, experiment=None, steps=UPDATES, batch_size=4,
                    learning_rate=5e-5, parent=PARENT):
    experiment = validate_experiment(experiment if experiment is not None else
                                     ("release_uniform_2000" if uniform else "release_focus_2000"))
    steps = positive_int(steps)
    batch_size = positive_int(batch_size)
    learning_rate = positive_learning_rate(learning_rate)
    parent = validate_checkpoint(parent)
    from course_config import make_config
    from openpi.training.weight_loaders import CheckpointWeightLoader

    cfg = make_config(steps, experiment)
    return dataclasses.replace(
        cfg, batch_size=batch_size,
        weight_loader=CheckpointWeightLoader(str(parent / "params")),
        lr_schedule=dataclasses.replace(cfg.lr_schedule, peak_lr=learning_rate, decay_lr=learning_rate),
        save_interval=min(1000, steps), keep_period=1,
    )


def install_sampling(data_loader, indices, expected_frames):
    """Scope the override to this process; upstream source files stay untouched."""
    original = data_loader.create_torch_dataset

    def create(data_config, action_horizon, model_config):
        if data_config.repo_id != "course/libero_black_bowl" or action_horizon != 10:
            raise ValueError("Sampling map is only valid for the pinned course task/horizon")
        dataset = original(data_config, action_horizon, model_config)
        if len(dataset) != expected_frames:
            raise ValueError("LeRobot dataset length differs from the sampling map")
        return RepeatedFrames(dataset, indices)

    data_loader.create_torch_dataset = create
    return original


def build_record(cfg, parent, record, *, dry_run):
    parent_manifest = parent / "params/manifest.ocdbt"
    stats = ROOT / "assets/pi05_handson_lora/course/libero_black_bowl/norm_stats.json"
    parent_stats = parent / "assets/course/libero_black_bowl/norm_stats.json"
    if digest(stats) != digest(parent_stats):
        raise ValueError("Normalization differs from the parent model")
    record = dict(record)
    record.update({
        "experiment": cfg.exp_name, "new_updates": cfg.num_train_steps, "batch_size": cfg.batch_size,
        "seed": cfg.seed, "action_horizon": cfg.model.action_horizon,
        "parent_checkpoint": str(parent), "parent_params_manifest_sha256": digest(parent_manifest),
        "checkpoint_directory": str(cfg.checkpoint_dir), "final_step_index": cfg.num_train_steps - 1,
        "lineage": f"Initialize ALL model weights from {parent.name}; reset Adam, scheduler, RNG and data cursor; not exact resume",
        "norm_stats_sha256": digest(stats), "norm_stats_recomputed": False,
        "learning_rate": {"warmup_steps": cfg.lr_schedule.warmup_steps,
                          "peak": cfg.lr_schedule.peak_lr, "final": cfg.lr_schedule.decay_lr},
        "optimizer": "AdamW, clip gradient norm 1.0, only LoRA trainable",
        "checkpoint_save_steps": sorted(set(range(cfg.save_interval, cfg.num_train_steps, cfg.save_interval))
                                        | {cfg.num_train_steps - 1}),
        "wandb_enabled": cfg.wandb_enabled,
        "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "openpi_commit": subprocess.check_output(["git", "-C", str(ROOT / "openpi"), "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {name: digest(ROOT / "scripts" / name) for name in ("train_release_focus.py", "course_config.py")},
        "comparison": "Compare focus vs uniform-control with equal parent/budget and fresh identical server seeds. Original 10000-step vs focus also changes training budget.",
        "dry_run": dry_run,
    })
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Check inputs/configuration only; never load model weights or train")
    parser.add_argument("--uniform-control", action="store_true", help="Same parent and budget, original uniform sampling")
    parser.add_argument("--experiment", type=validate_experiment)
    parser.add_argument("--steps", type=positive_int, default=UPDATES)
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument("--learning-rate", type=positive_learning_rate, default=5e-5)
    parser.add_argument("--parent-checkpoint", type=validate_checkpoint, default=str(PARENT))
    args = parser.parse_args(argv)
    experiment = args.experiment or ("release_uniform_2000" if args.uniform_control else "release_focus_2000")
    checkpoint_dir = ROOT / "checkpoints/pi05_handson_lora" / experiment
    launch_manifest = ROOT / "evidence" / f"{experiment}-launch.json"
    if checkpoint_dir.exists() or launch_manifest.exists():
        raise FileExistsError(f"Refusing to overwrite an existing experiment: {experiment}")
    # Set before importing LeRobot; avoid dependence on the interactive shell's data cache.
    os.environ["HF_LEROBOT_HOME"] = str(ROOT / "data/lerobot")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cfg = make_experiment(args.uniform_control, experiment=experiment, steps=args.steps,
                          batch_size=args.batch_size, learning_rate=args.learning_rate, parent=args.parent_checkpoint)
    indices, record = inspect_data(1 if args.uniform_control else MULTIPLIER)
    record = build_record(cfg, args.parent_checkpoint, record, dry_run=args.dry_run)
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        print("DRY_RUN_PASS: no model weights loaded, no optimizer created, no training started", flush=True)
        return
    with launch_manifest.open("x") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    import openpi.training.data_loader as data_loader
    original = install_sampling(data_loader, indices, record["original_frames"])
    try:
        spec = importlib.util.spec_from_file_location("course_release_trainer", ROOT / "openpi/scripts/train.py")
        trainer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trainer)
        trainer.main(cfg)
    finally:
        data_loader.create_torch_dataset = original


if __name__ == "__main__":
    main()
