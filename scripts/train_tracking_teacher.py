"""Train from a validated append-only teacher export; human checks remain intact."""
import argparse
import importlib.util
import json
import os
from pathlib import Path

from correction_data import digest
from prepare_corrections import contained, safe_name
from prepare_tracking_teacher import BASE_MANIFEST, ROOT, assert_base, validate_export
import train_corrections
from train_release_focus import positive_int, positive_learning_rate


def validate_inputs(manifest_path, parent, experiment, *, root=None):
    root = Path(root or ROOT).resolve()
    # Full original public/human validation must happen before accepting generated data.
    base, parent, launch = train_corrections.validate_inputs(root / BASE_MANIFEST, parent, experiment, root=root)
    assert_base(base)
    path = Path(manifest_path).resolve()
    if path.parent != root / "evidence":
        raise ValueError("Tracking teacher manifest must be in evidence")
    manifest = json.loads(path.read_text())
    lineage = path
    if "training_sampling_profile" in manifest:
        from recovery_sampling import prepare_sampling
        _, record = prepare_sampling(root, manifest)
        lineage = contained(root, record["base_manifest"])
    prefix, suffix = "tracking-teacher-", "-manifest.json"
    if not lineage.name.startswith(prefix) or not lineage.name.endswith(suffix):
        raise ValueError("Tracking teacher manifest identity differs")
    name = safe_name(lineage.name[len(prefix):-len(suffix)])
    repo = f"course/libero_tracking_teacher_{name}"
    if manifest.get("repo_id") != repo or manifest.get("dataset_dir") != f"data/lerobot/{repo}":
        raise ValueError("Tracking teacher dataset identity differs")
    validate_export(root, manifest, base)
    return manifest, parent, launch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--parent-checkpoint", required=True, type=Path)
    parser.add_argument("--experiment", required=True, type=safe_name)
    parser.add_argument("--steps", type=positive_int, default=2000)
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument("--learning-rate", type=positive_learning_rate, default=1e-5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    manifest, parent, launch = validate_inputs(args.manifest, args.parent_checkpoint, args.experiment)
    from recovery_sampling import prepare_sampling
    sampling = prepare_sampling(ROOT, manifest)
    os.environ["HF_LEROBOT_HOME"] = str(ROOT / "data/lerobot")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cfg = train_corrections.make_experiment(manifest["repo_id"], parent, args.experiment,
                                           args.steps, args.batch_size, args.learning_rate)
    if cfg.model.discrete_state_input is not False or cfg.model.action_horizon != 10 or cfg.seed != 42:
        raise ValueError("Tracking teacher experiment requires state=False, horizon10 and seed42")
    record = train_corrections.build_record(cfg, args.manifest, manifest, parent, args.dry_run)
    record.update(base_manifest=manifest["base_manifest"], teachers=manifest["teachers"],
                  discrete_state_input=False)
    for name in ("prepare_tracking_teacher.py", "train_tracking_teacher.py", "generate_tracking_teacher.py",
                 "prepare_expanded_public.py", "prepare_public_data.py"):
        record["source_sha256"][name] = digest(ROOT / "scripts" / name)
    if sampling is not None:
        if cfg.seed != sampling[1]["seed"]:
            raise ValueError("Training seed differs from sampling profile")
        record["sampling"] = "explicit profile over unchanged public/human and tracking teacher episodes"
        record["recovery_sampling"] = sampling[1]
        for name in ("recovery_sampling.py", "train_release_focus.py"):
            record["source_sha256"][name] = digest(ROOT / "scripts" / name)
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        print("DRY_RUN_PASS: no model weights loaded, no optimizer created, no training started", flush=True)
        return
    with launch.open("x") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    loader, original = None, None
    if sampling is not None:
        import openpi.training.data_loader as loader
        from recovery_sampling import install_sampling
        original = install_sampling(loader, *sampling)
    try:
        spec = importlib.util.spec_from_file_location("tracking_teacher_trainer", ROOT / "openpi/scripts/train.py")
        trainer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trainer)
        trainer.main(cfg)
    finally:
        if original is not None:
            loader.create_torch_dataset = original


if __name__ == "__main__":
    main()
