"""Run the pinned upstream trainer with a transparent course configuration."""
import argparse
import hashlib
import importlib.util
import json
import subprocess

from course_config import ROOT, make_config

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--experiment", default="lesson01_200")
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()
assert args.steps > 0
configuration = make_config(args.steps, args.experiment, resume=args.resume)
if configuration.checkpoint_dir.exists() and not args.resume:
    raise SystemExit("Checkpoint directory already exists; choose a new experiment or explicitly resume.")
manifest = ROOT / "evidence" / f"{args.experiment}-launch.json"
source_files = [ROOT / "scripts" / name for name in
                ("course_config.py", "train.py", "compute_stats.py", "prepare_public_data.py")]
manifest.write_text(json.dumps({
    "steps": args.steps, "experiment": args.experiment, "seed": configuration.seed,
    "batch_size": configuration.batch_size, "trainable": "LoRA parameters only",
    "resume": args.resume, "task": configuration.policy_metadata["task"],
    "upstream_commit": subprocess.check_output(
        ["git", "-C", str(ROOT / "openpi"), "rev-parse", "HEAD"], text=True).strip(),
    "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in source_files},
}, indent=2) + "\n")
spec = importlib.util.spec_from_file_location("course_upstream_train", ROOT / "openpi/scripts/train.py")
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)
trainer.main(configuration)

