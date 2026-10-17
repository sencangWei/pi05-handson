import dataclasses
import hashlib
import importlib.util
import json

from course_config import ROOT, make_config

configuration = dataclasses.replace(
    make_config(2000, "my_first_200", resume=True),
    keep_period=1,
)
parent = configuration.checkpoint_dir / "199"
assert parent.is_dir(), "找不到原来的 200 步 checkpoint"

existing = sorted(
    p.name for p in configuration.checkpoint_dir.iterdir()
    if p.is_dir() and p.name.isdecimal()
)
assert existing == ["199"], f"已有其他 checkpoint，先检查: {existing}"

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

record = {
    "parent_checkpoint": str(parent),
    "parent_manifest_sha256": sha256(parent / "params/manifest.ocdbt"),
    "original_launch_sha256": sha256(
        ROOT / "evidence/my_first_200-launch.json"
    ),
    "total_updates": 2000,
    "additional_updates": 1800,
    "batch_size": configuration.batch_size,
    "seed": configuration.seed,
    "dataset_manifest_sha256": sha256(
        ROOT / "evidence/data_manifest.json"
    ),
    "resume_script_sha256": sha256(
        ROOT / "scripts/resume_to2000.py"
    ),
    "checkpoint_retention": "Keep checkpoint 199 and the new checkpoint.",
    "dataset_cursor": "Restarts; upstream does not restore DataLoader position.",
}

manifest = ROOT / "evidence/my_first_200-resume-to2000.json"
with manifest.open("x") as f:
    json.dump(record, f, indent=2)
    f.write("\n")

print("恢复来源:", parent, flush=True)
print("目标总更新次数: 2000；新增更新次数: 1800", flush=True)
print("batch:", configuration.batch_size, flush=True)
print("保留原始 checkpoint: 199", flush=True)

spec = importlib.util.spec_from_file_location(
    "course_upstream_train", ROOT / "openpi/scripts/train.py"
)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)
trainer.main(configuration)
