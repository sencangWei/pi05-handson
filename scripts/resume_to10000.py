import dataclasses
import hashlib
import importlib.util
import json

from course_config import ROOT, make_config

cfg = dataclasses.replace(
    make_config(10000, "my_first_200", resume=True),
    batch_size=4,
    save_interval=2000,
    keep_period=1,
)

parent = cfg.checkpoint_dir / "1999"
existing = {
    p.name for p in cfg.checkpoint_dir.iterdir()
    if p.is_dir() and p.name.isdecimal()
}
assert existing == {"199", "1999"}, f"请先检查已有模型: {existing}"
for item in ("params", "train_state", "assets"):
    assert (parent / item).is_dir(), f"缺少恢复文件: {item}"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

record = {
    "parent_checkpoint": str(parent),
    "parent_manifest_sha256": digest(parent / "params/manifest.ocdbt"),
    "previous_resume_record_sha256": digest(
        ROOT / "evidence/my_first_200-resume-to2000.json"
    ),
    "dataset_manifest_sha256": digest(
        ROOT / "evidence/data_manifest.json"
    ),
    "source_sha256": {
        name: digest(ROOT / "scripts" / name)
        for name in ("course_config.py", "resume_to10000.py")
    },
    "initial_update": 2000,
    "total_updates": 10000,
    "additional_updates": 8000,
    "previous_batch_size": 1,
    "batch_size": cfg.batch_size,
    "seed": cfg.seed,
    "save_interval": cfg.save_interval,
    "keep_period": cfg.keep_period,
    "comparison_limit": "Both batch size and training budget changed.",
    "dataset_cursor": "Restarts; upstream does not restore DataLoader position.",
}
with (ROOT / "evidence/my_first_200-resume-to10000.json").open("x") as f:
    json.dump(record, f, indent=2)
    f.write("\n")

print("恢复来源:", parent, flush=True)
print("目标: 10000 次更新；batch=4；新增 32000 个样本", flush=True)

spec = importlib.util.spec_from_file_location(
    "course_upstream_train", ROOT / "openpi/scripts/train.py"
)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)
trainer.main(cfg)
