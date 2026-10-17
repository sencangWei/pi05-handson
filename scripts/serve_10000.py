import hashlib
import logging
from pathlib import Path

import jax
from openpi.policies.policy_config import create_trained_policy
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from course_config import NAME, ROOT, make_config

logging.basicConfig(level=logging.INFO)

checkpoint = ROOT / "checkpoints/pi05_handson_lora/my_first_200/9999"
assert (checkpoint / "params").is_dir(), f"找不到模型: {checkpoint}"

print("推理设备:", jax.devices("gpu"), flush=True)
print("加载你的 checkpoint:", checkpoint, flush=True)

configuration = make_config(steps=10000, experiment="my_first_200")
policy = create_trained_policy(configuration, checkpoint)

stats = list((checkpoint / "assets").rglob("norm_stats.json"))
assert len(stats) == 1, f"归一化文件数量异常: {stats}"

policy.metadata.update({
    "configuration": NAME,
    "checkpoint": str(checkpoint),
    "checkpoint_step_index": "9999",
    "model_stage": "student_lora_10000",
    "policy_sampling_seed_at_server_start": 0,
    "params_manifest_sha256": hashlib.sha256(
        (checkpoint / "params/manifest.ocdbt").read_bytes()
    ).hexdigest(),
    "norm_stats_sha256": hashlib.sha256(stats[0].read_bytes()).hexdigest(),
})

print("模型加载完成，准备监听 127.0.0.1:8755", flush=True)
WebsocketPolicyServer(
    policy,
    host="127.0.0.1",
    port=8755,
    metadata=policy.metadata,
).serve_forever()
