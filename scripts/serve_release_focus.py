"""Serve the final checkpoint of the separate release experiment (run after training)."""
import argparse
import hashlib
import logging

from train_release_focus import UPDATES, make_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uniform-control", action="store_true")
    args = parser.parse_args()
    cfg = make_experiment(args.uniform_control)
    checkpoint = cfg.checkpoint_dir / str(UPDATES - 1)
    if not (checkpoint / "params/manifest.ocdbt").is_file():
        raise FileNotFoundError(f"训练尚未完成，缺少模型：{checkpoint}")
    import jax
    from openpi.policies.policy_config import create_trained_policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    logging.basicConfig(level=logging.INFO)
    print("推理设备:", jax.devices("gpu"), flush=True)
    print("加载模型:", checkpoint, flush=True)
    policy = create_trained_policy(cfg, checkpoint)
    stats = list((checkpoint / "assets").rglob("norm_stats.json"))
    if len(stats) != 1:
        raise ValueError(f"Expected one normalization file: {stats}")
    policy.metadata.update({
        "configuration": cfg.name, "checkpoint": str(checkpoint),
        "checkpoint_step_index": str(UPDATES - 1), "model_stage": cfg.exp_name,
        "policy_sampling_seed_at_server_start": 0,
        "params_manifest_sha256": hashlib.sha256((checkpoint / "params/manifest.ocdbt").read_bytes()).hexdigest(),
        "norm_stats_sha256": hashlib.sha256(stats[0].read_bytes()).hexdigest(),
    })
    print("准备监听 127.0.0.1:8755", flush=True)
    WebsocketPolicyServer(policy, host="127.0.0.1", port=8755, metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    main()
