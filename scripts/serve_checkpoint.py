"""Serve a selected local course checkpoint with the original ten-action horizon."""
import argparse
import logging

from train_release_focus import digest, validate_checkpoint


def valid_port(value):
    port = int(value)
    if not 1 <= port <= 65535:
        raise ValueError("Port must be between 1 and 65535")
    return port


def checkpoint_assets(value):
    checkpoint = validate_checkpoint(value)
    stats = checkpoint / "assets/course/libero_black_bowl/norm_stats.json"
    if not stats.is_file():
        raise ValueError(f"Missing course normalization assets: {stats}")
    return checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=checkpoint_assets, required=True)
    parser.add_argument("--port", type=valid_port, default=8755)
    args = parser.parse_args(argv)

    from course_config import make_config
    import jax
    from openpi.policies.policy_config import create_trained_policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    checkpoint = args.checkpoint
    cfg = make_config(experiment=checkpoint.parent.name)
    logging.basicConfig(level=logging.INFO)
    print("推理设备:", jax.devices("gpu"), flush=True)
    print("加载模型:", checkpoint, flush=True)
    policy = create_trained_policy(cfg, checkpoint)
    policy.metadata.update({
        "configuration": cfg.name, "checkpoint": str(checkpoint),
        "checkpoint_step_index": checkpoint.name,
        "model_stage": f"{checkpoint.parent.name}/{checkpoint.name}",
        "policy_sampling_seed_at_server_start": 0,
        "params_manifest_sha256": digest(checkpoint / "params/manifest.ocdbt"),
        "norm_stats_sha256": digest(checkpoint / "assets/course/libero_black_bowl/norm_stats.json"),
    })
    print(f"准备监听 127.0.0.1:{args.port}", flush=True)
    WebsocketPolicyServer(policy, host="127.0.0.1", port=args.port,
                          metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    main()
