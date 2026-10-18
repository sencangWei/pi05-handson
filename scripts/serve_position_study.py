"""Isolated study server: reset policy RNG for each sequential WebSocket trial."""
import argparse
import logging

from serve_checkpoint import checkpoint_assets, valid_port
from train_release_focus import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=checkpoint_assets, required=True)
    parser.add_argument("--port", type=valid_port, default=8756)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.port == 8755 or args.seed < 0:
        parser.error("Use a separate diagnostic port and a nonnegative seed")
    import jax
    from course_config import make_config
    from openpi.policies.policy_config import create_trained_policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    class SequentialSeededServer(WebsocketPolicyServer):
        active = False

        async def _handler(self, websocket):
            if self.active:
                await websocket.close(code=1013, reason="One diagnostic trial at a time")
                return
            self.active = True
            try:
                # Pinned JAX Policy owns this RNG. Only this isolated process is modified.
                self._policy._rng = jax.random.key(args.seed)
                await super()._handler(websocket)
            finally:
                self.active = False

    logging.basicConfig(level=logging.INFO)
    cfg = make_config(experiment=args.checkpoint.parent.name)
    policy = create_trained_policy(cfg, args.checkpoint)
    metadata = {**policy.metadata, "checkpoint": str(args.checkpoint),
                "model_stage": f"{args.checkpoint.parent.name}/{args.checkpoint.name}",
                "params_manifest_sha256": digest(args.checkpoint / "params/manifest.ocdbt"),
                "norm_stats_sha256": digest(args.checkpoint / "assets/course/libero_black_bowl/norm_stats.json"),
                "policy_sampling_seed_at_server_start": args.seed,
                "diagnostic_rng_reset_per_connection": args.seed,
                "model_source": "pinned OpenPI JAX Policy._rng", "state_input": False}
    print(f"DIAGNOSTIC_READY {args.checkpoint} port={args.port} seed={args.seed}", flush=True)
    SequentialSeededServer(policy, host="127.0.0.1", port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
