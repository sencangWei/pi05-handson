"""Compare fixed observations with matched diffusion noise; never move a simulator."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
BUNDLE = ROOT / ".local/postgrasp-slow-v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x") as output:
        from course_config import make_config
        from openpi.policies.policy_config import create_trained_policy
        from openpi_client import image_tools
        from train_release_focus import validate_checkpoint

        checkpoint = validate_checkpoint(args.checkpoint)
        manifest_path = BUNDLE / "fixed-points.json"
        manifest = json.loads(manifest_path.read_text())
        for point in manifest["points"]:
            assert sha(BUNDLE / point["bundle"]) == point["bundle_sha256"]
        cfg = make_config(experiment=checkpoint.parent.name)
        print(f"Loading isolated diagnostic policy {checkpoint}", flush=True)
        policy = create_trained_policy(cfg, checkpoint)
        report = {"checkpoint": str(checkpoint),
                  "params_manifest_sha256": sha(checkpoint / "params/manifest.ocdbt"),
                  "norm_stats_sha256": sha(checkpoint / "assets/course/libero_black_bowl/norm_stats.json"),
                  "points_manifest_sha256": sha(manifest_path), "noise_seeds": manifest["noise_seeds"],
                  "method": "16 fixed teacher-forced observations, three identical noise tensors per observation across checkpoints",
                  "limitations": "Small action probe, not validation loss or closed-loop success; current GUI policy/RNG untouched",
                  "points": []}
        for point in manifest["points"]:
            with np.load(BUNDLE / point["bundle"], allow_pickle=False) as source:
                def picture(key):
                    return image_tools.convert_to_uint8(image_tools.resize_with_pad(source[key], 224, 224))
                obs = {"observation/state": source["state"], "prompt": cfg.policy_metadata["task"],
                       "observation/image": picture("image"),
                       "observation/wrist_image": picture("wrist_image")}
                gt = source["gt"].copy()
            predictions, durations = [], []
            for seed in manifest["noise_seeds"]:
                noise = np.random.default_rng(seed).standard_normal((10, cfg.model.action_dim)).astype(np.float32)
                start = time.monotonic()
                chunk = np.asarray(policy.infer(obs, noise=noise)["actions"])
                assert chunk.shape == (10, 7) and np.isfinite(chunk).all()
                durations.append(time.monotonic() - start)
                predictions.append(chunk)
            values = np.asarray(predictions)
            result = {"id": point["id"], "group": point["group"],
                      "gt_first": gt[0].tolist(), "predictions": values.tolist(),
                      "gt_first_xyz_norm": float(np.linalg.norm(gt[0, :3])),
                      "pred_first_xyz_norm_median": float(np.median(np.linalg.norm(values[:, 0, :3], axis=1))),
                      "pred_rest_xyz_norm_median": float(np.median(np.linalg.norm(values[:, 1:, :3], axis=2))),
                      "first_xyz_mae": float(np.mean(np.abs(values[:, 0, :3] - gt[0, :3]))),
                      "chunk_xyz_mae": float(np.mean(np.abs(values[:, :, :3] - gt[:, :3]))),
                      "first_gripper_median": float(np.median(values[:, 0, 6])),
                      "gripper_sign_correct": int(np.sum(values[:, 0, 6]*gt[0, 6] > 0)),
                      "wall_duration_s": durations}
            report["points"].append(result)
            print(f"point{point['id']} {point['group']} gt={result['gt_first_xyz_norm']:.4f} "
                  f"pred={result['pred_first_xyz_norm_median']:.4f}", flush=True)
        report["status"] = "COMPLETE_ACTION_PROBE"
        output.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"COMPLETE {args.output}", flush=True)


if __name__ == "__main__":
    main()
