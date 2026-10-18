"""Read-only training-observation action probe; never steps a simulator."""
import argparse
import json
import math
from pathlib import Path
import signal

import numpy as np

from correction_data import digest, validate_frame

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "corrections/human_1792311079911388886"
SOURCE_SHA = "8854b2899d86fd152cfa90bc56ad095814e606f4bb4f7a15a83d6e85f733fb72"
# An axis of None means gripper-only; -1 means a stationary XYZ target.
POINTS = [(15, "lift", 2), (35, "lift", 2), (65, "align_y", 1), (90, "align_y", 1),
          (120, "descend", 2), (145, "close", None), (170, "lift_bowl", 2),
          (280, "carry_x", 0), (330, "carry_x", 0), (390, "carry_y", 1),
          (450, "place", 2), (480, "release", None), (520, "retreat", 2), (575, "hold", -1)]


def check_predictions(gt, predictions, axis):
    """Empirical local gates, not task success or generalization criteria."""
    gt, predictions = np.asarray(gt), np.asarray(predictions)
    if (gt.shape != (7,) or predictions.ndim != 2 or predictions.shape[1] != 7
            or not len(predictions) or not np.isfinite(predictions).all()
            or not np.isfinite(gt).all() or gt[6] == 0):
        raise ValueError("Expected finite 7D actions and a nonzero demonstration gripper target")
    mean = predictions.mean(axis=0)
    required = math.ceil(5 * len(predictions) / 6)
    correct = int(np.count_nonzero(predictions[:, 6] * gt[6] > 0))
    checks = {"gripper_sign": correct >= required}
    movement_correct = None
    if axis == -1:
        movement_correct = int(np.count_nonzero(np.all(np.abs(predictions[:, :3]) <= .03, axis=1)))
        checks["stationary_xyz"] = movement_correct >= required
    elif axis is not None:
        if gt[axis] == 0:
            raise ValueError("Selected movement axis has a zero demonstration target")
        movement_correct = int(np.count_nonzero((predictions[:, axis] * gt[axis] > 0)
                                                & (np.abs(predictions[:, axis]) >= .03)))
        checks["movement_direction"] = movement_correct >= required
    return {"gt_action": gt.tolist(), "first_actions": predictions.tolist(), "mean": mean.tolist(),
            "std": predictions.std(axis=0).tolist(), "checks": checks,
            "gripper_correct": correct, "gripper_required": required,
            "movement_correct": movement_correct, "movement_required": required if axis is not None else None,
            "pass": all(checks.values())}


def validate_model(metadata, checkpoint, task):
    checkpoint = Path(checkpoint).resolve()
    stats = list((checkpoint / "assets").rglob("norm_stats.json"))
    if (Path(metadata.get("checkpoint", "")).resolve() != checkpoint
            or metadata.get("task") != task or metadata.get("action_horizon") != 10
            or len(stats) != 1
            or metadata.get("params_manifest_sha256") != digest(checkpoint / "params/manifest.ocdbt")
            or metadata.get("norm_stats_sha256") != digest(stats[0])):
        raise ValueError("Server checkpoint, task, action horizon, or weight/statistics checksum mismatch")


def probe_points(infer, frames, prompt, resize, samples):
    if not 1 <= samples <= 24:
        raise ValueError("samples must be between 1 and 24")
    results = []
    for index, phase, axis in POINTS:
        frame = frames[index]
        observation = {"observation/state": frame["state"], "prompt": prompt,
                       "observation/image": resize(frame["image"]),
                       "observation/wrist_image": resize(frame["wrist_image"])}
        predictions = []
        for _ in range(samples):
            chunk = np.asarray(infer(observation)["actions"])
            if chunk.shape != (10, 7) or not np.isfinite(chunk).all():
                raise ValueError(f"Invalid model action chunk at frame {index}: {chunk.shape}")
            predictions.append(chunk[0])
        results.append(dict(frame=index, phase=phase, axis=axis,
                            **check_predictions(frame["action"], predictions, axis)))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=6, choices=range(1, 25), metavar="1..24")
    args = parser.parse_args(argv)
    # Reserve exclusively before reading source or making any network request.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        report = {"status": "ERROR", "checkpoint": str(args.checkpoint.resolve()),
                  "source": str(SOURCE), "source_metadata_sha256": SOURCE_SHA, "samples": args.samples,
                  "gates": "Each frame: >=ceil(5*n/6) gripper signs correct; independently >=ceil(5*n/6) samples match movement-axis GT sign with abs>=0.03, or hold abs XYZ all<=0.03",
                  "caveats": ["Inference advances the running server RNG; simulator is never stepped.",
                              "PASS covers only specified movement-axis and gripper subchecks, not full 7D action correctness; gripper-only points impose no XYZ restriction.",
                              "Empirical local thresholds on training observations; not rollout success or verified generalization."]}
        client = None
        try:
            if digest(SOURCE / "metadata.json") != SOURCE_SHA:
                raise ValueError("Correction metadata checksum mismatch")
            metadata = json.loads((SOURCE / "metadata.json").read_text())
            if metadata["frame_count"] != 585 or metadata["status"] != "approved":
                raise ValueError("Correction count or approval differs")
            frames = {}
            for index, _, _ in POINTS:
                record = metadata["files"][index]
                path = SOURCE / f"frame-{index:06d}.npz"
                if record["path"] != path.name or path.is_symlink() or digest(path) != record["sha256"]:
                    raise ValueError(f"Correction frame checksum mismatch: {index}")
                with np.load(path, allow_pickle=False) as data:
                    frames[index] = dict(data)
                validate_frame(frames[index])
            from openpi_client import image_tools, websocket_client_policy

            def timeout(*_):
                raise TimeoutError("Probe exceeded its 120 second network/inference limit")

            signal.signal(signal.SIGALRM, timeout)
            signal.alarm(120)
            client = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=8755)
            report["model_metadata"] = client.get_server_metadata()
            validate_model(report["model_metadata"], args.checkpoint, metadata["task"])
            def resize(image):
                return image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))

            report["points"] = probe_points(client.infer, frames, metadata["task"], resize, args.samples)
            report["status"] = "PASS" if all(p["pass"] for p in report["points"]) else "NEEDS_TRAINING"
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"
        finally:
            signal.alarm(0)
            if client is not None:
                try:
                    client._ws.close()
                except Exception as error:
                    report.update(status="ERROR", error=f"Connection close failed: {error}")
            output.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"STATUS {report['status']}: {args.output}")
    return {"PASS": 0, "NEEDS_TRAINING": 2, "ERROR": 1}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
