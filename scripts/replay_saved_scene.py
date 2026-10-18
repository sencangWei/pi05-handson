"""Replay a saved interactive initial state in an isolated offscreen simulator."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import time

from evaluate_libero import DUMMY_ACTION, TASK_LANGUAGE, quaternion_axis_angle, update_benchmark_progress
from simulation_lab import LiberoTask, TrialRecord, state_digest
from train_release_focus import positive_int


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--port", type=int, default=8756)
    parser.add_argument("--max-steps", type=positive_int, default=800)
    parser.add_argument("--replan-steps", type=positive_int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stop-after-stable-release", action="store_true",
                        help="Stop after10 consecutive On, open, actual gap>=6cm and EEF-bowl>=10cm steps")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.replan_steps > 10 or (not args.smoke and args.checkpoint is None):
        parser.error("Need port1..65535, replan1..10 and expected checkpoint unless smoke")
    source, output = args.source_scene.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    original = json.loads((source / "scene.json").read_text())
    if original["task_language"] != TASK_LANGUAGE:
        raise ValueError("Saved scene task differs")
    os.environ.setdefault("MUJOCO_GL", "egl")
    import numpy as np
    from openpi_client import image_tools

    saved = np.load(source / "initial_state.npy", allow_pickle=False)
    if saved.ndim != 1 or not np.isfinite(saved).all() or state_digest(saved) != original["initial_state_sha256"]:
        raise ValueError("Saved initial state checksum/shape differs")
    client, metadata = None, None
    if not args.smoke:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        client = WebsocketClientPolicy("127.0.0.1", args.port)
        metadata = client.get_server_metadata()
        if Path(metadata["checkpoint"]).resolve() != args.checkpoint.resolve():
            raise ValueError("Diagnostic server checkpoint differs")
    task = LiberoTask()
    if task.metadata != original["task_metadata"]:
        raise ValueError("Saved scene BDDL or reference states differ")
    np.random.seed(original["seed"])
    env, _ = task.create(original["layout_id"])
    record = None
    try:
        obs = env.regenerate_obs_from_state(saved)
        if state_digest(env.get_sim_state()) != state_digest(saved):
            raise ValueError("Physics initial-state roundtrip changed")
        scene = {**original, "interactive_experiment": False, "formal_benchmark": False,
                 "untracked_native_gui_edits_possible": False,
                 "saved_scene_replay": str(source), "source_scene_sha256": hashlib.sha256((source / "scene.json").read_bytes()).hexdigest(),
                 "replan_steps": args.replan_steps, "max_steps": args.max_steps,
                 "stop_policy": "ten-step stable released retreat or budget" if args.stop_after_stable_release else "fixed full budget; On never stops early",
                 "policy_rng": (f"Isolated server resets RNG per connection to seed {metadata['diagnostic_rng_reset_per_connection']}"
                                if metadata is not None and "diagnostic_rng_reset_per_connection" in metadata
                                else "Use a fresh isolated server per comparison; client.reset does not reset sampling RNG"),
                 "replay_limit": "Restores MuJoCo integration state, not serialized controller internals or GUI policy RNG"}
        output.parent.mkdir(parents=True, exist_ok=True)
        record = TrialRecord(output, scene, obs, saved)
        record.result["model_metadata"] = metadata
        plan = collections.deque()
        stable, completion = 0, False
        start = time.monotonic()
        for _ in range(min(5, args.max_steps) if args.smoke else args.max_steps):
            obs = env.regenerate_obs_from_state(env.get_sim_state())  # Same fresh-observation timing as GUI.
            state = np.concatenate((obs["robot0_eef_pos"], quaternion_axis_angle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
            if args.smoke:
                action = np.asarray(DUMMY_ACTION)
            else:
                if not plan:
                    def picture(key):
                        return image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs[key][::-1, ::-1]), 224, 224))
                    response = client.infer({"observation/state": state, "prompt": TASK_LANGUAGE,
                                             "observation/image": picture("agentview_image"),
                                             "observation/wrist_image": picture("robot0_eye_in_hand_image")})
                    chunk = np.asarray(response["actions"])
                    if chunk.shape != (10, 7) or not np.isfinite(chunk).all():
                        raise ValueError("Invalid action window")
                    plan.extend(chunk[:args.replan_steps])
                    record.result["inference_calls"] += 1
                action = plan.popleft()
            obs, reward, done, _ = env.step(action.tolist())
            record.append(state, action, obs, reward, done)
            update_benchmark_progress(record.result, bool(done), args.max_steps)
            gap = float(np.abs(obs["robot0_gripper_qpos"]).sum())
            distance = float(np.linalg.norm(obs["robot0_eef_pos"] - obs["akita_black_bowl_1_pos"]))
            stable = stable + 1 if done and action[6] < 0 and gap >= .06 and distance >= .10 else 0
            if args.stop_after_stable_release and stable >= 10:
                completion = True
                break
        record.result["wall_duration_s"] = time.monotonic() - start
        record.result["ten_step_release_retreat_verified"] = completion
        reason = "smoke_complete" if args.smoke else "stable_release_retreat" if completion else "step_limit"
        record.save(reason, env.get_sim_state(), obs)
        print(json.dumps(record.result), flush=True)
    except Exception as error:
        if record is not None and not record.saved:
            record.result["exception"] = repr(error)
            record.save("exception", env.get_sim_state())
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
