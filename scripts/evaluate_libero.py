#!/usr/bin/env python3
"""Run the single training task in LIBERO; no hardware or training is started.

Video follows the upstream reference's 10 FPS simulation-step replay rate.
Inference pauses are measured separately: video duration is NOT wall duration.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback


TASK_LANGUAGE = "pick up the black bowl next to the plate and place it on the plate"
DUMMY_ACTION = [0.0] * 6 + [-1.0]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--initial-state-offset", type=int, default=0)
    parser.add_argument("--port", type=int, default=8755)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--smoke", action="store_true", help="Render five zero-motion steps; never connect to a model.")
    parser.add_argument("--viewer", action="store_true", help="Open MuJoCo's interactive 3D viewer, initially paused.")
    parser.add_argument("--post-success-steps", type=int, default=0,
                        help="Continue policy actions after the first official success signal; bounded by max-steps.")
    parser.add_argument("--target-shift-step", type=int, help="Before this control step, translate the ungrasped target bowl.")
    parser.add_argument("--target-shift-x-m", type=float, default=0.05, help="Instant world-X translation in meters.")
    args = parser.parse_args()
    if args.trials < 1 or args.replan_steps < 1 or args.max_steps < 1:
        parser.error("trials, replan-steps and max-steps must be positive")
    if args.initial_state_offset < 0:
        parser.error("initial-state-offset must be nonnegative")
    if args.post_success_steps < 0 or (args.smoke and args.post_success_steps):
        parser.error("post-success-steps must be nonnegative and requires a policy trial")
    if args.target_shift_step is not None:
        if args.smoke or not 0 <= args.target_shift_step < args.max_steps:
            parser.error("target-shift-step requires a policy trial and must be within its step limit")
        if not math.isfinite(args.target_shift_x_m) or not 0 < abs(args.target_shift_x_m) <= 0.1:
            parser.error("target-shift-x-m must be finite and nonzero, with magnitude at most 0.1 m")
    return args


def prepare_output(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=json_default, allow_nan=False) + "\n")


def quaternion_axis_angle(quaternion):
    """LIBERO quaternion is xyzw; match upstream without mutating the observation."""
    q = [float(x) for x in quaternion]
    w = min(1.0, max(-1.0, q[3]))
    denominator = math.sqrt(1.0 - w * w)
    if math.isclose(denominator, 0.0):
        return [0.0, 0.0, 0.0]
    return [x * 2.0 * math.acos(w) / denominator for x in q[:3]]


def update_benchmark_progress(result, done, extra_steps):
    """Keep the first On-predicate success distinct from subsequent policy observation."""
    first = result["first_benchmark_success_control_steps"]
    if done and first is None:
        first = result["control_steps"]
        result["first_benchmark_success_control_steps"] = first
        result["success"] = True
    result["final_benchmark_success"] = bool(done)
    result["post_success_steps_executed"] = 0 if first is None else result["control_steps"] - first
    return first is not None and result["post_success_steps_executed"] >= extra_steps


def shift_ungrasped_target(env, obs, dx):
    """Translate only the target free joint; obtain fresh observations before inference."""
    import numpy as np

    target = "akita_black_bowl_1"
    before = np.asarray(obs[f"{target}_pos"], dtype=float).copy()
    distance = float(np.linalg.norm(np.asarray(obs["robot0_eef_pos"]) - before))
    if distance < 0.12 or np.linalg.norm(obs["robot0_gripper_qpos"]) < 0.04:
        raise ValueError("Target shift requires an open gripper at least 0.12 m from the bowl")
    joints = env.env.objects_dict[target].joints
    if len(joints) != 1:
        raise ValueError(f"Expected one target free joint, got {joints}")
    joint = joints[0]
    qpos = np.asarray(env.sim.data.get_joint_qpos(joint)).copy()
    qvel = np.asarray(env.sim.data.get_joint_qvel(joint))
    if qpos.shape != (7,) or qvel.shape != (6,):
        raise ValueError("Target joint does not have free-joint position/velocity dimensions")
    qpos[0] += dx
    env.sim.data.set_joint_qpos(joint, qpos)
    env.sim.data.set_joint_qvel(joint, np.zeros_like(qvel))
    refreshed = env.regenerate_obs_from_state(env.get_sim_state())
    after = np.asarray(refreshed[f"{target}_pos"], dtype=float)
    if not np.allclose(after - before, [dx, 0, 0], rtol=0, atol=1e-6):
        raise ValueError("Observed target translation differs from the requested world-X shift")
    return refreshed, {"target": target, "kind": "instant_translation_before_grasp",
                       "before_xyz": before.tolist(), "after_xyz": after.tolist(),
                       "requested_dx_m": dx, "actual_delta_xyz": (after - before).tolist(),
                       "eef_distance_before_m": distance}


def main() -> int:
    args = arguments()
    output = prepare_output(args.output)
    # MuJoCo physics runs on CPU. EGL can use the GPU for rendering only.
    if args.viewer:
        os.environ["MUJOCO_GL"] = "glx"
        os.environ["PYOPENGL_PLATFORM"] = "glx"
    else:
        os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import imageio.v2 as imageio
    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools

    np.random.seed(args.seed)
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    matches = [(i, suite.get_task(i)) for i in range(suite.n_tasks)
               if suite.get_task(i).language.strip().casefold() == TASK_LANGUAGE]
    assert len(matches) == 1, f"Expected one exact language match, found {len(matches)}"
    task_id, task = matches[0]
    # These initial states come from our pinned, public LIBERO checkout.
    # PyTorch >=2.6 changed the default loader and rejects its NumPy payload.
    import torch
    initial_state_file = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    initial_states = torch.load(initial_state_file, weights_only=False)
    if args.initial_state_offset + args.trials > len(initial_states):
        raise ValueError(f"Only {len(initial_states)} distinct reference initial states available")
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    metadata = {
        "suite": "libero_spatial", "task_id": task_id, "task_language": task.language,
        "seed": args.seed, "trials_requested": args.trials, "smoke": args.smoke,
        "initial_state_offset": args.initial_state_offset,
        "host": "127.0.0.1", "port": args.port, "replan_steps": args.replan_steps,
        "max_steps": args.max_steps, "settling_steps": 10,
        "target_shift_step": args.target_shift_step,
        "target_shift_x_m": args.target_shift_x_m if args.target_shift_step is not None else None,
        "video_fps": 10, "video_timing": "one initial frame plus one per controlled simulation step; settling and inference wall pauses omitted",
        "physics": "CPU MuJoCo", "render_backend": os.environ["MUJOCO_GL"],
        "interactive_viewer": args.viewer,
        "manual_gui_edits_recorded": False,
        "success_definition": "Official LIBERO On(bowl, plate): contact, relative height and XY proximity; no release/retreat requirement",
        "post_success_steps_requested": args.post_success_steps,
        "state_contract": "8D: eef xyz, quaternion-derived axisangle xyz, gripper qpos 2",
        "action_contract": "7D LIBERO end-effector delta position/rotation and gripper command",
        "image_contract": "agentview and wrist RGB, 180-degree flip, pad-resize 224x224 uint8",
        "bddl_path": str(bddl), "bddl_sha256": hashlib.sha256(bddl.read_bytes()).hexdigest(),
        "client_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_metadata": None, "trials": [],
    }
    write_json(output / "summary.json", metadata)
    client = None
    if not args.smoke:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        client = WebsocketClientPolicy("127.0.0.1", args.port)
        metadata["model_metadata"] = client.get_server_metadata()
        write_json(output / "summary.json", metadata)

    for trial in range(args.trials):
        initial_state_id = args.initial_state_offset + trial
        prefix = f"trial-{trial:03d}_init-{initial_state_id:03d}"
        result = {"trial": trial, "initial_state_id": initial_state_id,
                  "initial_state_sha256": hashlib.sha256(np.asarray(initial_states[initial_state_id]).tobytes()).hexdigest(),
                  "outcome": "exception", "success": False, "control_steps": 0,
                  "inference_calls": 0, "exception": None}
        result["target_shift"] = None
        result.update({"first_benchmark_success_control_steps": None,
                       "post_success_steps_executed": 0, "final_benchmark_success": None,
                       "release_and_retreat_success": None})
        images, env, live = [], None, None
        viewer_closed = False
        started = time.perf_counter()
        try:
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
            env.seed(args.seed)
            env.reset()
            obs = env.set_init_state(initial_states[initial_state_id])
            # Save an initial image even if settling or the first inference fails.
            images.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
            for _ in range(10):
                obs, _, _, _ = env.step(DUMMY_ACTION)
            if args.viewer:
                from live_preview import LivePreview
                live = LivePreview(env)
            if client is not None:
                client.reset()
            plan = collections.deque()
            step_limit = min(5, args.max_steps) if args.smoke else args.max_steps
            with (output / f"{prefix}_actions.csv").open("w", newline="") as handle:
                columns = ["step", "wall_elapsed_s", "inference_latency_s", "new_chunk", "reward", "done"]
                columns += [f"state_{i}" for i in range(8)] + [f"action_{i}" for i in range(7)]
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                for step in range(step_limit):
                    if live is not None:
                        if not live.before_step():
                            viewer_closed = True
                            break
                        with live.lock():
                            obs = env.regenerate_obs_from_state(env.get_sim_state())
                    if args.target_shift_step is not None and step == args.target_shift_step:
                        imageio.imwrite(output / f"{prefix}_shift-before.png",
                                        np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                        with live.lock() if live is not None else contextlib.nullcontext():
                            obs, event = shift_ungrasped_target(env, obs, args.target_shift_x_m)
                        event["control_step"] = step
                        result["target_shift"] = event
                        plan.clear()
                        imageio.imwrite(output / f"{prefix}_shift-after.png",
                                        np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                        write_json(output / f"{prefix}_target-shift.json", event)
                        print("TARGET_SHIFT " + json.dumps(event), flush=True)
                    image = image_tools.convert_to_uint8(image_tools.resize_with_pad(
                        np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), 224, 224))
                    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(
                        np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), 224, 224))
                    state = np.concatenate((obs["robot0_eef_pos"],
                                            quaternion_axis_angle(obs["robot0_eef_quat"]),
                                            obs["robot0_gripper_qpos"]))
                    assert state.shape == (8,) and np.isfinite(state).all()
                    latency, fresh = 0.0, False
                    if args.smoke:
                        action = np.asarray(DUMMY_ACTION)
                    else:
                        if not plan:
                            query_started = time.perf_counter()
                            response = client.infer({"observation/image": image, "observation/wrist_image": wrist,
                                                     "observation/state": state, "prompt": str(task.language)})
                            latency = time.perf_counter() - query_started
                            chunk = np.asarray(response["actions"])
                            if chunk.ndim != 2 or chunk.shape[1] != 7 or len(chunk) < args.replan_steps:
                                raise ValueError(f"Invalid action chunk shape {chunk.shape}")
                            if not np.isfinite(chunk).all():
                                raise ValueError("Nonfinite policy action")
                            plan.extend(chunk[:args.replan_steps])
                            result["inference_calls"] += 1
                            fresh = True
                        action = plan.popleft()
                    with live.lock() if live is not None else contextlib.nullcontext():
                        obs, reward, done, _ = env.step(action.tolist())
                    if live is not None:
                        live.after_step()
                    images.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                    row = {"step": step, "wall_elapsed_s": time.perf_counter() - started,
                           "inference_latency_s": latency, "new_chunk": fresh,
                           "reward": float(reward), "done": bool(done)}
                    row.update({f"state_{i}": float(x) for i, x in enumerate(state)})
                    row.update({f"action_{i}": float(x) for i, x in enumerate(action)})
                    writer.writerow(row)
                    handle.flush()
                    result["control_steps"] += 1
                    result["final_gripper_command"] = float(action[6])
                    result["final_gripper_qpos"] = np.asarray(obs["robot0_gripper_qpos"]).tolist()
                    result["final_eef_xyz"] = np.asarray(obs["robot0_eef_pos"]).tolist()
                    result["final_target_xyz"] = np.asarray(obs["akita_black_bowl_1_pos"]).tolist()
                    if not args.smoke:
                        if done and result["first_benchmark_success_control_steps"] is None and args.post_success_steps:
                            print(f"Official On predicate reached at control step {result['control_steps']}; "
                                  f"observe up to {args.post_success_steps} more policy steps for release/retreat.", flush=True)
                        if update_benchmark_progress(result, done, args.post_success_steps):
                            break
            result["outcome"] = "smoke_only" if args.smoke else ("success" if result["success"] else "failure")
            if result["success"] and args.post_success_steps:
                result["outcome"] = "benchmark_success_followup"
                result["post_success_observation_complete"] = result["post_success_steps_executed"] >= args.post_success_steps
            if viewer_closed:
                result["outcome"] = "user_stopped"
            if not viewer_closed and args.target_shift_step is not None and result["target_shift"] is None:
                result["outcome"] = "perturbation_not_applied"
                result["success"] = False
        except Exception:
            result["success"] = False
            result["outcome"] = "exception"
            result["exception"] = traceback.format_exc()
        finally:
            result["wall_duration_s"] = time.perf_counter() - started
            if live is not None:
                try:
                    if result["outcome"] in ("success", "failure", "smoke_only", "benchmark_success_followup"):
                        live.hold_final(result["outcome"])
                finally:
                    live.close()
            if env is not None:
                try:
                    env.close()
                except Exception:
                    result["cleanup_exception"] = traceback.format_exc()
                    result["success"] = False
                    result["outcome"] = "exception"
            if images:
                video = output / f"{prefix}_{result['outcome']}.mp4"
                try:
                    imageio.mimwrite(video, images, fps=10)
                    result["video"] = video.name
                    result["video_frames"] = len(images)
                except Exception:
                    result["video_exception"] = traceback.format_exc()
                    result["success"] = False
                    result["outcome"] = "exception"
            write_json(output / f"{prefix}_result.json", result)
            metadata["trials"].append(result)
            metadata["successes"] = sum(x["success"] for x in metadata["trials"])
            valid = [x for x in metadata["trials"]
                     if x["outcome"] != "user_stopped"
                     and (args.target_shift_step is None or x["target_shift"] is not None)]
            metadata["trials_valid_for_success_rate"] = len(valid)
            metadata["success_rate"] = None if args.smoke or not valid else sum(x["success"] for x in valid) / len(valid)
            metadata["exception_count"] = sum(x["outcome"] == "exception" for x in metadata["trials"])
            write_json(output / "summary.json", metadata)
            print(json.dumps(result, default=json_default), flush=True)
        if viewer_closed:
            break
    return int(any(x["outcome"] in ("exception", "perturbation_not_applied")
                   or "video_exception" in x for x in metadata["trials"]))


if __name__ == "__main__":
    raise SystemExit(main())
