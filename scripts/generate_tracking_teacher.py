"""Bounded two-source carry-tracking pilot; generated trajectories are never human demonstrations."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from correction_data import approval_checks, digest, load_frames, validate_frame, write_json

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [("human_1792316444273677599", 796, (390, 550)),
           ("human_1792316988593839967", 735, (341, 495))]
GATE_NAMES = ("terminal_on_last_ten", "terminal_actual_open_last_ten", "terminal_open_command_last_ten",
              "terminal_retreat_ge_10cm", "carry_position_p95_le_5mm", "carry_orientation_p95_le_3deg",
              "carry_not_dropped", "carry_speedup_meets_declared_rate", "continuous_physics_trajectory",
              "actions_finite_and_bounded", "reference_coverage_complete")


def reference_schedule(count, carry, protected=10, rate=2):
    start, stop = carry
    if not 0 <= start < stop <= count or stop-start <= 2*protected or protected < 10 or rate not in (1, 2):
        raise ValueError("Invalid carry interval or boundary protection")
    schedule, index = [], 0
    while index < count:
        step_rate = rate if start+protected <= index < stop-protected else 1
        step_rate = min(step_rate, stop-protected-index) if step_rate == 2 else step_rate
        schedule.append((index, step_rate))
        index += step_rate
    return schedule


def world_rotation_error(reference_rotvec, actual_rotvec):
    """World-frame left-multiplicative SO(3) error, matching robosuite set_goal_orientation."""
    return (Rotation.from_rotvec(reference_rotvec) * Rotation.from_rotvec(actual_rotvec).inv()).as_rotvec()


def tracking_action(source_action, reference_state, actual_state, rate, scale):
    source_action, reference_state, actual_state, scale = map(
        np.asarray, (source_action, reference_state, actual_state, scale))
    if (source_action.shape != (7,) or reference_state.shape != (8,) or actual_state.shape != (8,)
            or scale.shape != (6,) or np.any(scale <= 0) or rate not in (1, 2)
            or not all(np.isfinite(x).all() for x in [source_action, reference_state, actual_state, scale])):
        raise ValueError("Invalid finite tracking inputs")
    error = np.concatenate((reference_state[:3]-actual_state[:3],
                            world_rotation_error(reference_state[3:6], actual_state[3:6])))
    command = source_action.copy()
    command[:6] = rate*source_action[:6] + error/scale
    return np.clip(command, -1., 1.).astype(np.float32), error


def frame_action(source_action, reference_state, actual_state, rate, scale, index, carry):
    """Preserve original approach/grasp/recovery/release; feedback only inside audited carry."""
    if carry[0] <= index < carry[1]:
        return tracking_action(source_action, reference_state, actual_state, rate, scale)
    if rate != 1:
        raise ValueError("Reference progress outside carry must remain one")
    return np.asarray(source_action, dtype=np.float32).copy(), np.zeros(6)


def quality_gates(metrics, rate=2):
    checks = {"terminal_on_last_ten": metrics["terminal_on_last_ten"],
              "terminal_actual_open_last_ten": metrics["terminal_min_gap_m"] >= .06,
              "terminal_open_command_last_ten": metrics["terminal_open_command_last_ten"],
              "terminal_retreat_ge_10cm": metrics["final_retreat_m"] >= .10,
              "carry_position_p95_le_5mm": metrics["carry_position_error_p95_m"] <= .005,
              "carry_orientation_p95_le_3deg": metrics["carry_orientation_error_p95_deg"] <= 3.,
              "carry_not_dropped": metrics["carry_min_bowl_height_delta_to_reference_m"] >= -.015
                  and metrics["carry_relative_bowl_eef_error_max_m"] <= .015,
              "carry_speedup_meets_declared_rate": (metrics["carry_control_step_speedup"] >= 1.5 if rate == 2
                                                     else bool(np.isclose(metrics["carry_control_step_speedup"], 1))),
              "continuous_physics_trajectory": metrics["continuous_physics_trajectory"],
              "actions_finite_and_bounded": metrics["actions_finite_and_bounded"],
              "reference_coverage_complete": metrics["reference_coverage_complete"]}
    return {key: bool(value) for key, value in checks.items()}


def state8(obs):
    from simulation_lab import quaternion_axis_angle
    return np.concatenate((obs["robot0_eef_pos"], quaternion_axis_angle(obs["robot0_eef_quat"]),
                           obs["robot0_gripper_qpos"])).astype(np.float32)


def camera_images(obs):
    return {name: np.ascontiguousarray(obs[key][::-1, ::-1]) for name, key in
            (("image", "agentview_image"), ("wrist_image", "robot0_eye_in_hand_image"))}


def restore_controller(env):
    records = []
    for robot in env.env.robots:
        robot.controller.update(force=True)
        robot.controller.reset_goal()
        ids = robot._ref_joint_gripper_actuator_indexes
        ranges = env.sim.model.actuator_ctrlrange[ids]
        bias, weight = ranges.mean(axis=1), (ranges[:, 1]-ranges[:, 0])/2
        qpos = np.asarray(env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes]).copy()
        robot.gripper.current_action = np.clip((qpos-bias)/weight, -1., 1.)
        env.sim.data.ctrl[ids] = bias+weight*robot.gripper.current_action
        records.append({"method": "update(force=True),reset_goal; inverse actuator-range gripper qpos mapping",
                        "not_exact_hidden_controller_restoration": True, "goal_pos": robot.controller.goal_pos.tolist(),
                        "goal_ori": robot.controller.goal_ori.tolist(), "initial_joint": robot.controller.initial_joint.tolist(),
                        "gripper_qpos": qpos.tolist(), "gripper_accumulator": robot.gripper.current_action.tolist()})
    return records


def generate_source(task, source_spec, output, reference_rate):
    import imageio.v2 as imageio
    from simulation_lab import TARGET, state_digest

    name, expected, carry = source_spec
    source = ROOT / "corrections" / name
    source_metadata = json.loads((source / "metadata.json").read_text())
    if (source_metadata["status"] != "approved" or source_metadata["split"] != "train"
            or source_metadata["frame_count"] != expected or source_metadata["task"] != task.task.language
            or source_metadata["scenario"]["layout_id"] != 24
            or source_metadata["scenario"]["task_metadata"] != task.metadata
            or not np.isclose(source_metadata["control_dt"], .05)):
        raise ValueError(f"Incompatible train source: {source}")
    frames = load_frames(source, source_metadata)
    approval_checks(frames)
    if not all(float(f["action"][6]) > 0 for f in frames[carry[0]:carry[1]]):
        raise ValueError("Carry interval includes an open command")
    initial = np.load(source / "initial_sim_state.npy", allow_pickle=False)
    if state_digest(initial) != source_metadata["initial_sim_state_sha256"]:
        raise ValueError("Initial state source checksum mismatch")
    output.mkdir(exist_ok=False)
    np.save(output / "initial_sim_state.npy", initial, allow_pickle=False)
    schedule = reference_schedule(expected, carry, rate=reference_rate)
    reference = np.asarray([index for index, _ in schedule], dtype=np.int64)
    np.save(output / "reference-index.npy", reference, allow_pickle=False)
    with (output / "reference-index.csv").open("x", newline="") as stream:
        csv.writer(stream).writerows([("generated_frame", "raw_reference_index", "reference_rate")]
                                     + [(i, index, rate) for i, (index, rate) in enumerate(schedule)])
    metadata = {"schema_version": 1, "generator_version": 2,
                "feedback_scope": "audited_carry_only", "noncarry_action_policy": "exact_source_action",
                "source_kind": "offline_tracking_teacher", "human_actions_only": False,
                "status": "candidate", "ready": False, "task": source_metadata["task"], "split": "train",
                "scenario_key": source_metadata["scenario_key"], "scenario": source_metadata["scenario"],
                "source_scene": source_metadata["source_scene"],
                "source_directory": str(source), "source_metadata_sha256": digest(source / "metadata.json"),
                "source_initial_sim_state_sha256": state_digest(initial), "initial_sim_state_sha256": state_digest(initial),
                "source_frame_count": expected, "reference_carry_interval": list(carry), "protected_source_steps": 10,
                "requested_reference_rate": reference_rate, "reference_rate": reference_rate,
                "role": "carry_rate2_treatment" if reference_rate == 2 else "carry_rate1_control",
                "control_dt": .05, "frame_count": 0, "files": [],
                "camera_orientation": source_metadata["camera_orientation"], "action_contract": source_metadata["action_contract"],
                "source_code_sha256": {name: digest(ROOT / "scripts" / name) for name in
                                       ["generate_tracking_teacher.py", "simulation_lab.py", "correction_data.py"]},
                "quality_gates": {name: False for name in GATE_NAMES}, "quality_metrics": {},
                "video": "trajectory.mp4", "video_fps": 20,
                "limits": ["A generated tracking teacher, never a human-approved recording.",
                           "Controller cache/goal/gripper restoration is consistent but not a serialized hidden-state restoration.",
                           "Ready requires both fixed source pilots to pass; task success alone is insufficient."]}
    write_json(output / "metadata.json", metadata)
    env, _ = task.create(24)
    try:
        obs = env.regenerate_obs_from_state(initial)
        if not np.array_equal(initial, env.get_sim_state()):
            raise ValueError("Physical initial state not exactly restored")
        metadata["controller_initialization"] = restore_controller(env)
        controller = env.env.robots[0].controller
        scale = controller.scale_action(np.ones(6))-controller.scale_action(np.zeros(6))
        if not np.allclose(scale, [.05, .05, .05, .5, .5, .5], atol=1e-12):
            raise ValueError(f"Unexpected OSC scale: {scale}")
        metadata["osc_scale"] = scale.tolist()
        t0, carry_pos, carry_ori, height_delta, relative_error = float(env.sim.data.time), [], [], [], []
        tail, all_actions, video_frames = [], [], 1
        initial_images = camera_images(obs)
        for key, value in initial_images.items():
            imageio.imwrite(output / f"initial-{key}.png", value)
        with imageio.get_writer(output / "trajectory.mp4", fps=20, codec="libx264", macro_block_size=16) as video:
            video.append_data(np.concatenate(list(initial_images.values()), axis=1))
            for generated_index, (raw_index, rate) in enumerate(schedule):
                pre = state8(obs)
                action, error = frame_action(frames[raw_index]["action"], frames[raw_index]["state"], pre,
                                             rate, scale, raw_index, carry)
                if raw_index in (carry[0], carry[0]+10):
                    grip = {"generated_frame": generated_index, "raw_reference": raw_index,
                            "actual_finger_qpos": pre[6:8].tolist(),
                            "actual_gap_m": float(np.abs(pre[6:8]).sum()),
                            "raw_gap_m": float(np.abs(frames[raw_index]["state"][6:8]).sum())}
                    metadata["carry_entry_grip" if raw_index == carry[0] else "accelerated_interval_entry_grip"] = grip
                    print(name, "CARRY_GRIP", json.dumps(grip), flush=True)
                images = camera_images(obs)
                _, _, _, _ = env.step(action.tolist())
                obs = env.regenerate_obs_from_state(env.get_sim_state())
                post, bowl = state8(obs), np.asarray(obs[f"{TARGET}_pos"], dtype=np.float32)
                done = bool(env.check_success())
                frame = {"state": pre, "action": action, **images, "post_state": post,
                         "post_bowl_pos": bowl, "done": np.array(done)}
                validate_frame(frame)
                path = output / f"frame-{generated_index:06d}.npz"
                with path.open("xb") as handle:
                    np.savez_compressed(handle, **frame)
                metadata["files"].append({"path": path.name, "sha256": digest(path)})
                metadata["frame_count"] = generated_index+1
                all_actions.append(action)
                tail = (tail+[frame])[-10:]
                if carry[0] <= raw_index < carry[1]:
                    carry_pos.append(float(np.linalg.norm(error[:3])))
                    carry_ori.append(float(np.rad2deg(np.linalg.norm(error[3:]))))
                    next_reference = min(raw_index+rate, expected-1)
                    expected_post = frames[max(0, next_reference-1)]
                    ref_bowl, ref_eef = expected_post["post_bowl_pos"], expected_post["post_state"][:3]
                    height_delta.append(float(bowl[2]-ref_bowl[2]))
                    relative_error.append(float(np.linalg.norm((bowl-post[:3])-(ref_bowl-ref_eef))))
                fresh_images = camera_images(obs)
                video.append_data(np.concatenate(list(fresh_images.values()), axis=1))
                video_frames += 1
                if (generated_index+1) % 100 == 0:
                    write_json(output / "metadata.json", metadata)
                    print(name, generated_index+1, "/", len(schedule), flush=True)
        np.save(output / "final_sim_state.npy", env.get_sim_state(), allow_pickle=False)
        for key, value in camera_images(obs).items():
            imageio.imwrite(output / f"final-{key}.png", value)
        actual_steps = len(schedule)
        carry_steps = sum(carry[0] <= index < carry[1] for index, _ in schedule)
        actions = np.asarray(all_actions)
        metrics = {"terminal_on_last_ten": len(tail) == 10 and all(bool(f["done"]) for f in tail),
                   "terminal_min_gap_m": min(float(np.abs(f["post_state"][6:8]).sum()) for f in tail),
                   "terminal_open_command_last_ten": len(tail) == 10 and all(float(f["action"][6]) < 0 for f in tail),
                   "final_retreat_m": float(np.linalg.norm(tail[-1]["post_state"][:3]-tail[-1]["post_bowl_pos"])),
                   "carry_position_error_p95_m": float(np.percentile(carry_pos, 95)),
                   "carry_orientation_error_p95_deg": float(np.percentile(carry_ori, 95)),
                   "carry_min_bowl_height_delta_to_reference_m": min(height_delta),
                   "carry_relative_bowl_eef_error_max_m": max(relative_error),
                   "carry_original_control_steps": carry[1]-carry[0], "carry_generated_control_steps": carry_steps,
                   "carry_control_step_speedup": (carry[1]-carry[0])/carry_steps,
                   "whole_episode_original_steps": expected, "whole_episode_generated_steps": actual_steps,
                   "whole_episode_control_step_speedup": expected/actual_steps,
                   "simulation_duration_s": float(env.sim.data.time)-t0,
                   "continuous_physics_trajectory": bool(np.isclose(float(env.sim.data.time)-t0, actual_steps*.05, atol=1e-7)),
                   "actions_finite_and_bounded": bool(np.isfinite(actions).all() and np.max(np.abs(actions)) <= 1),
                   "reference_coverage_complete": bool(reference[0] == 0 and reference[-1] == expected-1
                                                        and np.all(np.diff(reference) > 0)),
                   "video_frames": video_frames}
        metadata.update(status="quality_checked_pending_pair", quality_metrics=metrics, quality_gates=quality_gates(metrics, reference_rate),
                        final_sim_state_sha256=state_digest(env.get_sim_state()),
                        source_metadata_unchanged=digest(source / "metadata.json") == metadata["source_metadata_sha256"])
        write_json(output / "metadata.json", metadata)
        return metadata
    finally:
        env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rate", type=int, choices=(1, 2), default=2)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"ready": False, "status": "RUNNING", "source_kind": "offline_tracking_teacher",
              "reference_rate": args.rate, "feedback_scope": "audited_carry_only",
              "source_order": [spec[0] for spec in reversed(SOURCES)], "sources": []}
    write_json(args.output / "report.json", report)
    try:
        from simulation_lab import LiberoTask
        task = LiberoTask()
        for spec in reversed(SOURCES):
            metadata = generate_source(task, spec, args.output / spec[0], args.rate)
            report["sources"].append({"directory": str((args.output/spec[0]).resolve()),
                                      "quality_gates": metadata["quality_gates"], "quality_metrics": metadata["quality_metrics"]})
            write_json(args.output / "report.json", report)
            if not all(metadata["quality_gates"].values()):
                break
        ready = all(all(s["quality_gates"].values()) for s in report["sources"]) and len(report["sources"]) == 2
        report.update(ready=ready, status="PASS" if ready else "REJECTED_NO_TRAINING_READY_TEACHER")
        for source in report["sources"]:
            path = Path(source["directory"]) / "metadata.json"
            metadata = json.loads(path.read_text())
            metadata.update(ready=ready, status="validated" if ready else "rejected")
            write_json(path, metadata)
    except Exception as error:
        report.update(status="ERROR_STOPPED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(args.output / "report.json", report)
    print(report["status"], args.output, flush=True)
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
