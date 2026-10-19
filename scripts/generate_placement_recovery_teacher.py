"""Offline, privileged-state placement teacher; never a runtime policy or human recording."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from correction_data import digest, validate_frame, write_json
from generate_tracking_teacher import camera_images, restore_controller, state8

ROOT = Path(__file__).resolve().parents[1]


def position_action(error, gripper, limit):
    error = np.asarray(error, dtype=float)
    if error.shape != (3,) or not np.isfinite(error).all() or not 0 < limit <= 1 or gripper not in (-1, 1):
        raise ValueError("Invalid bounded translation input")
    xyz = error / .05
    xyz *= min(1., limit / max(float(np.linalg.norm(xyz)), 1e-12))
    return np.r_[xyz, np.zeros(3), gripper].astype(np.float32)


def box_vertical_bounds(position, rotation, half_size):
    position, rotation, half_size = map(np.asarray, (position, rotation, half_size))
    if position.shape != (3,) or rotation.shape != (3, 3) or half_size.shape != (3,):
        raise ValueError("Expected box pose and half size")
    extent = np.abs(rotation[2]) @ half_size
    return float(position[2] - extent), float(position[2] + extent)


def collision_bounds(env, name):
    bounds = []
    for geom in env.env.objects_dict[name].contact_geoms:
        gid = env.sim.model.geom_name2id(geom)
        if int(env.sim.model.geom_type[gid]) != 6:
            raise ValueError("This bounded pilot supports box collision geometry only")
        bounds.append(box_vertical_bounds(env.sim.data.geom_xpos[gid],
                                         env.sim.data.geom_xmat[gid].reshape(3, 3),
                                         env.sim.model.geom_size[gid]))
    return min(x[0] for x in bounds), max(x[1] for x in bounds)


def released(frame):
    return (bool(frame["done"]) and frame["action"][6] < 0
            and np.abs(frame["post_state"][6:8]).sum() >= .06
            and np.linalg.norm(frame["post_state"][:3] - frame["post_bowl_pos"]) >= .10)


def terminal_gates(frames, verified_step, continuous, unexpected_collision):
    tail = frames[-10:]
    followup = frames[verified_step:] if verified_step is not None else []
    stable = bool(followup) and max(np.linalg.norm(f["post_bowl_pos"] - frames[verified_step-1]["post_bowl_pos"])
                                    for f in followup) <= .005
    return {"terminal_ten_on_open_gap_retreat": len(tail) == 10 and all(released(f) for f in tail),
            "terminal_sixty_on_open_gap_retreat": len(frames) >= 60 and all(released(f) for f in frames[-60:]),
            "fifty_followup_steps": len(followup) == 50,
            "followup_all_released": len(followup) == 50 and all(released(f) for f in followup),
            "followup_bowl_motion_le_5mm": bool(stable),
            "continuous_physics": bool(continuous),
            "no_unexpected_deep_target_contact": not unexpected_collision,
            "finite_bounded_actions": bool(frames) and all(np.isfinite(f["action"]).all()
                                                           and np.abs(f["action"]).max() <= 1 for f in frames)}


def generate(source, output):
    import imageio.v2 as imageio
    from libero.libero.envs import OffScreenRenderEnv
    from simulation_lab import LiberoTask, TARGET, state_digest

    source, output = Path(source).resolve(), Path(output).resolve()
    result = json.loads((source / "result.json").read_text())
    initial = np.load(source / "final_state.npy", allow_pickle=False)
    if state_digest(initial) != result["final_state_sha256"]:
        raise ValueError("Saved failure state hash mismatch")
    task = LiberoTask()
    if result["task_metadata"] != task.metadata or result["layout_id"] != 24:
        raise ValueError("Task/layout mismatch")
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"schema_version": 1, "source_kind": "offline_placement_recovery_teacher",
                "human_actions_only": False, "approval_type": None,
                "ready": False, "status": "running", "split": "train",
                "task": result["task_language"], "task_metadata": task.metadata, "layout_id": 24,
                "scenario_key": f"placement_recovery_{source.parent.name}_{source.name}_final",
                "scenario": json.loads((source / "scene.json").read_text()),
                "source_scene": str(source.relative_to(ROOT)), "source_directory": str(source),
                "source_state_file": str((source / "final_state.npy").relative_to(ROOT)),
                "source_state_file_sha256": digest(source / "final_state.npy"),
                "source_files": {name: digest(source / name) for name in
                                 ["result.json", "scene.json", "final_state.npy", "actions.csv"]},
                "initial_sim_state_sha256": state_digest(initial), "frame_count": 0, "files": [],
                "control_dt": .05, "max_control_steps": 500, "video": "trajectory.mp4", "video_fps": 20,
                "camera_orientation": "RGB double-axis flipped exactly once from simulator output",
                "source_code_sha256": {name: digest(ROOT / "scripts" / name) for name in
                                       ["generate_placement_recovery_teacher.py", "generate_tracking_teacher.py", "simulation_lab.py"]},
                "limits": ["Privileged object poses are used only by this offline teacher; no runtime policy modification.",
                           "Consistent cache/goal/gripper initialization is not exact hidden-controller restoration.",
                           "This recovery starts already holding a bowl; it does not teach approach or failed-grasp recovery."]}
    write_json(output / "metadata.json", metadata)
    env = OffScreenRenderEnv(bddl_file_name=str(task.bddl), camera_heights=256, camera_widths=256, ignore_done=True)
    frames, trajectory, verified_step = [], [], None
    unexpected, continuity = [], True
    try:
        env.seed(42)
        env.reset()
        obs = env.regenerate_obs_from_state(initial)
        if not np.array_equal(initial, env.get_sim_state()):
            raise ValueError("Integration state restore is not exact")
        metadata["controller_initialization"] = restore_controller(env)
        np.save(output / "initial_sim_state.npy", initial)
        plate = np.asarray(obs["plate_1_pos"]).copy()
        bowl = np.asarray(obs[f"{TARGET}_pos"]).copy()
        bottom, _ = collision_bounds(env, TARGET)
        _, top = collision_bounds(env, "plate_1")
        release_z = top - (bottom - bowl[2]) + .001
        carry_z = max(bowl[2], release_z + .12)
        metadata["geometry"] = {"plate_xyz": plate.tolist(), "initial_bowl_xyz": bowl.tolist(),
                                "initial_eef_minus_bowl": (np.asarray(obs["robot0_eef_pos"]) - bowl).tolist(),
                                "plate_collision_top_z": top, "bowl_collision_bottom_z": bottom,
                                "release_bowl_center_z": release_z, "carry_bowl_center_z": carry_z,
                                "release_clearance_m": .001}
        phase, phase_steps, streak = "carry", 0, 0
        hold_eef = None
        t0 = float(env.sim.data.time)
        with imageio.get_writer(output / "trajectory.mp4", fps=20, codec="libx264", macro_block_size=16) as video:
            first_images = camera_images(obs)
            for key, image in first_images.items():
                imageio.imwrite(output / f"initial-{key}.png", image)
            video.append_data(np.concatenate(list(first_images.values()), axis=1))
            for index in range(500):
                pre, images = state8(obs), camera_images(obs)
                bowl = np.asarray(obs[f"{TARGET}_pos"])
                if phase == "carry" and np.linalg.norm(np.r_[plate[:2] - bowl[:2], carry_z - bowl[2]]) < .006:
                    phase, phase_steps = "lower", 0
                if phase == "lower" and np.linalg.norm(np.r_[plate[:2] - bowl[:2], release_z - bowl[2]]) < .003:
                    phase, phase_steps, hold_eef = "release", 0, pre[:3].copy()
                if phase == "release" and phase_steps >= 12 and np.abs(pre[6:8]).sum() >= .065:
                    phase, phase_steps, hold_eef = "retreat", 0, pre[:3] + [0, 0, .15]
                if phase in ("carry", "lower"):
                    goal = np.r_[plate[:2], carry_z if phase == "carry" else release_z]
                    action = position_action(goal - bowl, 1, .30 if phase == "carry" else .18)
                else:
                    action = position_action(hold_eef - pre[:3], -1, .20)
                env.step(action.tolist())
                obs = env.regenerate_obs_from_state(env.get_sim_state())
                frame = {"state": pre, "action": action, **images, "post_state": state8(obs),
                         "post_bowl_pos": np.asarray(obs[f"{TARGET}_pos"], dtype=np.float32),
                         "done": np.array(bool(env.check_success()))}
                validate_frame(frame)
                if frames:
                    continuity &= np.array_equal(frames[-1]["post_state"], pre)
                name = f"frame-{index:06d}.npz"
                np.savez_compressed(output / name, **frame)
                metadata["files"].append({"path": name, "sha256": digest(output / name)})
                frames.append(frame)
                contacts = []
                for contact in env.sim.data.contact[:env.sim.data.ncon]:
                    names = [env.sim.model.geom_id2name(int(g)) for g in (contact.geom1, contact.geom2)]
                    if any(TARGET in str(n) for n in names):
                        other = next((n for n in names if TARGET not in str(n)), "")
                        contacts.append({"geoms": names, "distance_m": float(contact.dist)})
                        if contact.dist < -.001 and not any(s in str(other) for s in ("gripper", "plate_1")):
                            unexpected.append({"frame": index, "geoms": names, "distance_m": float(contact.dist)})
                trajectory.append({"frame": index, "phase": phase, "bowl_xyz": frame["post_bowl_pos"].tolist(),
                                   "eef_xyz": frame["post_state"][:3].tolist(), "action": action.tolist(),
                                   "on": bool(frame["done"]), "gap_m": float(np.abs(frame["post_state"][6:8]).sum()),
                                   "retreat_m": float(np.linalg.norm(frame["post_state"][:3] - frame["post_bowl_pos"])),
                                   "target_contacts": contacts})
                video.append_data(np.concatenate(list(camera_images(obs).values()), axis=1))
                phase_steps += 1
                streak = streak + 1 if released(frame) else 0
                if verified_step is None and streak >= 10:
                    verified_step = len(frames)
                    phase, phase_steps, hold_eef = "followup", 0, frame["post_state"][:3].copy()
                if verified_step is not None and len(frames) - verified_step == 50:
                    break
        np.save(output / "final_sim_state.npy", env.get_sim_state())
        for key, image in camera_images(obs).items():
            imageio.imwrite(output / f"final-{key}.png", image)
        write_json(output / "trajectory.json", trajectory)
        with (output / "phase.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["frame", "phase", "on", "bowl_x", "bowl_y", "bowl_z", "plate_x", "plate_y", "plate_z"])
            writer.writerows((r["frame"], r["phase"], r["on"], *r["bowl_xyz"], *plate) for r in trajectory)
        continuity &= np.isclose(float(env.sim.data.time) - t0, len(frames) * .05, atol=1e-7)
        gates = terminal_gates(frames, verified_step, continuity, bool(unexpected))
        metadata.update(frame_count=len(frames), video_frames=len(frames) + 1,
                        quality_gates=gates, ready=all(gates.values()),
                        status="validated" if all(gates.values()) else "rejected",
                        first_ten_step_release_retreat_control_steps=verified_step,
                        final_sim_state_sha256=state_digest(env.get_sim_state()), unexpected_contacts=unexpected,
                        quality_metrics={"terminal_min_gap_m": min(float(np.abs(f["post_state"][6:8]).sum()) for f in frames[-10:]),
                                         "final_retreat_m": trajectory[-1]["retreat_m"],
                                         "final_on": trajectory[-1]["on"], "simulation_duration_s": float(env.sim.data.time) - t0},
                        source_files_unchanged=all(digest(source / f) == h for f, h in metadata["source_files"].items()))
        if not metadata["source_files_unchanged"]:
            metadata.update(ready=False, status="rejected")
        write_json(output / "metadata.json", metadata)
        return metadata
    except Exception as exc:
        metadata.update(status="error", ready=False, frame_count=len(frames), error=f"{type(exc).__name__}: {exc}")
        write_json(output / "metadata.json", metadata)
        raise
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = generate(args.source, args.output)
    print(json.dumps({k: metadata[k] for k in ("status", "ready", "frame_count", "quality_gates", "quality_metrics")}))
    return 0 if metadata["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
