"""Continuous offline full-task geometry teacher; privileged state is never a policy input."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from correction_data import digest, validate_frame, write_json
from generate_placement_recovery_teacher import collision_bounds, released, terminal_gates
from generate_tracking_teacher import camera_images, restore_controller, state8, world_rotation_error

ROOT = Path(__file__).resolve().parents[1]


def servo_action(goal, actual, grip, limit):
    goal, actual = np.asarray(goal), np.asarray(actual)
    if (goal.shape != (6,) or actual.shape != (8,) or not np.isfinite(goal).all()
            or not np.isfinite(actual).all() or grip not in (-1, 1) or not 0 < limit <= 1):
        raise ValueError("Invalid servo input")
    xyz = (goal[:3] - actual[:3]) / .05
    xyz *= min(1., limit / max(float(np.linalg.norm(xyz)), 1e-12))
    rotation = world_rotation_error(goal[3:6], actual[3:6]) / .5
    rotation *= min(1., .4 / max(float(np.linalg.norm(rotation)), 1e-12))
    return np.r_[xyz, rotation, grip].astype(np.float32)


def load_reference(reference, target):
    reference, target = Path(reference), Path(target)
    old = json.loads((reference / "scene.json").read_text())
    new = json.loads((target / "scene.json").read_text())
    result = json.loads((reference / "result.json").read_text())
    if (not result["success"] or old["task_metadata"] != new["task_metadata"]
            or old["task_language"] != new["task_language"] or new["layout_id"] != 24):
        raise ValueError("Reference must be a successful matching task/layout")
    rows = list(csv.DictReader((reference / "actions.csv").open()))
    states = np.asarray([[float(r[f"state_{i}"]) for i in range(8)] for r in rows])
    # The reference's sustained closed grasp precedes target lift; noisy late release is not replayed.
    close = np.asarray([float(r["action_6"]) > 0 for r in rows])
    grasp = next(i for i in range(len(rows)-10) if close[i:i+10].all()) + 6
    bowl = np.asarray([[float(r[f"post_bowl_{i}"]) for i in range(3)] for r in rows])
    lift = next(i for i in range(grasp, len(rows)) if close[i] and bowl[i, 2]-bowl[0, 2] > .04)
    on = next(i for i, r in enumerate(rows) if float(r["reward"]) > 0)
    def after(scene):
        return np.asarray(scene["bowl_edit"]["after_xyz"])
    return states, grasp, lift, on, after(new)-after(old), new


def generate(reference, source, output, probe=False):
    import imageio.v2 as imageio
    from libero.libero.envs import OffScreenRenderEnv
    from simulation_lab import LiberoTask, TARGET, state_digest

    reference, source, output = map(lambda x: Path(x).resolve(), (reference, source, output))
    states, grasp, lift, on_index, delta, scene = load_reference(reference, source)
    initial = np.load(source / "initial_state.npy", allow_pickle=False)
    if state_digest(initial) != scene["initial_state_sha256"]:
        raise ValueError("Target initial state checksum mismatch")
    output.mkdir(parents=True, exist_ok=False)
    codes = ["generate_retargeted_task_teacher.py", "generate_tracking_teacher.py", "simulation_lab.py",
             "generate_placement_recovery_teacher.py"]
    metadata = {"schema_version": 1, "source_kind": "offline_retargeted_task_teacher",
                "human_actions_only": False, "approval_type": None, "split": "train",
                "ready": False, "status": "running", "probe": probe,
                "task": scene["task_language"], "task_metadata": scene["task_metadata"],
                "scenario_key": f"retarget_{source.parent.name}_{source.name}", "scenario": scene,
                "source_scene": str(source.relative_to(ROOT)),
                "source_state_file": str((source / "initial_state.npy").relative_to(ROOT)),
                "source_state_file_sha256": digest(source / "initial_state.npy"),
                "source_files": {n: digest(source/n) for n in ("scene.json", "initial_state.npy")},
                "reference_scene": str(reference.relative_to(ROOT)),
                "reference_source_files": {n: digest(reference/n) for n in ("scene.json", "result.json", "actions.csv")},
                "initial_sim_state_sha256": state_digest(initial), "control_dt": .05,
                "max_control_steps": 600, "frame_count": 0, "files": [],
                "source_code_sha256": {n: digest(ROOT/"scripts"/n) for n in codes},
                "video": "trajectory.mp4", "video_fps": 20,
                "camera_orientation": "RGB double-axis flipped exactly once",
                "action_contract": "fresh pre-action RGB/state; normalized OSC world XYZ(.05m)/world rotation(.5rad); +1 close,-1 open",
                "algorithm": "006-derived translated grasp/hover geometry; privileged bowl-relative closed-loop carry/lower to unchanged original plate; no pointwise reference tracking",
                "reference_geometry": {"grasp_index": grasp, "lift_index": lift, "on_index": on_index,
                                       "grasp_translation_delta_m": delta.tolist()},
                "limits": ["Offline privileged geometry teacher; not human approval or policy evaluation.",
                           "One continuous env.step trajectory from exact saved initial state; no intermediate teleport.",
                           "Controller cache initialization does not restore serialized hidden controller state."]}
    write_json(output/"metadata.json", metadata)
    task = LiberoTask()
    if task.metadata != scene["task_metadata"]:
        raise ValueError("Installed task differs from source")
    env = OffScreenRenderEnv(bddl_file_name=str(task.bddl), camera_heights=256, camera_widths=256, ignore_done=True)
    frames, trajectory, unexpected = [], [], []
    verified, continuity, streak = None, True, 0
    phase, phase_steps = "hover", 0
    try:
        env.seed(42)
        env.reset()
        obs = env.regenerate_obs_from_state(initial)
        if not np.array_equal(initial, env.get_sim_state()):
            raise ValueError("Initial integration state did not restore exactly")
        metadata["controller_initialization"] = restore_controller(env)
        controller = env.env.robots[0].controller
        scale = controller.scale_action(np.ones(6))-controller.scale_action(np.zeros(6))
        if not np.allclose(scale, [.05, .05, .05, .5, .5, .5], rtol=0, atol=1e-12):
            raise ValueError("Unexpected OSC world action scale")
        metadata["osc_scale"] = scale.tolist()
        np.save(output/"initial_sim_state.npy", initial)
        pre = state8(obs)
        initial_bowl = np.asarray(obs[f"{TARGET}_pos"]).copy()
        plate = np.asarray(obs["plate_1_pos"]).copy()
        grasp_pose = states[grasp, :6].copy()
        grasp_pose[:3] += delta
        carry_z = max(states[lift:on_index, 2].max()+.015, grasp_pose[2]+.16)
        hover = grasp_pose.copy()
        hover[2] += .13
        goal = hover.copy()
        t0 = float(env.sim.data.time)
        hold = None
        with imageio.get_writer(output/"trajectory.mp4", fps=20, codec="libx264", macro_block_size=16) as video:
            for key, im in camera_images(obs).items():
                imageio.imwrite(output/f"initial-{key}.png", im)
            video.append_data(np.concatenate(list(camera_images(obs).values()), axis=1))
            for index in range(600):
                pre, images = state8(obs), camera_images(obs)
                bowl = np.asarray(obs[f"{TARGET}_pos"])
                if phase == "hover" and np.linalg.norm(pre[:3]-hover[:3]) < .004:
                    phase, phase_steps = "descend", 0
                if phase == "descend" and np.linalg.norm(pre[:3]-grasp_pose[:3]) < .0025:
                    phase, phase_steps = "close", 0
                if phase == "close" and phase_steps >= 24:
                    phase, phase_steps = "lift", 0
                if phase == "lift" and pre[2] >= carry_z-.006:
                    if bowl[2]-initial_bowl[2] < .04:
                        metadata["rejection_reason"] = "Target failed verified lift"
                        break
                    phase, phase_steps = "carry", 0
                if phase == "carry" and np.linalg.norm(bowl[:2]-plate[:2]) < .005:
                    phase, phase_steps = "lower", 0
                if phase == "lower":
                    bottom, _ = collision_bounds(env, TARGET)
                    _, plate_top = collision_bounds(env, "plate_1")
                    release_z = plate_top - (bottom-bowl[2]) + .001
                    if np.linalg.norm(np.r_[plate[:2]-bowl[:2], release_z-bowl[2]]) < .003:
                        phase, phase_steps, hold = "release", 0, pre[:6].copy()
                if phase == "release" and phase_steps >= 16 and np.abs(pre[6:8]).sum() >= .065:
                    phase, phase_steps = "retreat", 0
                    hold = pre[:6].copy()
                    hold[2] += .15
                grip, limit = (-1, .65) if phase == "hover" else (1, .45)
                if phase == "hover":
                    goal = hover.copy()
                elif phase in ("descend", "close"):
                    goal, limit = grasp_pose.copy(), .20
                    grip = -1 if phase == "descend" else 1
                elif phase == "lift":
                    goal = grasp_pose.copy()
                    goal[2] = carry_z
                elif phase in ("carry", "lower"):
                    # Target-relative feedback continuously compensates small grasp/rotation shifts.
                    desired = np.r_[plate[:2], carry_z-.03 if phase == "carry" else release_z]
                    goal = np.r_[pre[:3]+desired-bowl, states[on_index, 3:6]]
                    limit = .55 if phase == "carry" else .18
                else:
                    goal, grip, limit = hold.copy(), -1, .25
                action = servo_action(goal, pre, grip, limit)
                env.step(action.tolist())
                obs = env.regenerate_obs_from_state(env.get_sim_state())
                frame = {"state": pre, "action": action, **images, "post_state": state8(obs),
                         "post_bowl_pos": np.asarray(obs[f"{TARGET}_pos"], dtype=np.float32),
                         "done": np.array(bool(env.check_success()))}
                validate_frame(frame)
                if frames:
                    continuity &= np.array_equal(frames[-1]["post_state"], pre)
                frames.append(frame)
                if not probe:
                    name = f"frame-{index:06d}.npz"
                    np.savez_compressed(output/name, **frame)
                    metadata["files"].append({"path": name, "sha256": digest(output/name)})
                for contact in env.sim.data.contact[:env.sim.data.ncon]:
                    names = [env.sim.model.geom_id2name(int(g)) for g in (contact.geom1, contact.geom2)]
                    if any(TARGET in str(n) for n in names):
                        other = next((n for n in names if TARGET not in str(n)), "")
                        if contact.dist < -.001 and not any(s in str(other) for s in ("gripper", "plate_1")):
                            unexpected.append({"frame": index, "geoms": names, "distance_m": float(contact.dist)})
                trajectory.append({"frame": index, "phase": phase, "on": bool(frame["done"]),
                                   "bowl_xyz": frame["post_bowl_pos"].tolist(), "eef_xyz": frame["post_state"][:3].tolist(),
                                   "action": action.tolist(), "gap_m": float(np.abs(frame["post_state"][6:8]).sum()),
                                   "goal_xyz": goal[:3].tolist()})
                video.append_data(np.concatenate(list(camera_images(obs).values()), axis=1))
                phase_steps += 1
                streak = streak+1 if released(frame) else 0
                if verified is None and streak >= 10:
                    verified, phase, phase_steps = len(frames), "followup", 0
                    hold = frame["post_state"][:6].copy()
                if index % 25 == 0:
                    print(index, phase, "bowl", bowl.round(4).tolist(), "gap", round(trajectory[-1]["gap_m"], 4), flush=True)
                if verified is not None and len(frames)-verified == 50:
                    break
                if phase_steps > 150:
                    metadata["rejection_reason"] = f"Bounded phase failed to converge: {phase}"
                    break
        duration = float(env.sim.data.time)-t0
        continuity &= bool(np.isclose(duration, len(frames)*.05, rtol=0, atol=1e-7))
        gates = terminal_gates(frames, verified, continuity, bool(unexpected))
        np.save(output/"final_sim_state.npy", env.get_sim_state())
        for key, im in camera_images(obs).items():
            imageio.imwrite(output/f"final-{key}.png", im)
        write_json(output/"trajectory.json", trajectory)
        with (output/"phase.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "phase", "on", "gap_m", "bowl_x", "bowl_y", "bowl_z", "eef_x", "eef_y", "eef_z"])
            writer.writerows((r["frame"], r["phase"], r["on"], r["gap_m"], *r["bowl_xyz"], *r["eef_xyz"]) for r in trajectory)
        unchanged = (all(digest(source/n) == sha for n, sha in metadata["source_files"].items())
                     and all(digest(reference/n) == sha for n, sha in metadata["reference_source_files"].items()))
        passed = all(gates.values()) and unchanged
        metadata.update(frame_count=len(frames), video_frames=len(frames)+1, quality_gates=gates,
                        quality_metrics={"simulation_duration_s": duration}, unexpected_contacts=unexpected,
                        first_ten_step_release_retreat_control_steps=verified,
                        source_files_unchanged=unchanged, final_sim_state_sha256=state_digest(env.get_sim_state()),
                        ready=bool(passed and not probe), status="probe_passed" if passed and probe else "validated" if passed else "rejected")
        write_json(output/"metadata.json", metadata)
        return metadata
    except Exception as exc:
        metadata.update(ready=False, status="error", frame_count=len(frames), error=f"{type(exc).__name__}: {exc}")
        write_json(output/"metadata.json", metadata)
        raise
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    result = generate(args.reference, args.source, args.output, args.probe)
    print(json.dumps({k: result[k] for k in ("status", "ready", "frame_count", "quality_gates")}))
    return 0 if result["status"] in ("validated", "probe_passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
