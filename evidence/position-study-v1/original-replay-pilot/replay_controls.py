"""Two bounded original-action replay controls; produces diagnostic artifacts only."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from correction_data import approval_checks, digest, load_frames
from simulation_lab import LiberoTask, TARGET, quaternion_axis_angle, state_digest

HERE = Path(__file__).parent
REPORT = HERE / "report.json"
SOURCES = [("human_1792316444273677599", 796), ("human_1792316988593839967", 735)]
report = {"status": "RUNNING", "trials": [], "training_data_generated": False,
          "policy_loaded": False, "gui_used": False, "original_speed": "1x; exact source action per control step",
          "control_dt_s": .05, "limits": [
              "MuJoCo physical state is restored; serialized controller goals/gripper accumulator/internal state are unavailable.",
              "Fresh create(24) controller state is retained, not claimed to reproduce original controller internals.",
              "Fresh post observation is next-step pre observation; no physics step between them. Returned-step observations are also retained for matching original reference semantics.",
              "This pilot generates diagnostics only; it does not edit labels, approve new demonstrations, retime actions, or write training data."]}


def save_report():
    temporary = REPORT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(REPORT)


def state8(obs):
    return np.concatenate([obs["robot0_eef_pos"], quaternion_axis_angle(obs["robot0_eef_quat"]),
                           obs["robot0_gripper_qpos"]])


def images(directory, prefix, obs):
    for name, key in [("agentview", "agentview_image"), ("wrist", "robot0_eye_in_hand_image")]:
        imageio.imwrite(directory / f"{prefix}-{name}.png", np.ascontiguousarray(obs[key][::-1, ::-1]))


def gates(frames):
    tail = frames[-10:]
    checks = {"ten_frames": len(tail) == 10,
              "task_done_last_ten": len(tail) == 10 and all(bool(f["done"]) for f in tail),
              "actual_fingers_open_last_ten": len(tail) == 10 and all(float(np.abs(f["post_state"][6:8]).sum()) >= .06 for f in tail),
              "open_command_last_ten": len(tail) == 10 and all(float(f["action"][6]) < 0 for f in tail),
              "eef_clear_of_bowl": bool(tail) and float(np.linalg.norm(tail[-1]["post_state"][:3] - tail[-1]["post_bowl_pos"])) >= .10}
    return {"checks": checks, "pass": all(checks.values())}


save_report()
task = LiberoTask()
report["task_metadata"] = task.metadata
report["source_code_sha256"] = {name: digest(ROOT / "scripts" / name) for name in
                                 ["simulation_lab.py", "correction_data.py", "correction_workbench.py"]}
for name, expected_count in SOURCES:
    directory = ROOT / "corrections" / name
    metadata = json.loads((directory / "metadata.json").read_text())
    if (metadata["status"] != "approved" or metadata["split"] != "train"
            or metadata["frame_count"] != expected_count or metadata["task"] != task.task.language
            or metadata["scenario"]["layout_id"] != 24 or metadata["scenario"]["task_metadata"] != task.metadata
            or not np.isclose(metadata["control_dt"], .05)):
        raise ValueError(f"Source metadata incompatibility: {name}")
    frames = load_frames(directory, metadata)
    approval_checks(frames)
    initial = np.load(directory / "initial_sim_state.npy", allow_pickle=False)
    if state_digest(initial) != metadata["initial_sim_state_sha256"]:
        raise ValueError(f"Initial physical state checksum mismatch: {name}")
    out = HERE / name
    out.mkdir(exist_ok=False)
    trial = {"source": str(directory), "source_metadata_sha256": digest(directory / "metadata.json"),
             "source_initial_state_file_sha256": digest(directory / "initial_sim_state.npy"),
             "source_initial_state_array_sha256": state_digest(initial), "source_frame_count": expected_count,
             "source_split": metadata["split"], "source_scenario": metadata["scenario"],
             "source_frame_hashes_verified": True, "source_original_approval_gates_pass": True,
             "status": "RUNNING", "output": str(out)}
    report["trials"].append(trial)
    save_report()
    env, _ = task.create(24)
    started = time.monotonic()
    try:
        obs = env.regenerate_obs_from_state(initial)
        trial["physical_state_exactly_restored_before_actions"] = bool(np.array_equal(initial, env.get_sim_state()))
        trial["first_fresh_state8_error"] = (state8(obs) - frames[0]["state"]).tolist()
        trial["controller_initialization"] = []
        for robot in env.env.robots:
            controller = {"controller_class": type(robot.controller).__name__,
                          "gripper_current_action": np.asarray(robot.gripper.current_action).tolist()}
            for attr in ["goal_pos", "goal_ori", "initial_joint"]:
                value = getattr(robot.controller, attr, None)
                controller[attr] = None if value is None else np.asarray(value).tolist()
            trial["controller_initialization"].append(controller)
        images(out, "initial", obs)
        current_states, returned_states, fresh_post_states, bowls, returned_bowls, actions = [], [], [], [], [], []
        actual_tail, returned_tail = [], []
        pre_errors, post_errors, returned_errors, bowl_errors = [], [], [], []
        with (out / "trajectory.csv").open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["step", "on_fresh", "on_returned", "pre_eef_error_m", "fresh_post_eef_error_m",
                             "returned_post_eef_error_m", "fresh_bowl_error_m", "fresh_finger_gap_m", "fresh_eef_bowl_distance_m"]
                            + [f"fresh_post_eef_{i}" for i in range(3)] + [f"fresh_bowl_{i}" for i in range(3)]
                            + [f"reference_post_eef_{i}" for i in range(3)] + [f"action_{i}" for i in range(7)])
            for index, frame in enumerate(frames):
                pre = state8(obs)
                returned, _, done_returned, _ = env.step(frame["action"].tolist())
                returned_state = state8(returned)
                returned_bowl = np.asarray(returned[f"{TARGET}_pos"]).copy()
                obs = env.regenerate_obs_from_state(env.get_sim_state())
                post = state8(obs)
                bowl = np.asarray(obs[f"{TARGET}_pos"]).copy()
                done = bool(env.check_success())
                error_pre = float(np.linalg.norm(pre[:3] - frame["state"][:3]))
                error_post = float(np.linalg.norm(post[:3] - frame["post_state"][:3]))
                error_returned = float(np.linalg.norm(returned_state[:3] - frame["post_state"][:3]))
                error_bowl = float(np.linalg.norm(bowl - frame["post_bowl_pos"]))
                writer.writerow([index, done, bool(done_returned), error_pre, error_post, error_returned, error_bowl,
                                 float(np.abs(post[6:8]).sum()), float(np.linalg.norm(post[:3] - bowl))]
                                + post[:3].tolist() + bowl.tolist() + frame["post_state"][:3].tolist() + frame["action"].tolist())
                current_states.append(pre); returned_states.append(returned_state); fresh_post_states.append(post)
                bowls.append(bowl); returned_bowls.append(returned_bowl); actions.append(frame["action"])
                pre_errors.append(error_pre); post_errors.append(error_post); returned_errors.append(error_returned); bowl_errors.append(error_bowl)
                actual_tail.append({"done": done, "post_state": post, "post_bowl_pos": bowl, "action": frame["action"]})
                returned_tail.append({"done": bool(done_returned), "post_state": returned_state, "post_bowl_pos": returned_bowl, "action": frame["action"]})
                actual_tail = actual_tail[-10:]; returned_tail = returned_tail[-10:]
                if (index + 1) % 200 == 0:
                    stream.flush()
                    trial["executed_control_steps"] = index + 1
                    save_report()
                    print(name, index + 1, "EEF_error_mm", round(error_post * 1000, 3), flush=True)
        images(out, "final", obs)
        np.save(out / "final_sim_state.npy", env.get_sim_state(), allow_pickle=False)
        np.savez_compressed(out / "trajectory.npz", pre_state=np.asarray(current_states),
                            returned_post_state=np.asarray(returned_states), fresh_post_state=np.asarray(fresh_post_states),
                            fresh_bowl=np.asarray(bowls), returned_bowl=np.asarray(returned_bowls), actions=np.asarray(actions))
        trial["reference_errors_m"] = {}
        for key, errors in [("fresh_pre_eef", pre_errors), ("fresh_post_eef", post_errors),
                            ("returned_post_eef", returned_errors), ("fresh_post_bowl", bowl_errors)]:
            trial["reference_errors_m"][key] = {"mean": float(np.mean(errors)), "max": float(np.max(errors)),
                                              "p95": float(np.percentile(errors, 95)), "final": errors[-1]}
        trial.update(status="COMPLETE", executed_control_steps=len(actions), wall_duration_s=time.monotonic() - started,
                     final_on=actual_tail[-1]["done"], final_finger_gap_m=float(np.abs(state8(obs)[6:8]).sum()),
                     final_eef_bowl_distance_m=float(np.linalg.norm(obs["robot0_eef_pos"] - obs[f"{TARGET}_pos"])),
                     fresh_terminal_gates=gates(actual_tail), returned_terminal_gates=gates(returned_tail),
                     source_metadata_unchanged=digest(directory / "metadata.json") == trial["source_metadata_sha256"])
        print(json.dumps({"source": name, "fresh_gates": trial["fresh_terminal_gates"],
                          "mean_eef_error_mm": 1000*np.mean(pre_errors), "max_eef_error_mm": 1000*np.max(pre_errors)}, ensure_ascii=False), flush=True)
        save_report()
    finally:
        env.close()
    del frames
all_pass = all(t.get("fresh_terminal_gates", {}).get("pass", False) for t in report["trials"])
report["status"] = "ORIGINAL_SPEED_TERMINAL_GATES_PASS_PATH_ERRORS_REQUIRE_REVIEW" if all_pass else "ORIGINAL_SPEED_REPLAY_FAILED_NO_RETIMING"
report["next_step"] = "Review path errors before any retiming" if all_pass else "Diagnose restoration/physical divergence; do not retime or generate training data"
save_report()
print(report["status"], REPORT, flush=True)
