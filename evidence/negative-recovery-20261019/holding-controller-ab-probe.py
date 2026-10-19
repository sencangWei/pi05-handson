"""Four fixed-action holding steps; no model, server, GUI, or privileged action control."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".local/libero"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "openpi/third_party/libero"))

import numpy as np

from generate_tracking_teacher import restore_controller
from simulation_lab import LiberoTask, TARGET, state_digest

# Exact first two recorded control/seed-0/holding008 actions, frozen before archival.
ACTIONS = [
    [-.007582702323191359, -.032268387627492734, .0009148670581579088,
     -.001587409489817268, -.000016730719083091206, -.0056366156900791164, .9953360997488501],
    [.03849710951624252, -.023920421381762536, .024758945461571202,
     .006820714857349211, -.0015219271038469095, -.007744766292733185, 1.0128927758550645],
]


def observation(obs):
    return {"eef_xyz": np.asarray(obs["robot0_eef_pos"]).tolist(),
            "bowl_xyz": np.asarray(obs[f"{TARGET}_pos"]).tolist(),
            "finger_qpos": np.asarray(obs["robot0_gripper_qpos"]).tolist(),
            "gap_m": float(np.abs(np.diff(obs["robot0_gripper_qpos"]))[0])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-scene", type=Path,
                        default=ROOT / ".local/negative-recovery-20261019/holding-scenes/holding008")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    scene = json.loads((args.source_scene / "scene.json").read_text())
    saved = np.load(args.source_scene / "initial_state.npy", allow_pickle=False)
    assert state_digest(saved) == scene["initial_state_sha256"]
    task = LiberoTask()
    assert task.metadata == scene["task_metadata"] and scene["layout_id"] == 24
    branches = []
    for canonical in [False, True]:
        env, _ = task.create(scene["layout_id"])
        try:
            obs = env.regenerate_obs_from_state(saved)
            assert np.array_equal(env.get_sim_state(), saved)
            initialization = restore_controller(env) if canonical else None
            assert np.array_equal(env.get_sim_state(), saved)
            robot = env.env.robots[0]
            branch = {"canonical_restore": canonical, "initial_observation": observation(obs),
                      "controller_initialization": initialization,
                      "initial_gripper_accumulator": robot.gripper.current_action.tolist(), "steps": []}
            # Match evaluator's initial refresh. Thereafter each post-refresh is
            # the next pre-refresh, avoiding an extra observable-cache update.
            obs = env.regenerate_obs_from_state(env.get_sim_state())
            for index, action in enumerate(ACTIONS):
                before = observation(obs)
                cached, reward, done, _ = env.step(action)
                cached_values = observation(cached)
                fresh = env.regenerate_obs_from_state(env.get_sim_state())
                branch["steps"].append({"control_step": index + 1, "action": action,
                                        "pre": before, "post_cached": cached_values,
                                        "post_fresh": observation(fresh), "On": bool(done),
                                        "reward": float(reward),
                                        "gripper_accumulator": robot.gripper.current_action.tolist()})
                obs = fresh
            branches.append(branch)
        finally:
            env.close()
    result = {"status": "COMPLETE", "source_scene": str(args.source_scene.resolve()),
              "initial_state_sha256": state_digest(saved), "diagnostic_holding_steps": 4,
              "setup": "Each fresh branch uses the same task.create home setup (10 dummy steps) as the evaluator before restoring the saved state.",
              "observation_timing": "Initial evaluator refresh, then exactly one post refresh reused for next pre; cached post retained before refresh.",
              "branches": branches, "limits": ["Fixed recorded actions; no policy inference or GT-derived action.",
              "Canonical helper is not exact original hidden-controller-history restoration.",
              "This isolates the evaluator initialization defect and does not explain original GUI slow behavior."],
              "source_sha256": {name: hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest()
                                for name in ["replay_saved_scene.py", "generate_tracking_teacher.py", "simulation_lab.py"]},
              "probe_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "branches": [
        {"canonical": b["canonical_restore"], "initial_accumulator": b["initial_gripper_accumulator"],
         "last_gap_m": b["steps"][-1]["post_fresh"]["gap_m"],
         "last_bowl_z_m": b["steps"][-1]["post_fresh"]["bowl_xyz"][2]} for b in branches]}), flush=True)


if __name__ == "__main__":
    main()
