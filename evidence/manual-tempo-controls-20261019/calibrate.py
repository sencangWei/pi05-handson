"""Bounded offscreen manual-motion calibration; no policy or training labels."""
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from generate_tracking_teacher import restore_controller  # noqa: E402
from simulation_lab import LiberoTask, TARGET, state_digest  # noqa: E402

OUT = Path(__file__).with_name("calibration.json")


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    task = LiberoTask()
    poses = {
        "free": ROOT / "evidence/latest-approach-stall-20261019/source-trials/scene-005_init-024/initial_state.npy",
        "holding008": ROOT / "evidence/negative-stall-20261019/source-trials/scene-008_init-024/final_state.npy",
        "holding009": ROOT / "evidence/negative-stall-20261019/source-trials/scene-009_init-024/final_state.npy"}
    report = {"status": "RUNNING", "policy_calls": 0, "training_data_generated": False,
              "control_steps_per_branch": 10, "scales": [.1, .3, .6], "branches": [],
              "limits": "External diagnostic controls, not model recovery or approved demonstrations. Saved integration states restored exactly; controller cache/goals/gripper reconstructed, not bit-exact hidden history. Short motions do not certify all workspace contacts."}
    for name, path in poses.items():
        state = np.load(path, allow_pickle=False)
        env = OffScreenRenderEnv(bddl_file_name=str(task.bddl), camera_heights=256, camera_widths=256, ignore_done=True)
        try:
            env.seed(42)
            env.reset()
            directions = [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1)]
            if name == "free":
                directions.append((2, -1))
            for axis, sign in directions:
                for scale in report["scales"]:
                    obs = env.regenerate_obs_from_state(state)
                    assert np.array_equal(env.get_sim_state(), state)
                    controller = restore_controller(env)
                    start_eef, start_bowl = obs["robot0_eef_pos"].copy(), obs[TARGET+"_pos"].copy()
                    relative = start_bowl-start_eef
                    action = np.zeros(7)
                    action[axis], action[6] = sign*scale, -1 if name == "free" else 1
                    samples = []
                    for _ in range(10):
                        env.step(action.tolist())
                        obs = env.regenerate_obs_from_state(env.get_sim_state())
                        eef, bowl = obs["robot0_eef_pos"], obs[TARGET+"_pos"]
                        samples.append({"eef_xyz": eef.tolist(), "bowl_xyz": bowl.tolist(),
                                        "relative_drift_mm": float(np.linalg.norm(bowl-eef-relative)*1000),
                                        "bowl_height_change_mm": float((bowl[2]-start_bowl[2])*1000)})
                    delta = np.array(samples[-1]["eef_xyz"])-start_eef
                    entry = {"pose": name, "axis": axis, "sign": sign, "scale": scale,
                             "source": str(path.relative_to(ROOT)), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "initial_state_sha256": state_digest(state), "controller": controller,
                             "control_dt": float(env.env.control_timestep), "action": action.tolist(),
                             "signed_axis_displacement_mm": float(delta[axis]*sign*1000),
                             "net_displacement_mm": float(np.linalg.norm(delta)*1000),
                             "maximum_relative_drift_mm": max(s["relative_drift_mm"] for s in samples),
                             "samples": samples}
                    report["branches"].append(entry)
                    OUT.write_text(json.dumps(report, indent=2)+"\n")
                    print(json.dumps({k: entry[k] for k in ["pose", "axis", "sign", "scale", "net_displacement_mm", "maximum_relative_drift_mm"]}), flush=True)
        finally:
            env.close()
    report["status"] = "COMPLETE"
    report["total_control_steps"] = len(report["branches"])*10
    report["source_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    OUT.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
