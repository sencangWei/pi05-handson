"""Read-only saved-trial audit; isolated rendering, zero policy calls or motion."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from robosuite.utils.camera_utils import get_camera_segmentation, get_camera_transform_matrix  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from simulation_lab import LiberoTask, TARGET, state_digest  # noqa: E402

SESSION = ROOT / "evaluations/interactive_lab_responsive_1792399080114971854"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_rows(rows, key):
    values = [float(row[key]) for row in rows if row.get(key)]
    return float(np.mean(values)) if values else None


def inspect_state(env, saved, output, tag):
    obs = env.regenerate_obs_from_state(saved)
    assert np.array_equal(saved, env.get_sim_state())
    sim = env.sim
    target_ids = [i for i in range(sim.model.ngeom)
                  if TARGET in str(sim.model.geom_id2name(i))]
    assert target_ids
    cameras = {}
    canvas = Image.new("RGB", (512, 280), "white")
    draw = ImageDraw.Draw(canvas)
    for index, camera in enumerate(["agentview", "robot0_eye_in_hand"]):
        # segmentation is a second independent render; neither render advances time.
        rgb = np.ascontiguousarray(obs[camera + "_image"][::-1, ::-1])
        seg = get_camera_segmentation(sim, camera, 256, 256)[:, ::-1]
        mask = (seg[:, :, 0] == 5) & np.isin(seg[:, :, 1], target_ids)
        yy, xx = np.where(mask)
        matrix = get_camera_transform_matrix(sim, camera, 256, 256)
        projected = matrix @ np.r_[obs[TARGET + "_pos"], 1.]
        u, v = projected[:2] / projected[2]
        cameras[camera] = {
            "visible_target_pixels": int(mask.sum()),
            "target_pixel_bbox_xyxy": [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())] if len(xx) else None,
            "target_center_depth_m": float(projected[2]),
            "target_center_input_xy": [float(255-u), float(v)],
            "center_inside_image": bool(projected[2] > 0 and 0 <= u < 256 and 0 <= v < 256),
            "limits": "Visible pixels include occlusion. Projected body origin is not the object silhouette.",
        }
        Image.fromarray(rgb).save(output / f"audit-{tag}-{camera}.png")
        highlight = rgb.copy()
        highlight[mask] = (highlight[mask].astype(float) * .35 + np.array([0, 255, 0]) * .65).astype(np.uint8)
        canvas.paste(Image.fromarray(highlight), (index*256, 24))
        draw.text((index*256+3, 4), f"{camera}: {mask.sum()} target pixels", fill="black")
    canvas.save(output / f"{tag}-target-visibility.png")
    contacts = []
    for c in sim.data.contact[:sim.data.ncon]:
        names = [sim.model.geom_id2name(int(c.geom1)), sim.model.geom_id2name(int(c.geom2))]
        if any(TARGET in str(name) or "gripper" in str(name) for name in names):
            contacts.append({"geom_names": names, "distance_m": float(c.dist)})
    robot = env.env.robots[0]
    q = sim.data.qpos[robot._ref_joint_pos_indexes]
    bounds = sim.model.jnt_range[robot._ref_joint_indexes]
    margin = np.minimum(q-bounds[:, 0], bounds[:, 1]-q)
    assert np.array_equal(saved, env.get_sim_state()), "Rendering altered integration state"
    return {"cameras": cameras, "contacts": contacts,
            "minimum_joint_limit_margin_rad": float(margin.min()),
            "state_unchanged": True, "eef_xyz": obs["robot0_eef_pos"].tolist(),
            "bowl_xyz": obs[TARGET + "_pos"].tolist(),
            "actual_gripper_gap_m": float(np.abs(obs["robot0_gripper_qpos"]).sum()),
            "on": bool(env.check_success())}


def audit(destination):
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    task = LiberoTask()
    report = {"status": "RUNNING", "policy_calls": 0, "physics_steps": 0,
              "training_data_generated": False, "task_metadata": task.metadata,
              "source_session": str(SESSION), "trials": [],
              "limits": ["Observational GUI trials, not independent seeded comparisons.",
                         "Fresh pre-state differences measure motion; cached post-state measures are labeled separately.",
                         "Initial/final contacts and visibility do not reconstruct intermediate history.",
                         "No motion probe: this audit cannot prove mechanical mobility or identify training root cause."]}
    for number in [4, 5, 6, 7]:
        source = SESSION / f"scene-{number:03d}_init-024"
        result = json.loads((source / "result.json").read_text())
        assert result["task_metadata"] == task.metadata
        assert result["control_loop"] == "responsive_v1"
        target = destination / "source-trials" / source.name
        shutil.copytree(source, target)
        rows = list(csv.DictReader((target / "actions.csv").open()))
        timing = list(csv.DictReader((target / "timing.csv").open()))
        states = np.array([[float(row[f"state_{j}"]) for j in range(8)] for row in rows])
        actions = np.array([[float(row[f"action_{j}"]) for j in range(7)] for row in rows])
        bowls = np.array([[float(row[f"post_bowl_{j}"]) for j in range(3)] for row in rows])
        stages = []
        for start in range(0, len(rows), 100):
            stop = min(start+100, len(rows))
            fresh_moves = np.linalg.norm(np.diff(states[start:stop, :3], axis=0), axis=1)*1000
            stages.append({"control_steps": [start+1, stop],
                           "xyz_action_norm_median": float(np.median(np.linalg.norm(actions[start:stop, :3], axis=1))),
                           "fresh_pre_state_move_median_mm": float(np.median(fresh_moves)),
                           "closed_command_fraction": float(np.mean(actions[start:stop, 6] > 0)),
                           "cached_bowl_max_lift_mm": float((bowls[start:stop, 2].max()-bowls[0, 2])*1000),
                           "mean_action_interval_s": mean_rows(timing[start:stop], "action_interval_s"),
                           "mean_client_inference_s": mean_rows(timing[start:stop], "client_inference_s")})
        entry = {"scene": number, "source_sha256": {p.name: sha(p) for p in source.iterdir() if p.is_file()},
                 "bowl_edit": result["bowl_edit"], "model_metadata": result["model_metadata"],
                 "control_steps": result["control_steps"], "stop_reason": result["stop_reason"],
                 "release_and_retreat_success": result["release_and_retreat_success"],
                 "final_on": result["final_benchmark_success"], "phases": stages,
                 "timing_summary": result["timing_summary"], "states": {}}
        env = OffScreenRenderEnv(bddl_file_name=str(task.bddl), camera_heights=256, camera_widths=256, ignore_done=True)
        try:
            env.seed(42)
            env.reset()
            for tag in ["initial", "final"]:
                saved = np.load(target / f"{tag}_state.npy", allow_pickle=False)
                assert state_digest(saved) == result[f"{tag}_state_sha256"]
                entry["states"][tag] = inspect_state(env, saved, target, tag)
        finally:
            env.close()
        # All raw source artifacts are unchanged; new pictures have distinct names.
        assert all(sha(source/name) == value == sha(target/name) for name, value in entry["source_sha256"].items())
        report["trials"].append(entry)
        (destination / "audit.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps({"scene": number, "steps": entry["control_steps"], "states": entry["states"]}), flush=True)
    report["status"] = "COMPLETE_DIAGNOSIS_ONLY"
    report["audit_script_sha256"] = sha(Path(__file__))
    (destination / "audit.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Pass a new output directory")
    audit(Path(sys.argv[1]).resolve())
