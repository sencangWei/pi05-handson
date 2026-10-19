"""Validate all paired recordings and add fresh-pre-state movement diagnostics."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_negative_recovery import summarize  # noqa: E402

EVIDENCE = Path(__file__).resolve().parent
RUNS = ROOT / "evaluations/precorrection-comparison-20261019"


def main():
    result = summarize(EVIDENCE / "plan.json", RUNS)
    for run in result["runs"]:
        if run["status"] != "COMPLETE":
            continue
        directory = Path(run["directory"])
        rows = list(csv.DictReader((directory / "actions.csv").open()))
        states = np.array([[float(row[f"state_{j}"]) for j in range(8)] for row in rows])
        actions = np.array([[float(row[f"action_{j}"]) for j in range(7)] for row in rows])
        bowls = np.array([[float(row[f"post_bowl_{j}"]) for j in range(3)] for row in rows])
        first_lift = np.flatnonzero(bowls[:, 2] > bowls[0, 2]+.02)
        intervals = []
        for start in range(0, len(rows), 100):
            stop = min(len(rows), start+100)
            movement = np.linalg.norm(np.diff(states[start:stop, :3], axis=0), axis=1)*1000
            intervals.append({"control_steps": [start+1, stop],
                              "fresh_pre_state_move_median_mm": float(np.median(movement)) if len(movement) else None,
                              "action_xyz_norm_median": float(np.median(np.linalg.norm(actions[start:stop, :3], axis=1))),
                              "closed_command_fraction": float(np.mean(actions[start:stop, 6] > 0))})
        run["fresh_movement"] = {
            "definition": "Adjacent fresh pre-state XYZ, mm/control-step; 100-step bins, no wall-time conversion. Bowl lift uses cached post-state and is not proof of stable grasp.",
            "first_cached_bowl_lift_2cm_step": int(first_lift[0])+1 if len(first_lift) else None,
            "bins": intervals}
        video = directory / "video.mp4"
        run["video_sha256"] = hashlib.sha256(video.read_bytes()).hexdigest()
    (EVIDENCE / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"status": result["status"], "valid": result["completed_valid_runs"],
                      "planned": result["planned_runs"], "arms": result["arms"], "pairs": result["pairs"]}, indent=2))


if __name__ == "__main__":
    main()
