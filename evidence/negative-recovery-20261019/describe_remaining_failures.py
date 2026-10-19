"""Describe observed movement in complete failed trials; do not infer grasp or causes."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "evidence/negative-recovery-20261019"


def main():
    summary_path = EVIDENCE / "evaluation-summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "COMPLETE"
    output = []
    for run in summary["runs"]:
        if run["arm"] != "candidate" or run["metrics"]["followup"]["strict_success"] is True:
            continue
        path = Path(run["directory"]) / "actions.csv"
        rows = list(csv.DictReader(path.open()))
        def values(prefix, width):
            return np.array([[float(row[f"{prefix}_{i}"]) for i in range(width)] for row in rows])
        bowl, eef, actions = values("post_bowl", 3), values("post_eef", 3), values("action", 7)
        output.append({
            "case": run["case"], "role": run["role"], "policy_seed": run["policy_seed"],
            "control_steps": len(rows), "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "any_On": run["metrics"]["any_On"],
            "maximum_target_bowl_rise_from_first_post_cm": float((bowl[:, 2].max() - bowl[0, 2]) * 100),
            "final_target_bowl_minus_first_post_cm": ((bowl[-1] - bowl[0]) * 100).tolist(),
            "final_eef_target_distance_cm": float(np.linalg.norm(eef[-1] - bowl[-1]) * 100),
            "last100_post_to_post_eef_median_displacement_mm": float(np.median(np.linalg.norm(np.diff(eef[-100:], axis=0), axis=1)) * 1000),
            "last100_requested_action_xyz_median_norm": float(np.median(np.linalg.norm(actions[-100:, :3], axis=1))),
            "summary_diagnostic_proxies": run["metrics"]["diagnostic_proxies"],
        })
    report = {"status": "COMPLETE", "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
              "trials": output,
              "limits": ["These are observed positions/actions, not verified grasp/contact labels or causal diagnoses.",
                         "Rises are measured relative to the first post observation; held-state cases already start above the table.",
                         "Action XYZ is a normalized controller request, not meters or a joint displacement.",
                         "Small but nonzero observed movement is distinct from an early step-budget termination; causes require controlled follow-up studies."]}
    path = EVIDENCE / "remaining-failure-diagnostics.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "candidate_non_strict_trials": len(output)}))


if __name__ == "__main__":
    main()
