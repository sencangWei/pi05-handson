"""Read-only paired validation of teacher retraining and its50-step release followup."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLLOWUP = 50


def sha(data):
    return hashlib.sha256(data).hexdigest()


def first(mask):
    indices = np.flatnonzero(mask)
    return int(indices[0]) + 1 if len(indices) else None


def stats(values):
    values = np.asarray(values, dtype=float)
    return {"n": len(values), "mean": float(values.mean()) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None}


def analyze(rows, initial_bowl_z, budget=800):
    """Events count executed actions; CSV step0 is event/control-step1."""
    n = len(rows)
    if not 0 < n <= budget or [int(row["step"]) for row in rows] != list(range(n)):
        raise ValueError("Action count or contiguous zero-based CSV steps differ")

    def array(prefix, count):
        data = np.array([[float(row[f"{prefix}_{i}"]) for i in range(count)] for row in rows])
        if not np.isfinite(data).all():
            raise ValueError(f"Nonfinite {prefix} values")
        return data

    state, action = array("state", 8), array("action", 7)
    eef, bowl, finger = array("post_eef", 3), array("post_bowl", 3), array("post_finger", 2)
    if any(row["done"].lower() not in {"true", "false", "1", "0"} for row in rows):
        raise ValueError("Invalid done field")
    on = np.array([row["done"].lower() in {"true", "1"} for row in rows])
    opened, closed = action[:, 6] < 0, action[:, 6] > 0
    gap = np.abs(finger).sum(axis=1)
    distance = np.linalg.norm(eef - bowl, axis=1)
    gate = on & opened & (gap >= .06) & (distance >= .10)
    first_gate = None
    if n >= 10:
        window = first(np.convolve(gate.astype(int), np.ones(10, dtype=int), "valid") == 10)
        first_gate = window + 9 if window is not None else None
    carry = closed & (bowl[:, 2] > initial_bowl_z + .03) & ~on
    carry_adjacent = carry[:-1] & carry[1:]
    displacement = np.linalg.norm(np.diff(state[:, :3], axis=0), axis=1) * 1000
    follow_count = min(FOLLOWUP, n - first_gate) if first_gate is not None else 0
    complete = first_gate is not None and follow_count == FOLLOWUP
    follow = slice(first_gate, first_gate + follow_count) if first_gate is not None else slice(0, 0)
    terms = {"On": on, "open_command": opened, "actual_gap_ge_6cm": gap >= .06,
             "retreat_ge_10cm": distance >= .10, "full_gate": gate}
    maintained = {}
    for name, condition in terms.items():
        values = condition[follow]
        violation = first(~values) if follow_count else None
        maintained[name] = {
            "observed_fraction": float(values.mean()) if follow_count else None,
            "all_observed_steps": bool(values.all()) if follow_count else None,
            "first_violation_control_step": first_gate + violation if violation is not None else None,
        }
    if follow_count:
        drift = bowl[follow] - bowl[first_gate - 1]
        drift_3d = np.linalg.norm(drift, axis=1)
        drift_xy = np.linalg.norm(drift[:, :2], axis=1)
        reclosed = closed[follow]
        reclose_first = first(reclosed)
    else:
        drift_3d = drift_xy = np.array([])
        reclosed, reclose_first = np.array([], dtype=bool), None
    on_step = first(on)
    after_on = slice(on_step - 1, n) if on_step is not None else slice(0, 0)
    return {
        "control_steps": n, "any_On": bool(on.any()), "final_On": bool(on[-1]),
        "first_close_control_step": first(closed),
        "first_lift_over_2cm_control_step": first(bowl[:, 2] > initial_bowl_z + .02),
        "first_On_control_step": first(on), "first_stable_release_control_step": first_gate,
        "release_after_first_On": {
            "steps_to_first_stable_release": first_gate - on_step if first_gate is not None else None,
            "observed_actions_from_first_On_through_stop": n - on_step + 1 if on_step is not None else 0,
            "close_command_fraction": float(closed[after_on].mean()) if on_step is not None else None,
            "actual_open_gap_fraction": float((gap[after_on] >= .06).mean()) if on_step is not None else None,
        },
        "time_to_stable_release": {"observed_control_steps": first_gate,
                                   "right_censored_at_control_steps": n if first_gate is None and n == budget else None,
                                   "truncated_before_budget": first_gate is None and n < budget},
        "carry_proxy": {"definition": "closed command+post target z>initial+.03m+not On; not verified grasp",
                        "first_control_step": first(carry), "frames": int(carry.sum()),
                        "action_xyz_norm": stats(np.linalg.norm(action[carry, :3], axis=1)),
                        "fresh_adjacent_eef_displacement_mm_per_control_step": stats(displacement[carry_adjacent])},
        "all_motion_proxy": {"action_xyz_norm": stats(np.linalg.norm(action[:, :3], axis=1)),
                             "fresh_adjacent_eef_displacement_mm_per_control_step": stats(displacement)},
        "followup": {
            "requested_steps": FOLLOWUP, "observed_steps": follow_count, "quantity_complete": complete,
            "right_censored": first_gate is not None and not complete and n == budget,
            "not_reached": first_gate is None, "maintained": maintained,
            "strict_50_step_gate_maintained": bool(gate[follow].all()) if complete else None,
            "observed_instability": bool((~gate[follow]).any()) if follow_count else None,
            "reclose_command_steps": int(reclosed.sum()),
            "first_reclose_control_step": first_gate + reclose_first if reclose_first is not None else None,
            "bowl_drift_anchor_control_step": first_gate,
            "maximum_bowl_drift_3d_m": float(drift_3d.max()) if follow_count else None,
            "maximum_bowl_drift_xy_m": float(drift_xy.max()) if follow_count else None,
            "final_bowl_drift_3d_m": float(drift_3d[-1]) if follow_count else None,
        },
    }


def read_run(directory, plan):
    result_bytes = (directory / "result.json").read_bytes()
    result = json.loads(result_bytes)
    metadata = result.get("model_metadata") or {}
    stage = metadata.get("model_stage", "")
    experiment = stage.rsplit("/", 1)[0]
    arm = next((arm for arm, name in plan["arms"].items() if experiment == name), None)
    case = Path(result.get("saved_scene_replay", "")).name
    seed = metadata.get("diagnostic_rng_reset_per_connection")
    spec = plan["followup_evaluation"]
    if arm is None or case not in spec["physical_cases"] or type(seed) is not int or seed not in spec["policy_seeds"]:
        raise ValueError(f"Unplanned arm/case/seed in {directory}")
    identity = {"arm": arm, "case": case, "policy_seed": seed, "directory": str(directory)}
    if result.get("exception"):
        return {**identity, "status": "ERROR", "error": result["exception"],
                "result_sha256": sha(result_bytes)}
    if "video_frames" not in result:
        return None  # TrialRecord publishes result once before video encoding is complete.
    csv_bytes = (directory / "actions.csv").read_bytes()
    metrics = analyze(list(csv.DictReader(io.StringIO(csv_bytes.decode()))),
                      result["bowl_edit"]["after_xyz"][2], spec["max_control_steps"])
    initial = np.load(directory / "initial_state.npy", allow_pickle=False)
    if initial.ndim != 1 or not np.isfinite(initial).all():
        raise ValueError("Invalid saved initial state array")
    initial_sha = sha(initial.tobytes())
    source = Path(result["saved_scene_replay"])
    source_bytes = (source / "scene.json").read_bytes()
    source_scene = json.loads(source_bytes)
    c, n = metrics["first_stable_release_control_step"], metrics["control_steps"]
    follow_count = n - c if c is not None else 0
    expected_stop = ("stable_release_retreat_followup" if c is not None and follow_count == FOLLOWUP
                     else "step_limit_after_release" if c is not None else "step_limit")
    validations = {
        "initial_state_identity": initial_sha == result["initial_state_sha256"] == source_scene["initial_state_sha256"],
        "source_scene_sha": sha(source_bytes) == result["source_scene_sha256"],
        "checkpoint_stage": stage == f"{plan['arms'][arm]}/{plan['new_updates_per_arm'] - 1}",
        "checkpoint_path_stage": Path(metadata.get("checkpoint", "")).parts[-2:]
        == (plan["arms"][arm], str(plan["new_updates_per_arm"] - 1)),
        "normalization": metadata.get("norm_stats_sha256") == plan["normalization_sha256"],
        "model_contract": metadata.get("action_horizon") == plan["action_horizon"]
        and metadata.get("state_input") is plan["discrete_state_input"],
        "layout_sim_seed_guard": result["layout_id"] == spec["layout_id"]
        and result["seed"] == spec["sim_seed"] and result.get("guard_checked_no_deep_overlap") is True,
        "replan_budget_followup": result["max_steps"] == spec["max_control_steps"]
        and result["replan_steps"] == spec["replan_steps"] and result.get("post_release_steps_requested") == FOLLOWUP,
        "steps_and_On": result["control_steps"] == n and result["success"] == metrics["any_On"]
        and result["final_benchmark_success"] == metrics["final_On"]
        and result["first_benchmark_success_control_steps"] == metrics["first_On_control_step"],
        "stable_event": result.get("first_ten_step_release_retreat_control_steps") == c
        and result.get("ten_step_release_retreat_verified") == (c is not None),
        "followup_quantity_only": result.get("post_release_steps_executed") == follow_count
        and result.get("post_release_followup_complete") == metrics["followup"]["quantity_complete"]
        and follow_count <= FOLLOWUP,
        "stop_and_budget": result["stop_reason"] == expected_stop and (follow_count == FOLLOWUP or n == spec["max_control_steps"]),
        "video_frame_contract": result["video_frames"] == n + 1 and result.get("video_fps") == 10,
    }
    wall = result.get("wall_duration_s")
    metrics["system_timing"] = {"wall_duration_s": wall,
                                "control_steps_per_wall_second": n / wall if isinstance(wall, (int, float)) and wall > 0 else None,
                                "meaning": "Inference/render/record system throughput; not physical robot velocity."}
    return {**identity, "status": "COMPLETE" if all(validations.values()) else "INVALID",
            "validations": validations, "initial_state_sha256": initial_sha,
            "source_scene_sha256": sha(source_bytes), "model_metadata": metadata,
            "source_sha256": {"actions.csv": sha(csv_bytes), "result.json": sha(result_bytes)},
            "metrics": metrics}


def paired_report(records, plan):
    lookup = {(r["arm"], r["case"], r["policy_seed"]): r for r in records}
    pairs = []
    for case in plan["followup_evaluation"]["physical_cases"]:
        for seed in plan["followup_evaluation"]["policy_seeds"]:
            key = {"case": case, "policy_seed": seed}
            a, b = (lookup.get((arm, case, seed)) for arm in ["control", "candidate"])
            if any(r is None or r["status"] != "COMPLETE" for r in [a, b]):
                pairs.append({**key, "status": "PENDING_OR_INVALID"})
                continue
            match = a["initial_state_sha256"] == b["initial_state_sha256"] and a["source_scene_sha256"] == b["source_scene_sha256"]
            if not match:
                pairs.append({**key, "status": "INVALID_INITIAL_PAIR"})
                continue
            am, bm = a["metrics"], b["metrics"]
            ac, bc = am["first_stable_release_control_step"], bm["first_stable_release_control_step"]
            full = am["followup"]["quantity_complete"] and bm["followup"]["quantity_complete"]
            pairs.append({**key, "status": "PAIRED", "initial_state_sha256": a["initial_state_sha256"],
                          "stable_release_observed": {"control": ac is not None, "candidate": bc is not None},
                          "candidate_minus_control_steps_to_stable_release": bc - ac if ac is not None and bc is not None else None,
                          "both_followups_quantity_complete": full,
                          "strict_50_step_gate_maintained": {"control": am["followup"]["strict_50_step_gate_maintained"],
                                                              "candidate": bm["followup"]["strict_50_step_gate_maintained"]},
                          "censoring": {"control": am["time_to_stable_release"], "candidate": bm["time_to_stable_release"]},
                          "followup": {"control": am["followup"], "candidate": bm["followup"]},
                          "carry_proxy": {"control": am["carry_proxy"], "candidate": bm["carry_proxy"]}})
    return pairs


def summarize(plan_path, runs):
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    if set(plan["arms"]) != {"control", "candidate"}:
        raise ValueError("Expected the predeclared control/candidate arms")
    spec = plan["followup_evaluation"]
    expected = {(arm, case, seed) for arm in plan["arms"] for case in spec["physical_cases"] for seed in spec["policy_seeds"]}
    if len(expected) != 24 or spec["max_control_steps"] != 800 or spec["replan_steps"] != 1:
        raise ValueError("Expected the fixed two-arm four-case three-seed800-step validation")
    records, errors = [], []
    for path in sorted(runs.rglob("result.json")):
        try:
            record = read_run(path.parent, plan)
            if record is not None:
                records.append(record)
        except (OSError, ValueError, KeyError) as error:
            errors.append({"directory": str(path.parent), "error": str(error)})
    keys = [(r["arm"], r["case"], r["policy_seed"]) for r in records]
    duplicates = {key for key in keys if keys.count(key) > 1}
    for record in records:
        key = record["arm"], record["case"], record["policy_seed"]
        if key in duplicates:
            record.update(status="INVALID", error="Duplicate planned arm/case/seed")
    scene_consistency = {}
    for case in spec["physical_cases"]:
        hashes = {(r["initial_state_sha256"], r["source_scene_sha256"]) for r in records
                  if r["case"] == case and r["status"] == "COMPLETE"}
        scene_consistency[case] = len(hashes) == 1 if hashes else None
    pairs = paired_report(records, plan)
    valid = sum(r["status"] == "COMPLETE" for r in records)
    invalid = errors or any(r["status"] != "COMPLETE" for r in records) or any(v is False for v in scene_consistency.values())
    return {"status": "INVALID" if invalid else "COMPLETE" if valid == len(expected) else "RUNNING",
            "experiment_plan_sha256": sha(plan_bytes), "plan": plan, "planned_runs": len(expected),
            "completed_valid_runs": valid, "pending_slots": [list(key) for key in sorted(expected - set(keys))],
            "input_errors": errors, "same_case_initial_identity_over_available_runs": scene_consistency,
            "runs": records, "paired_case_seed_results": pairs,
            "limits": ["These are predeclared validation cases, not unseen tests or an unbiased generalization success rate.",
                       "Policy seeds0/1/2 vary sampling; all use simulation seed42/layout24 and the same saved state per case.",
                       "No stable release by800 is right-censored. Missing/invalid/exception runs are not silently counted as task failures.",
                       "A50-step complete followup is a quantity check. Only independently recomputed per-step terms describe maintained stability.",
                       "Incomplete followups have unknown full50-step outcomes even if every observed step passed; any observed violation is still reported.",
                       "Event-time deltas are computed only if both paired releases were observed, never substituting800 for censored times.",
                       "Carry and lift are diagnostic proxies, not verified grasps. Motion is fresh-state mm per control step; wall rate is system throughput.",
                       "No model/policy/physics/training is loaded by this analyzer."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "evidence/retrain-20261019/experiment-plan.json")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.plan.resolve() or args.output.resolve().is_relative_to(args.runs.resolve()):
        parser.error("Summary output must be separate from plan and raw run directories")
    report = summarize(args.plan.resolve(), args.runs.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ["status", "planned_runs", "completed_valid_runs"]}))


if __name__ == "__main__":
    main()
