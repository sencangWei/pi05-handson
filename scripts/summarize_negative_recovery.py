"""Recompute plan-declared paired recovery outcomes from saved CSVs, without a policy."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLLOWUP = 50


def digest(data):
    return hashlib.sha256(data).hexdigest()


def first(mask):
    found = np.flatnonzero(mask)
    return int(found[0]) + 1 if len(found) else None


def stats(values):
    return {"n": len(values), "mean": float(np.mean(values)) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None}


def analyze(rows, budget):
    """CSV row0 records control step1. Followup starts after the tenth gate action."""
    n = len(rows)
    if not 0 < n <= budget or [int(row["step"]) for row in rows] != list(range(n)):
        raise ValueError("Expected contiguous zero-based CSV steps within the case budget")

    def array(prefix, width):
        values = np.array([[float(row[f"{prefix}_{j}"]) for j in range(width)] for row in rows])
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite {prefix}")
        return values

    def boolean(value):
        if not isinstance(value, str) or value.lower() not in {"true", "false", "0", "1"}:
            raise ValueError("CSV done must be True/False or 1/0")
        return value.lower() in {"true", "1"}

    action, eef, bowl = array("action", 7), array("post_eef", 3), array("post_bowl", 3)
    finger = array("post_finger", 2)
    on = np.array([boolean(row["done"]) for row in rows])
    gap = np.abs(finger[:, 0] - finger[:, 1])
    gate = on & (action[:, 6] < 0) & (gap >= .06) & (np.linalg.norm(eef - bowl, axis=1) >= .10)
    event = None
    if n >= 10:
        start = first(np.convolve(gate.astype(int), np.ones(10, dtype=int), "valid") == 10)
        event = start + 9 if start is not None else None
    observed = min(FOLLOWUP, n - event) if event is not None else 0
    follow = slice(event, event + observed) if event is not None else slice(0, 0)
    complete = observed == FOLLOWUP
    drift = np.linalg.norm(bowl[follow] - bowl[event - 1], axis=1) if observed else np.array([])
    maximum_drift = float(drift.max()) if observed else None
    strict = bool(gate[follow].all() and maximum_drift <= .005) if complete else None
    low = (action[:, 6] > 0) & ~on & (np.linalg.norm(action[:, :3], axis=1) < .08)
    edges = np.flatnonzero(np.diff(np.r_[False, low, False]))
    ranges = list(zip(edges[::2], edges[1::2]))
    longest = max(ranges, key=lambda bounds: bounds[1] - bounds[0], default=(0, 0))
    low_pairs = low[:-1] & low[1:]
    displacement = np.linalg.norm(np.diff(eef, axis=0), axis=1) * 1000
    return {
        "control_steps": n, "any_On": bool(on.any()), "final_On": bool(on[-1]),
        "first_On_control_step": first(on), "first_ten_step_release_control_step": event,
        "followup": {"requested_steps": FOLLOWUP, "observed_steps": observed,
                     "quantity_complete": complete,
                     "all_observed_gate_steps_pass": bool(gate[follow].all()) if observed else None,
                     "first_gate_violation_control_step": event + first(~gate[follow])
                     if observed and first(~gate[follow]) is not None else None,
                     "maximum_bowl_drift_m": maximum_drift, "strict_success": strict},
        "completion_control_steps": event + FOLLOWUP if strict else None,
        "right_censored_without_release_at_budget": event is None and n == budget,
        "diagnostic_proxies": {
            "definition": "closed command, not On, xyz action norm<.08; not verified grasp or a cause",
            "action_xyz_norm": stats(np.linalg.norm(action[:, :3], axis=1)),
            "post_to_post_eef_displacement_mm": stats(displacement),
            "low_action_closed_not_On_frames": int(low.sum()),
            "longest_low_action_closed_not_On": {
                "steps": int(longest[1] - longest[0]),
                "start_control_step": int(longest[0]) + 1 if longest[1] > longest[0] else None,
                "end_control_step": int(longest[1]) if longest[1] > longest[0] else None},
            "post_to_post_eef_displacement_within_low_run_mm": stats(displacement[low_pairs]),
        },
    }


def stage_for(value, plan):
    if isinstance(value, dict):
        value = value.get("checkpoint", value.get("experiment"))
    if not isinstance(value, str) or not value:
        raise ValueError("Each arm needs an experiment name or checkpoint path")
    parts = Path(value).parts
    step = plan.get("checkpoint_step", plan.get("new_updates_per_arm", 2000) - 1)
    return "/".join(parts[-2:]) if parts[-1].isdigit() else f"{value}/{step}"


def read_run(directory, plan, case, arm, seed, project_root):
    identity = {"arm": arm, "case": case["case"], "role": case["role"],
                "policy_seed": seed, "directory": str(directory)}
    required = ["actions.csv", "result.json", "scene.json", "initial_state.npy"]
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        return {**identity, "status": "INCOMPLETE", "missing_files": missing}
    try:
        raw = {name: (directory / name).read_bytes() for name in required}
        result, scene = json.loads(raw["result.json"]), json.loads(raw["scene.json"])
        if result.get("exception"):
            return {**identity, "status": "INVALID", "error": str(result["exception"])}
        if "video_frames" not in result:
            return {**identity, "status": "INCOMPLETE", "error": "Result/video publication unfinished"}
        rows = list(csv.DictReader(io.StringIO(raw["actions.csv"].decode())))
        metrics = analyze(rows, case["max_steps"])
        source = Path(case["source_scene"])
        if not source.is_absolute():
            source = project_root / source
        source = source.resolve()
        source_json = (source / "scene.json").read_bytes()
        saved = np.load(directory / "initial_state.npy", allow_pickle=False)
        expected_bytes = (source / "initial_state.npy").read_bytes()
        expected = np.load(io.BytesIO(expected_bytes), allow_pickle=False)
        initial_hash = digest(saved.tobytes())
        metadata = result.get("model_metadata") or {}
        n, event = metrics["control_steps"], metrics["first_ten_step_release_control_step"]
        observed = metrics["followup"]["observed_steps"]

        def equal_bool(key, value):
            return type(result.get(key)) is bool and result[key] is value

        def equal_int(key, value):
            return type(result.get(key)) is int and result[key] == value

        def equal_event(key, value):
            return result.get(key) is None if value is None else equal_int(key, value)

        checks = {
            "initial_state": saved.ndim == 1 and np.isfinite(saved).all() and np.array_equal(saved, expected)
            and result.get("initial_state_sha256") == initial_hash == scene.get("initial_state_sha256"),
            "source_scene": result.get("saved_scene_replay") == str(source)
            and result.get("source_scene_sha256") == digest(source_json),
            "plan_scene_pin": "scene_sha256" not in case or case["scene_sha256"] == digest(source_json),
            "plan_state_pin": "initial_state_file_sha256" not in case
            or case["initial_state_file_sha256"] == digest(expected_bytes),
            "checkpoint": metadata.get("model_stage") == stage_for(plan["arms"][arm], plan),
            "policy_seed": type(metadata.get("diagnostic_rng_reset_per_connection")) is int
            and metadata["diagnostic_rng_reset_per_connection"] == seed,
            "model_contract": metadata.get("action_horizon") == plan.get("action_horizon", 10)
            and metadata.get("state_input") is plan.get("discrete_state_input", False),
            "holding_controller_initialization": bool(result.get("controller_initialization"))
            if plan.get("holding_controller_initialization") == "canonical"
            and case["role"] == "training_holding_state" else True,
            "normalization": metadata.get("norm_stats_sha256") == plan.get("normalization_sha256")
            if "normalization_sha256" in plan else True,
            "budget_and_replanning": equal_int("max_steps", case["max_steps"])
            and equal_int("replan_steps", 1) and equal_int("post_release_steps_requested", FOLLOWUP),
            "control_steps": equal_int("control_steps", n),
            "On_labels": equal_bool("success", metrics["any_On"])
            and equal_bool("final_benchmark_success", metrics["final_On"])
            and equal_event("first_benchmark_success_control_steps", metrics["first_On_control_step"]),
            "release_labels": equal_event("first_ten_step_release_retreat_control_steps", event)
            and equal_bool("ten_step_release_retreat_verified", event is not None),
            "followup_labels": equal_int("post_release_steps_executed", observed)
            and equal_bool("post_release_followup_complete", metrics["followup"]["quantity_complete"]),
            "video_frame_count": equal_int("video_frames", n + 1),
        }
        status = "COMPLETE"
        if (event is None and n < case["max_steps"]) or (event is not None and observed < FOLLOWUP):
            status = "INCOMPLETE"
        elif event is not None and n != event + FOLLOWUP:
            checks["stop_after_fifty"] = False
        if not all(checks.values()):
            status = "INVALID"
        return {**identity, "status": status, "validations": checks, "metrics": metrics,
                "initial_state_sha256": initial_hash, "source_scene_sha256": digest(source_json),
                "model_metadata": metadata, "source_sha256": {name: digest(data) for name, data in raw.items()}}
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {**identity, "status": "INVALID", "error": str(error)}


def summarize(plan_path, runs, *, project_root=ROOT):
    plan_bytes = Path(plan_path).read_bytes()
    plan = json.loads(plan_bytes)
    cases, seeds = plan["evaluation_cases"], plan["policy_seeds"]
    if (set(plan["arms"]) != {"control", "candidate"} or not cases or not seeds
            or len({case["case"] for case in cases}) != len(cases)
            or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds)
            or any(not isinstance(case["case"], str) or Path(case["case"]).name != case["case"]
                   or case["case"] in {"", ".", ".."} or not isinstance(case["role"], str)
                   or type(case["max_steps"]) is not int or case["max_steps"] < 1 for case in cases)):
        raise ValueError("Invalid control/candidate case/seed plan")
    records = [read_run(Path(runs) / arm / f"seed-{seed}" / case["case"], plan, case, arm, seed, Path(project_root))
               for arm in plan["arms"] for seed in seeds for case in cases]
    lookup = {(r["arm"], r["case"], r["policy_seed"]): r for r in records}
    pairs = []
    for case in cases:
        for seed in seeds:
            a, b = [lookup[(arm, case["case"], seed)] for arm in ["control", "candidate"]]
            pair = {"case": case["case"], "role": case["role"], "policy_seed": seed}
            if any(r["status"] != "COMPLETE" for r in [a, b]):
                pairs.append({**pair, "status": "INCOMPLETE_OR_INVALID"})
                continue
            same = a["initial_state_sha256"] == b["initial_state_sha256"] and a["source_scene_sha256"] == b["source_scene_sha256"]
            both = all(r["metrics"]["followup"]["strict_success"] is True for r in [a, b])
            pairs.append({**pair, "status": "PAIRED" if same else "INVALID_INITIAL_PAIR",
                          "strict_success": {r["arm"]: r["metrics"]["followup"]["strict_success"] is True for r in [a, b]},
                          "candidate_minus_control_completion_steps": b["metrics"]["completion_control_steps"]
                          - a["metrics"]["completion_control_steps"] if same and both else None})
    def counts(selected):
        available = [r for r in selected if r["status"] == "COMPLETE"]
        return {"planned_trials": len(selected), "completed_trials": len(available),
                "strict_successes": sum(r["metrics"]["followup"]["strict_success"] is True for r in available),
                "task_On_trials": sum(r["metrics"]["any_On"] for r in available),
                "no_release_right_censored_at_budget": sum(r["metrics"]["right_censored_without_release_at_budget"] for r in available)}

    summaries = {}
    for arm in plan["arms"]:
        selected = [r for r in records if r["arm"] == arm]
        summaries[arm] = {**counts(selected), "by_role": {
            role: counts([r for r in selected if r["role"] == role]) for role in dict.fromkeys(r["role"] for r in selected)}}
    invalid = any(r["status"] == "INVALID" for r in records) or any(r["status"] == "INVALID_INITIAL_PAIR" for r in pairs)
    complete = sum(r["status"] == "COMPLETE" for r in records)
    return {"status": "INVALID" if invalid else "COMPLETE" if complete == len(records) else "INCOMPLETE",
            "plan_sha256": digest(plan_bytes), "plan": plan, "planned_runs": len(records),
            "completed_valid_runs": complete, "arms": summaries, "runs": records, "pairs": pairs,
            "limits": ["Case roles are predeclared in the plan; training-source cases and independent-position cases must be reported separately.",
                       "Small paired development study, not a benchmark success rate or unseen-position guarantee. One training seed42; policy seeds vary sampling only.",
                       "Strict success needs the first ten consecutive release gates followed by fifty policy steps maintaining the same gates and <=5mm maximum bowl drift.",
                       "Missing, invalid, exceptional or short-followup trials are not silently counted as task failures. No release at the full budget is right censored.",
                       "Completion time includes the fifty-step followup. Paired time deltas require both strict successes and identical saved initial state.",
                       "Motion uses adjacent post-observation positions. Closed/not-On/low-action runs are diagnostic proxies, not verified grasps or causes.",
                       "No model, inference, simulator or training is loaded by this summarizer."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "evidence/negative-recovery-20261019/experiment-plan.json")
    parser.add_argument("--runs", type=Path, default=ROOT / "evaluations/negative-recovery-20261019")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.plan.resolve() or args.output.resolve().is_relative_to(args.runs.resolve()):
        parser.error("Summary output must be separate from plan and raw runs")
    report = summarize(args.plan.resolve(), args.runs.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ["status", "planned_runs", "completed_valid_runs"]}))


if __name__ == "__main__":
    main()
