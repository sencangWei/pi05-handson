"""CPU-only counterexamples for release gates, missing trials and paired identities."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_negative_recovery import analyze, stage_for, summarize


def rows(count=60, on="True"):
    result = []
    for i in range(count):
        row = {"step": str(i), "done": on}
        for prefix, values in [("state", [99, 0, 1.1, 0, 0, 0, .04, -.04]),
                               ("action", [0, 0, 0, 0, 0, 0, -1]),
                               ("post_eef", [i*.001, 0, 1.1]), ("post_bowl", [0, 0, .9]),
                               ("post_finger", [.04, -.04])]:
            row.update({f"{prefix}_{j}": str(value) for j, value in enumerate(values)})
        result.append(row)
    return result


def test_first_ten_then_fifty_and_post_to_post_displacement():
    data = analyze(rows(), 800)
    assert data["first_ten_step_release_control_step"] == 10
    assert data["followup"]["observed_steps"] == 50
    assert data["followup"]["strict_success"] is True
    assert data["completion_control_steps"] == 60
    movement = data["diagnostic_proxies"]["post_to_post_eef_displacement_mm"]
    assert movement["n"] == 59 and movement["median"] == pytest.approx(1)


@pytest.mark.parametrize("field,value", [("done", "False"), ("action_6", "1"),
                                         ("post_finger_0", "0"), ("post_eef_2", ".95")])
def test_any_followup_gate_violation_fails_even_if_final_frame_is_good(field, value):
    data = rows()
    data[10][field] = value
    report = analyze(data, 800)
    assert report["followup"]["quantity_complete"]
    assert report["followup"]["strict_success"] is False
    assert report["followup"]["first_gate_violation_control_step"] == 11
    assert report["completion_control_steps"] is None


def test_transient_bowl_drift_fails_despite_all_release_gates_and_return_home():
    data = rows()
    data[20]["post_bowl_0"] = ".008"
    report = analyze(data, 800)
    assert report["followup"]["all_observed_gate_steps_pass"]
    assert report["followup"]["maximum_bowl_drift_m"] == pytest.approx(.008)
    assert report["followup"]["strict_success"] is False


def test_false_strings_are_not_truthy_and_missing_fifty_is_unknown():
    report = analyze(rows(20, "False"), 20)
    assert report["any_On"] is False
    assert report["first_ten_step_release_control_step"] is None
    assert report["right_censored_without_release_at_budget"]
    follow = analyze(rows(20), 20)["followup"]
    assert follow["observed_steps"] == 10
    assert follow["strict_success"] is None
    with pytest.raises(ValueError, match="done"):
        analyze([{**row, "done": "maybe"} for row in rows()], 800)


def test_actual_gap_is_not_sum_of_same_sign_finger_coordinates():
    data = rows()
    for row in data:
        row["post_finger_1"] = ".04"
    assert analyze(data, 60)["first_ten_step_release_control_step"] is None


def test_low_action_proxy_longest_run_and_displacement_never_cross_mask_gap():
    data = rows(6, "False")
    for row in data:
        row.update(action_6="1", action_0=".04")
    data[2]["action_0"] = ".08"  # Strictly less than .08, not <=.
    for row, x in zip(data, [0, .001, 8, 9, 9.001, 9.002], strict=True):
        row["post_eef_0"] = str(x)
    report = analyze(data, 6)["diagnostic_proxies"]
    assert report["longest_low_action_closed_not_On"] == {
        "steps": 3, "start_control_step": 4, "end_control_step": 6}
    motion = report["post_to_post_eef_displacement_within_low_run_mm"]
    assert motion["n"] == 3 and motion["mean"] == pytest.approx(1)


def plan(tmp_path, cases=2, seeds=(0, 1)):
    value = {"arms": {"control": "control_exp", "candidate": "candidate_exp"},
             "new_updates_per_arm": 2000, "training_seed": 42, "policy_seeds": list(seeds),
             "normalization_sha256": "norm", "action_horizon": 10, "discrete_state_input": False,
             "evaluation_cases": []}
    for i in range(cases):
        source = tmp_path / "sources" / f"case{i}"
        source.mkdir(parents=True)
        state = np.array([1., i, 3.])
        np.save(source / "initial_state.npy", state)
        scene = {"initial_state_sha256": hashlib.sha256(state.tobytes()).hexdigest()}
        (source / "scene.json").write_text(json.dumps(scene))
        value["evaluation_cases"].append({"case": f"case{i}", "source_scene": str(source),
                                           "role": "training_source" if i == 0 else "independent_position_holdout",
                                           "max_steps": 80 if i == 0 else 60})
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(value))
    return path, value


def trial(root, spec, arm="control", case_index=0, seed=0, data=None, corrupt=None):
    data = rows() if data is None else data
    case = spec["evaluation_cases"][case_index]
    directory = root / arm / f"seed-{seed}" / case["case"]
    directory.mkdir(parents=True)
    source = Path(case["source_scene"])
    state = np.load(source / "initial_state.npy", allow_pickle=False)
    np.save(directory / "initial_state.npy", state)
    scene = json.loads((source / "scene.json").read_text())
    (directory / "scene.json").write_text(json.dumps(scene))
    with (directory / "actions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    metrics = analyze(data, case["max_steps"])
    event, n = metrics["first_ten_step_release_control_step"], len(data)
    result = {"model_metadata": {"model_stage": stage_for(spec["arms"][arm], spec),
                                 "diagnostic_rng_reset_per_connection": seed, "action_horizon": 10,
                                 "state_input": False, "norm_stats_sha256": "norm"},
              "initial_state_sha256": scene["initial_state_sha256"],
              "saved_scene_replay": str(source),
              "source_scene_sha256": hashlib.sha256((source / "scene.json").read_bytes()).hexdigest(),
              "max_steps": case["max_steps"], "replan_steps": 1,
              "control_steps": n, "success": metrics["any_On"],
              "final_benchmark_success": metrics["final_On"],
              "first_benchmark_success_control_steps": metrics["first_On_control_step"],
              "first_ten_step_release_retreat_control_steps": event,
              "ten_step_release_retreat_verified": event is not None, "post_release_steps_requested": 50,
              "post_release_steps_executed": metrics["followup"]["observed_steps"],
              "post_release_followup_complete": metrics["followup"]["quantity_complete"],
              "video_frames": n+1}
    if corrupt:
        result.update(corrupt)
    (directory / "result.json").write_text(json.dumps(result))
    return directory


def test_plan_slots_not_fixed_four_cases_or_twenty_four_and_missing_not_failure(tmp_path):
    path, spec = plan(tmp_path)
    trial(tmp_path / "runs", spec)
    result = summarize(path, tmp_path / "runs")
    assert result["planned_runs"] == 8
    assert result["status"] == "INCOMPLETE" and result["completed_valid_runs"] == 1
    assert result["arms"]["control"]["strict_successes"] == 1
    assert result["arms"]["candidate"]["completed_trials"] == 0
    assert sum(r["status"] == "INCOMPLETE" for r in result["runs"]) == 7
    assert {r["role"] for r in result["runs"]} == {"training_source", "independent_position_holdout"}


def test_full_plan_success_and_paired_completion_delta(tmp_path):
    path, spec = plan(tmp_path)
    for arm in spec["arms"]:
        for seed in spec["policy_seeds"]:
            for i in range(2):
                trial(tmp_path / "runs", spec, arm, i, seed)
    result = summarize(path, tmp_path / "runs")
    assert result["status"] == "COMPLETE" and result["completed_valid_runs"] == 8
    assert all(p["status"] == "PAIRED" and p["candidate_minus_control_completion_steps"] == 0 for p in result["pairs"])
    assert result["arms"]["control"]["strict_successes"] == 4
    roles = result["arms"]["control"]["by_role"]
    assert roles["training_source"]["strict_successes"] == 2
    assert roles["independent_position_holdout"]["strict_successes"] == 2


@pytest.mark.parametrize("corrupt", [{"success": True}, {"success": "False"},
                                     {"ten_step_release_retreat_verified": "False"}])
def test_forged_success_or_string_json_booleans_invalid_not_success(tmp_path, corrupt):
    path, spec = plan(tmp_path, cases=1, seeds=(0,))
    spec["evaluation_cases"][0]["max_steps"] = 60
    path.write_text(json.dumps(spec))
    trial(tmp_path / "runs", spec, data=rows(60, "False"), corrupt=corrupt)
    result = summarize(path, tmp_path / "runs")
    assert result["status"] == "INVALID"
    assert result["arms"]["control"]["strict_successes"] == 0
    assert result["arms"]["control"]["completed_trials"] == 0


def test_only_On_is_not_release_and_short_followup_is_incomplete(tmp_path):
    path, spec = plan(tmp_path, cases=1, seeds=(0,))
    spec["evaluation_cases"][0]["max_steps"] = 60
    path.write_text(json.dumps(spec))
    closed = rows()
    for row in closed:
        row["action_6"] = "1"
    trial(tmp_path / "runs", spec, "control", data=closed)
    trial(tmp_path / "runs", spec, "candidate", data=rows(20))
    result = summarize(path, tmp_path / "runs")
    assert result["status"] == "INCOMPLETE"
    assert result["arms"]["control"]["task_On_trials"] == 1
    assert result["arms"]["control"]["strict_successes"] == 0
    assert result["arms"]["candidate"]["completed_trials"] == 0
    assert result["pairs"][0]["status"] == "INCOMPLETE_OR_INVALID"
    assert result["runs"][1]["metrics"]["followup"]["strict_success"] is None


def test_wrong_checkpoint_seed_and_initial_state_invalid(tmp_path):
    path, spec = plan(tmp_path, cases=1, seeds=(0,))
    directory = trial(tmp_path / "runs", spec)
    result_path = directory / "result.json"
    raw = json.loads(result_path.read_text())
    raw["model_metadata"]["diagnostic_rng_reset_per_connection"] = 1
    raw["model_metadata"]["model_stage"] = "wrong/1999"
    result_path.write_text(json.dumps(raw))
    np.save(directory / "initial_state.npy", np.array([9., 9., 9.]))
    result = summarize(path, tmp_path / "runs")
    assert result["runs"][0]["status"] == "INVALID"
    checks = result["runs"][0]["validations"]
    assert not checks["initial_state"] and not checks["checkpoint"] and not checks["policy_seed"]


def test_plan_hash_pins_reject_a_changed_source_even_when_trial_matches_it(tmp_path):
    path, spec = plan(tmp_path, cases=1, seeds=(0,))
    spec["evaluation_cases"][0].update(scene_sha256="wrong", initial_state_file_sha256="wrong")
    path.write_text(json.dumps(spec))
    trial(tmp_path / "runs", spec)
    result = summarize(path, tmp_path / "runs")
    checks = result["runs"][0]["validations"]
    assert not checks["plan_scene_pin"] and not checks["plan_state_pin"]
    assert result["status"] == "INVALID"
