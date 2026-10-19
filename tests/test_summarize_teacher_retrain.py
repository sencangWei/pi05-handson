"""Separate release events, observed instability and incomplete followup windows."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_teacher_retrain import analyze, paired_report, summarize


def rows(count=60):
    result = []
    for i in range(count):
        row = {"step": str(i), "done": "True"}
        for prefix, values in [("state", [i * .001, 0, 1.1, 0, 0, 0, .04, -.04]),
                               ("action", [0, 0, 0, 0, 0, 0, -1]),
                               ("post_eef", [0, 0, 1.1]), ("post_bowl", [0, 0, .9]),
                               ("post_finger", [.04, -.04])]:
            row.update({f"{prefix}_{j}": str(value) for j, value in enumerate(values)})
        result.append(row)
    return result


def test_followup_starts_after_tenth_gate_action_and_counts_exactly_fifty():
    data = analyze(rows(), .9)
    assert data["first_stable_release_control_step"] == 10
    follow = data["followup"]
    assert follow["observed_steps"] == 50 and follow["quantity_complete"]
    assert follow["strict_50_step_gate_maintained"]
    assert follow["maximum_bowl_drift_3d_m"] == 0
    assert data["all_motion_proxy"]["fresh_adjacent_eef_displacement_mm_per_control_step"]["median"] == pytest.approx(1)


def test_quantity_complete_does_not_certify_stability_or_hide_temporary_drift():
    data = rows()
    data[10]["action_6"] = "1"  # First followup action, control-step11.
    data[10]["post_bowl_0"] = ".01"
    report = analyze(data, .9)
    follow = report["followup"]
    assert follow["quantity_complete"] and not follow["strict_50_step_gate_maintained"]
    assert follow["reclose_command_steps"] == 1 and follow["first_reclose_control_step"] == 11
    assert follow["maintained"]["On"]["all_observed_steps"] is True
    assert follow["maintained"]["open_command"]["first_violation_control_step"] == 11
    assert follow["maximum_bowl_drift_xy_m"] == pytest.approx(.01)
    assert follow["final_bowl_drift_3d_m"] == 0


@pytest.mark.parametrize("field,value,term", [("done", "False", "On"),
                                              ("post_finger_0", "0", "actual_gap_ge_6cm"),
                                              ("post_eef_2", ".95", "retreat_ge_10cm")])
def test_each_post_release_stability_term_is_recomputed(field, value, term):
    data = rows()
    data[20][field] = value
    follow = analyze(data, .9)["followup"]
    assert follow["quantity_complete"] and not follow["strict_50_step_gate_maintained"]
    assert follow["maintained"][term]["first_violation_control_step"] == 21


def test_short_followup_is_unknown_even_if_all_observed_steps_pass():
    data = analyze(rows(20), .9, budget=20)
    follow = data["followup"]
    assert follow["observed_steps"] == 10 and follow["right_censored"]
    assert follow["strict_50_step_gate_maintained"] is None
    assert follow["maintained"]["full_gate"]["all_observed_steps"]
    assert data["time_to_stable_release"]["right_censored_at_control_steps"] is None


def test_no_gate_is_censored_at_budget_not_assigned_a_completion_time():
    data = rows(20)
    for row in data:
        row["done"] = "False"
    report = analyze(data, .9, budget=20)
    assert report["first_stable_release_control_step"] is None
    assert report["time_to_stable_release"]["right_censored_at_control_steps"] == 20
    assert report["followup"]["strict_50_step_gate_maintained"] is None
    assert report["followup"]["maintained"]["On"]["all_observed_steps"] is None


def test_open_contact_lift_is_not_carry_and_fresh_displacements_stay_inside_carry():
    data = rows(5)
    for i in range(3):
        data[i].update(done="False", post_bowl_2=".94")
    for i in [1, 2]:
        data[i].update(action_6="1", action_0=".1")
    data[3]["state_0"] = "9"
    report = analyze(data, .9)
    assert report["first_lift_over_2cm_control_step"] == 1
    assert report["carry_proxy"]["first_control_step"] == 2
    assert report["carry_proxy"]["frames"] == 2
    assert report["carry_proxy"]["fresh_adjacent_eef_displacement_mm_per_control_step"]["n"] == 1
    assert report["carry_proxy"]["action_xyz_norm"]["median"] == pytest.approx(.1)


def plan():
    return {"arms": {"control": "control_exp", "candidate": "candidate_exp"},
            "new_updates_per_arm": 2000, "normalization_sha256": "a" * 64,
            "action_horizon": 10, "discrete_state_input": False,
            "followup_evaluation": {"physical_cases": ["zero", "negative54", "negative55", "positive57"],
                                    "policy_seeds": [0, 1, 2], "max_control_steps": 800,
                                    "replan_steps": 1, "layout_id": 24, "sim_seed": 42}}


def write_trial(root, spec, arm, *, corrupt=None):
    directory = root / "runs" / arm
    directory.mkdir(parents=True)
    source = root / "sources" / "zero"
    source.mkdir(parents=True, exist_ok=True)
    state = np.array([1., 2., 3.])
    state_sha = hashlib.sha256(state.tobytes()).hexdigest()
    scene = json.dumps({"initial_state_sha256": state_sha}).encode()
    (source / "scene.json").write_bytes(scene)
    np.save(directory / "initial_state.npy", state)
    data = rows()
    with (directory / "actions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    stage = f"{spec['arms'][arm]}/1999"
    result = {"model_metadata": {"model_stage": stage, "checkpoint": f"/checkpoints/{stage}",
                                 "diagnostic_rng_reset_per_connection": 0, "norm_stats_sha256": "a" * 64,
                                 "action_horizon": 10, "state_input": False},
              "saved_scene_replay": str(source), "source_scene_sha256": hashlib.sha256(scene).hexdigest(),
              "initial_state_sha256": state_sha, "bowl_edit": {"after_xyz": [0, 0, .9]},
              "control_steps": 60, "success": True, "final_benchmark_success": True,
              "first_benchmark_success_control_steps": 1, "first_ten_step_release_retreat_control_steps": 10,
              "ten_step_release_retreat_verified": True, "post_release_steps_requested": 50,
              "post_release_steps_executed": 50, "post_release_followup_complete": True,
              "stop_reason": "stable_release_retreat_followup", "max_steps": 800, "replan_steps": 1,
              "seed": 42, "layout_id": 24, "guard_checked_no_deep_overlap": True,
              "video_frames": 61, "video_fps": 10, "wall_duration_s": 2.}
    if corrupt:
        result.update(corrupt)
    (directory / "result.json").write_text(json.dumps(result))
    return directory


def test_paired_report_reads_real_csv_and_leaves_unfinished_slots_pending(tmp_path):
    spec = plan()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(spec))
    write_trial(tmp_path, spec, "control")
    write_trial(tmp_path, spec, "candidate")
    result = summarize(path, tmp_path / "runs")
    assert result["status"] == "RUNNING" and result["completed_valid_runs"] == 2
    assert len(result["pending_slots"]) == 22
    pair = result["paired_case_seed_results"][0]
    assert pair["status"] == "PAIRED" and pair["candidate_minus_control_steps_to_stable_release"] == 0
    assert pair["strict_50_step_gate_maintained"] == {"control": True, "candidate": True}


def test_quantity_metadata_must_match_recomputed_csv(tmp_path):
    spec = plan()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(spec))
    write_trial(tmp_path, spec, "control", corrupt={"post_release_steps_executed": 49})
    result = summarize(path, tmp_path / "runs")
    assert result["status"] == "INVALID"
    assert not result["runs"][0]["validations"]["followup_quantity_only"]


def test_pair_does_not_substitute_budget_for_censored_time_or_pair_wrong_state():
    spec = plan()
    a, b = analyze(rows(), .9), analyze([{**r, "done": "False"} for r in rows()], .9, budget=60)
    records = [{"arm": arm, "case": "zero", "policy_seed": 0, "status": "COMPLETE", "metrics": metrics,
                "initial_state_sha256": "same", "source_scene_sha256": "same"}
               for arm, metrics in [("control", a), ("candidate", b)]]
    pair = paired_report(records, spec)[0]
    assert pair["candidate_minus_control_steps_to_stable_release"] is None
    assert not pair["both_followups_quantity_complete"]
    records[1]["initial_state_sha256"] = "wrong"
    assert paired_report(records, spec)[0]["status"] == "INVALID_INITIAL_PAIR"


def test_duplicate_runs_are_invalid_instead_of_picking_the_last(tmp_path):
    spec = plan()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(spec))
    directory = write_trial(tmp_path, spec, "control")
    copy = tmp_path / "runs" / "duplicate"
    import shutil
    shutil.copytree(directory, copy)
    result = summarize(path, tmp_path / "runs")
    assert result["status"] == "INVALID"
    assert all(run["status"] == "INVALID" for run in result["runs"])


def test_no_outputs_are_running_without_claiming_verified_initial_identity(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan()))
    result = summarize(path, tmp_path / "not_created_yet")
    assert result["status"] == "RUNNING" and result["completed_valid_runs"] == 0
    assert len(result["pending_slots"]) == 24
    assert set(result["same_case_initial_identity_over_available_runs"].values()) == {None}


@pytest.mark.parametrize("exception", [False, True])
def test_encoding_in_progress_is_pending_but_finished_exception_is_not_hidden(tmp_path, exception):
    spec = plan()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(spec))
    directory = write_trial(tmp_path, spec, "control")
    result_path = directory / "result.json"
    raw = json.loads(result_path.read_text())
    raw.pop("video_frames")
    if exception:
        raw["exception"] = "fixture encoding failed"
    result_path.write_text(json.dumps(raw))
    result = summarize(path, tmp_path / "runs")
    assert result["completed_valid_runs"] == 0
    if exception:
        assert result["status"] == "INVALID" and result["runs"][0]["status"] == "ERROR"
    else:
        assert result["status"] == "RUNNING" and result["runs"] == []
