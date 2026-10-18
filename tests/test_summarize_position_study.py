"""Saved CSV acceptance gates and bounded matrix aggregation."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_position_study import analyze_rows, build_summary, render_html


def make_rows(count=10):
    rows = []
    for i in range(count):
        row = {"step": str(i), "done": "True"}
        for prefix, values in [("state", [i * .001, 0, 1.1, 0, 0, 0, .04, -.04]),
                               ("action", [0, 0, 0, 0, 0, 0, -1]),
                               ("post_eef", [0, 0, 1.1]), ("post_bowl", [0, 0, .9]),
                               ("post_finger", [.04, -.04])]:
            row.update({f"{prefix}_{j}": str(value) for j, value in enumerate(values)})
        rows.append(row)
    return rows


def test_stable_gate_requires_ten_unbroken_post_observations():
    rows = make_rows(19)
    rows[9]["action_6"] = "1"
    data = analyze_rows(rows, .9, np.array([.01, 0]))
    assert data["first_ten_step_release_retreat_control_step"] is None
    rows.append(make_rows(20)[19])
    data = analyze_rows(rows, .9, np.array([.01, 0]))
    assert data["first_ten_step_release_retreat_control_step"] == 20
    assert data["final_bowl_plate_xy_center_error_m"] == pytest.approx(.01)
    assert data["fresh_adjacent_eef_displacement_mm"]["p10_p50_p90"][1] == pytest.approx(1)


@pytest.mark.parametrize("field,value", [("done", "False"), ("post_finger_0", "0"),
                                        ("post_eef_2", ".95"), ("action_6", "0")])
def test_each_gate_term_matters(field, value):
    rows = make_rows()
    rows[5][field] = value
    assert not analyze_rows(rows, .9, np.array([0, 0]))["ten_step_release_retreat_observed"]


def test_carry_fresh_displacement_does_not_cross_noncarry_rows():
    rows = make_rows(5)
    for i in [1, 2]:
        rows[i].update(done="False", action_6="1", action_0=".1", post_bowl_2=".94")
    rows[3]["state_0"] = "5"
    data = analyze_rows(rows, .9, np.zeros(2))
    assert data["first_close_control_step"] == 2
    assert data["first_lift_over_2cm_control_step"] == 2
    assert data["carry_proxy"]["frames"] == 2
    assert data["carry_proxy"]["fresh_adjacent_eef_displacement_mm"]["n"] == 1
    assert data["carry_proxy"]["action_xyz_norm"]["p10_p50_p90"][1] == pytest.approx(.1)


def test_running_matrix_skips_unregistered_partial_output(tmp_path):
    study = tmp_path / ".local/position-study-v1"
    study.mkdir(parents=True)
    plan = {"cases": [{"case": "one", "dx_m": 0, "dy_m": 0}], "control_settings": {},
            "predeclared_models": {"m1": "a/1", "m2": "b/2", "m3": "c/3"}}
    for name, value in [("matrix-plan.json", plan), ("matrix-results.json", {"status": "RUNNING", "plan": plan, "runs": []}),
                        ("placement-grid.json", {"cases": []})]:
        (study / name).write_text(json.dumps(value))
    data = build_summary(tmp_path, study)
    assert data["status"] == "RUNNING" and data["pending_runs"] == 3
    page = render_html(data, tmp_path, tmp_path / "evidence/position-study-v1")
    assert page.count('<article>') == 3
    assert "data-all=\"pause\"" in page and "v.currentTime=t" in page


def test_invalid_steps_and_nonfinite_state_are_rejected():
    rows = make_rows()
    rows[2]["step"] = "7"
    with pytest.raises(ValueError, match="contiguous"):
        analyze_rows(rows, .9, np.zeros(2))
    rows[2]["step"] = "2"
    rows[2]["state_0"] = "nan"
    with pytest.raises(ValueError, match="Nonfinite"):
        analyze_rows(rows, .9, np.zeros(2))


@pytest.mark.parametrize("corrupt", [None, "initial_sha", "completion_flag"])
def test_completed_trial_hash_and_recomputed_stopflag(tmp_path, corrupt):
    study = tmp_path / ".local/position-study-v1"
    source, run = study / "sources/one", study / "runs/m__one"
    source.mkdir(parents=True)
    run.mkdir(parents=True)
    source_scene = b'{}'
    (source / "scene.json").write_bytes(source_scene)
    source_sha = hashlib.sha256(source_scene).hexdigest()
    state = np.array([1., 2.])
    sha = hashlib.sha256(state.tobytes()).hexdigest()
    np.save(run / "initial_state.npy", state)
    rows = make_rows()
    with (run / "actions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (run / "video.mp4").touch()
    settings = {"budget": 800, "replan_steps": 1, "sim_seed": 42, "layout_id": 24,
                "policy_seed_per_trial": 0}
    case = {"case": "one", "dx_m": 0., "dy_m": 0., "source": str(source.relative_to(tmp_path)),
            "source_scene_file_sha256": source_sha, "initial_state_sha256": sha}
    plan = {"cases": [case], "control_settings": settings, "predeclared_models": {"m": "abc/9"}}
    entry = {"model": "m", "case": "one", "output": str(run.relative_to(tmp_path)),
             "checkpoint": "checkpoints/abc/9", "steps": 10, "initial_state_sha256": sha,
             "firstOn": 1, "finalOn": True, "completion": True}
    result = {"bowl_edit": {"after_xyz": [0., 0., .9]}, "initial_state_sha256": sha,
              "source_scene_sha256": source_sha, "control_steps": 10, "success": True,
              "final_benchmark_success": True, "first_benchmark_success_control_steps": 1,
              "ten_step_release_retreat_verified": True, "stop_reason": "stable_release_retreat",
              "max_steps": 800, "replan_steps": 1, "seed": 42, "layout_id": 24,
              "guard_checked_no_deep_overlap": True, "video_frames": 11, "video_fps": 10,
              "model_metadata": {"model_stage": "abc/9", "checkpoint": str(tmp_path / "checkpoints/abc/9"),
                                 "diagnostic_rng_reset_per_connection": 0}}
    if corrupt == "initial_sha":
        result["initial_state_sha256"] = "bad"
    elif corrupt == "completion_flag":
        result["ten_step_release_retreat_verified"] = False
    for path, value in [(study / "matrix-plan.json", plan),
                        (study / "matrix-results.json", {"status": "COMPLETE", "plan": plan, "runs": [entry]}),
                        (study / "placement-grid.json", {"cases": [{"case": "one", "dx_m": 0., "dy_m": 0.,
                                                                    "original_positions": {"plate_1": [0., 0., .9]}}]}),
                        (run / "result.json", result)]:
        path.write_text(json.dumps(value))
    data = build_summary(tmp_path, study)
    assert data["status"] == ("INVALID" if corrupt else "COMPLETE")
