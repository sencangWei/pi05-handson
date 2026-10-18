"""Summarize completed saved-scene trials without loading a policy or simulator."""
import argparse
import csv
import hashlib
import html
import io
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def distribution(values):
    values = np.asarray(values, dtype=float)
    return {"n": len(values), "mean": float(values.mean()) if len(values) else None,
            "p10_p50_p90": np.quantile(values, [.1, .5, .9]).tolist() if len(values) else None}


def first_step(mask):
    indices = np.flatnonzero(mask)
    return int(indices[0]) + 1 if len(indices) else None


def analyze_rows(rows, bowl_initial_z, plate_xy):
    """CSV step zero is the first action; returned event steps count executed actions."""
    if not rows:
        raise ValueError("Completed trial has no action rows")
    steps = [int(r["step"]) for r in rows]
    if steps != list(range(len(rows))):
        raise ValueError("CSV action steps are not contiguous from zero")

    def array(prefix, size):
        values = np.array([[float(r[f"{prefix}_{i}"]) for i in range(size)] for r in rows])
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite {prefix} values")
        return values

    states, actions = array("state", 8), array("action", 7)
    eef, bowl, fingers = array("post_eef", 3), array("post_bowl", 3), array("post_finger", 2)
    if any(r["done"] not in {"True", "False", "true", "false", "1", "0"} for r in rows):
        raise ValueError("Invalid CSV done value")
    on = np.array([r["done"].lower() in {"true", "1"} for r in rows])
    opened, closed = actions[:, 6] < 0, actions[:, 6] > 0
    gap = np.abs(fingers).sum(axis=1)
    clearance = np.linalg.norm(eef - bowl, axis=1)
    gate = on & opened & (gap >= .06) & (clearance >= .10)
    streak, first_gate = 0, None
    for i, passes in enumerate(gate):
        streak = streak + 1 if passes else 0
        if streak >= 10 and first_gate is None:
            first_gate = i + 1
    carry = closed & (bowl[:, 2] > bowl_initial_z + .03) & ~on
    adjacent_carry = carry[:-1] & carry[1:]
    # Fresh pre-action states on consecutive control steps, not cached post/pre pairs.
    motion_mm = np.linalg.norm(np.diff(states[:, :3], axis=0), axis=1) * 1000
    lift_step = first_step(bowl[:, 2] > bowl_initial_z + .02)
    if first_gate:
        diagnosis = "stable_release_retreat_observed"
    elif on.any():
        diagnosis = "On_without_stable_release_retreat"
    elif lift_step:
        diagnosis = "lift_proxy_without_On"
    elif closed.any():
        diagnosis = "close_command_without_lift_proxy"
    else:
        diagnosis = "no_close_command_by_stop"
    return {
        "control_steps": len(rows), "first_close_control_step": first_step(closed),
        "first_lift_over_2cm_control_step": lift_step,
        "first_On_control_step": first_step(on), "any_On": bool(on.any()), "final_On": bool(on[-1]),
        "first_ten_step_release_retreat_control_step": first_gate,
        "ten_step_release_retreat_observed": first_gate is not None,
        "final_ten_step_release_retreat": streak >= 10,
        "final_gate_streak_steps": streak,
        "On_to_stable_release_retreat_steps": first_gate - first_step(on) if first_gate else None,
        "first_carry_proxy_control_step": first_step(carry),
        "carry_proxy": {"definition": "close command + post bowl > initial z + 3cm + not On; not verified grasp",
                        "frames": int(carry.sum()), "action_xyz_norm": distribution(np.linalg.norm(actions[carry, :3], axis=1)),
                        "fresh_adjacent_eef_displacement_mm": distribution(motion_mm[adjacent_carry])},
        "all_action_xyz_norm": distribution(np.linalg.norm(actions[:, :3], axis=1)),
        "fresh_adjacent_eef_displacement_mm": distribution(motion_mm),
        "tail_open_command_fraction_last50": float(opened[-50:].mean()),
        "tail_open_command_fraction_last10": float(opened[-10:].mean()),
        "final_finger_gap_m": float(gap[-1]), "final_eef_bowl_distance_m": float(clearance[-1]),
        "final_bowl_plate_xy_center_error_m": float(np.linalg.norm(bowl[-1, :2] - plate_xy)),
        "diagnostic_phase_proxy": diagnosis,
    }


def plate_reference(grid, case):
    matches = [c for c in grid["cases"] if abs(c["dx_m"] - case["dx_m"]) < 1e-9
               and abs(c["dy_m"] - case["dy_m"]) < 1e-9 and "original_positions" in c]
    if not matches:
        matches = [c for c in grid["cases"] if "original_positions" in c]
    if not matches:
        raise ValueError("Placement grid has no original plate position")
    reference = matches[0]
    plate = np.asarray(reference["original_positions"]["plate_1"])
    # Same layout must have a consistent plate even for cases not in the static grid.
    for entry in grid["cases"]:
        other = entry.get("original_positions", {}).get("plate_1")
        if other is not None and not np.allclose(plate, other, rtol=0, atol=1e-6):
            raise ValueError("Placement grid original plate positions disagree")
    return plate[:2], {"case": reference["case"], "plate_xyz": plate.tolist(),
                       "exact_offset_case_match": bool(matches and reference["dx_m"] == case["dx_m"]
                                                       and reference["dy_m"] == case["dy_m"])}


def summarize_run(root, case, model, entry, settings, grid, expected_stage):
    record = {"case": case["case"], "model": model, "status": "RUNNING"}
    if entry is None:
        return record
    directory = root / entry["output"]
    if entry.get("exception"):
        record.update(status="ERROR", exception=entry["exception"], source_directory=entry["output"])
        return record
    result_path, csv_path = directory / "result.json", directory / "actions.csv"
    if not result_path.exists() or not csv_path.exists() or not (directory / "video.mp4").exists():
        record["pending_reason"] = "Completed matrix entry is missing result/CSV/video"
        return record
    result_bytes, csv_bytes = result_path.read_bytes(), csv_path.read_bytes()
    result = json.loads(result_bytes)
    if "video_frames" not in result:
        record["pending_reason"] = "Video encoding/final result update not complete"
        return record
    plate, plate_info = plate_reference(grid, case)
    rows = list(csv.DictReader(io.StringIO(csv_bytes.decode())))
    metrics = analyze_rows(rows, result["bowl_edit"]["after_xyz"][2], plate)
    expected = case["initial_state_sha256"]
    state = np.load(directory / "initial_state.npy", allow_pickle=False)
    state_sha = digest(np.asarray(state).tobytes())
    source_path = root / case["source"] / "scene.json"
    validations = {
        "initial_sha_plan_result_entry_file": expected == result.get("initial_state_sha256")
        == entry.get("initial_state_sha256") == state_sha,
        "source_scene_file_sha": digest(source_path.read_bytes()) == case["source_scene_file_sha256"],
        "source_scene_result_sha": result.get("source_scene_sha256") == case["source_scene_file_sha256"],
        "control_steps_csv_result_entry": len(rows) == result["control_steps"] == entry["steps"],
        "On_csv_result_entry": metrics["any_On"] == result["success"]
        and metrics["final_On"] == result["final_benchmark_success"] == entry["finalOn"],
        "first_On_csv_result_entry": metrics["first_On_control_step"]
        == result["first_benchmark_success_control_steps"] == entry["firstOn"],
        "stable_gate_csv_result_entry": metrics["ten_step_release_retreat_observed"]
        == result.get("ten_step_release_retreat_verified") == entry["completion"],
        "stopflag_matches_gate": (result["stop_reason"] == "stable_release_retreat")
        == metrics["ten_step_release_retreat_observed"],
        "stable_stop_is_first_gate": result["stop_reason"] != "stable_release_retreat"
        or metrics["first_ten_step_release_retreat_control_step"] == len(rows),
        "budget_and_replan": result["max_steps"] == settings["budget"]
        and len(rows) <= settings["budget"] and result["replan_steps"] == settings["replan_steps"],
        "seed_and_legal_scene": result["seed"] == settings["sim_seed"]
        and result["layout_id"] == settings["layout_id"] and result.get("guard_checked_no_deep_overlap") is True
        and result["model_metadata"].get("diagnostic_rng_reset_per_connection") == settings["policy_seed_per_trial"],
        "video_steps_frames_fps": result["video_frames"] == len(rows) + 1 and result["video_fps"] == 10,
        "checkpoint_identity": result["model_metadata"].get("model_stage") == expected_stage
        and Path(result["model_metadata"]["checkpoint"]).resolve() == (root / entry["checkpoint"]).resolve(),
    }
    record.update(status="COMPLETE" if all(validations.values()) else "INVALID",
                  source_directory=entry["output"], initial_state_sha256=state_sha,
                  source_sha256={"actions.csv": digest(csv_bytes), "result.json": digest(result_bytes)},
                  validations=validations, metrics=metrics, plate_reference=plate_info,
                  stop_reason=result["stop_reason"], video_frames=result["video_frames"], video_fps=10,
                  video_source=str((directory / "video.mp4").relative_to(root)),
                  model_metadata=result["model_metadata"],
                  budget_censored=len(rows) >= settings["budget"] and not metrics["ten_step_release_retreat_observed"],
                  observation_ends_at_stop=True)
    return record


def build_summary(root, study):
    plan_path, results_path = study / "matrix-plan.json", study / "matrix-results.json"
    plan_bytes, result_bytes = plan_path.read_bytes(), results_path.read_bytes()
    plan, results = json.loads(plan_bytes), json.loads(result_bytes)
    grid_path = study / "placement-grid.json"
    grid_bytes = grid_path.read_bytes()
    grid = json.loads(grid_bytes)
    if results.get("plan") != plan:
        raise ValueError("Matrix results do not match the frozen matrix plan")
    entries = {(r["model"], r["case"]): r for r in results.get("runs", [])}
    if len(entries) != len(results.get("runs", [])):
        raise ValueError("Matrix has duplicate model/case entries")
    runs = []
    for case in plan["cases"]:
        for model in plan["predeclared_models"]:
            try:
                run = summarize_run(root, case, model, entries.get((model, case["case"])),
                                    plan["control_settings"], grid, plan["predeclared_models"][model])
            except (ValueError, KeyError, OSError) as error:
                run = {"case": case["case"], "model": model, "status": "INVALID", "error": str(error)}
            runs.append(run)
    invalid = sum(r["status"] in {"INVALID", "ERROR"} for r in runs)
    pending = sum(r["status"] == "RUNNING" for r in runs)
    return {
        "status": "INVALID" if invalid else "RUNNING" if pending or results.get("status") == "RUNNING" else "COMPLETE",
        "inputs_sha256": {str(plan_path.relative_to(root)): digest(plan_bytes),
                          str(results_path.relative_to(root)): digest(result_bytes),
                          str(grid_path.relative_to(root)): digest(grid_bytes)},
        "plan": plan, "completed_runs": sum(r["status"] == "COMPLETE" for r in runs),
        "pending_runs": pending, "invalid_or_error_runs": invalid, "runs": runs,
        "definitions": {"control_step": "One CSV action; event steps are one-based action counts, CSV step is zero-based.",
                        "stable_gate": "10 consecutive post_On AND command<0 AND sum(abs(post_finger))>=.06m AND ||post_EEF-post_bowl||>=.10m",
                        "carry": "closed command>0 + post bowl z > initial saved bowl z+.03m + not On; diagnostic proxy, not verified grasp",
                        "motion": "Consecutive fresh pre-action state XYZ displacement in mm; no conversion to wall-clock velocity.",
                        "plate_error": "Final bowl-to-plate XY center distance using static grid original plate_1; not the On success predicate.",
                        "video": "Initial frame + one per action at 10 FPS. Playback aligns by control step, not physical or wall-clock time."},
        "limits": ["One policy seed (0), simulation seed42, layout24 and eight predeclared guarded legal cases; no population/generalization success-rate claim.",
                   "Unsuccessful budget-limited trials are right-censored at800 actions, not proof they can never finish.",
                   "Successful trials stop at their first ten-step stable release/retreat; later stability or recovery is not observed.",
                   "Diagnostic phase labels are proxies, not causal failure diagnoses or grasp verification.",
                   "Q4 positive-X/negative-Y legal-scene coverage gap remains; rejected placements were not evaluated.",
                   "Video links are local only. The remote repository does not contain these raw videos.",
                   "No policy loading, simulation, training or changes to source run outputs performed."],
    }


def render_html(summary, root, output):
    escape = html.escape
    models = list(summary["plan"]["predeclared_models"])
    lookup = {(r["case"], r["model"]): r for r in summary["runs"]}
    sections = []
    labels = [("control_steps", "控制步"), ("first_close_control_step", "首次闭爪"),
              ("first_lift_over_2cm_control_step", "抬碗>2cm代理"), ("first_On_control_step", "首次 On"),
              ("first_ten_step_release_retreat_control_step", "连续10步释放撤手"),
              ("final_bowl_plate_xy_center_error_m", "最终碗盘中心误差/m"),
              ("tail_open_command_fraction_last50", "末50步开爪比例"),
              ("diagnostic_phase_proxy", "阶段诊断代理")]
    for case in summary["plan"]["cases"]:
        cards = []
        for model in models:
            run = lookup[(case["case"], model)]
            if run["status"] != "COMPLETE":
                body = f'<p class="pending">{escape(run["status"])} · {escape(run.get("error", run.get("pending_reason", "等待本次矩阵记录")))}</p>'
            else:
                link = Path(os.path.relpath(root / run["video_source"], output)).as_posix()
                rows = "".join(f'<tr><th>{escape(label)}</th><td>{escape(format_value(run["metrics"][key]))}</td></tr>'
                               for key, label in labels)
                carry = run["metrics"]["carry_proxy"]
                for label, value in [("搬运代理帧数", carry["frames"]),
                                     ("搬运XYZ范数中位", median(carry["action_xyz_norm"])),
                                     ("搬运相邻位移/mm中位", median(carry["fresh_adjacent_eef_displacement_mm"]))]:
                    rows += f'<tr><th>{escape(label)}</th><td>{escape(format_value(value))}</td></tr>'
                body = (f'<video muted playsinline preload="metadata" data-last="{run["video_frames"] - 1}">'
                        f'<source src="{escape(link, quote=True)}" type="video/mp4"></video>'
                        f'<p><a href="{escape(link, quote=True)}">原始10FPS视频</a> · {escape(run["stop_reason"])}</p><table>{rows}</table>')
            cards.append(f'<article><h3>{escape(model)}</h3>{body}</article>')
        sections.append(f'<section><h2>{escape(case["case"])} · ΔX={case["dx_m"]:.4f}m / ΔY={case["dy_m"]:.4f}m</h2>'
                        '<div class="toolbar"><button data-action="play">统一播放</button><button data-action="pause">暂停</button>'
                        '<button data-action="reset">从头</button><label>倍率 <select data-action="speed">'
                        '<option>.5</option><option selected>1</option><option>2</option><option>4</option></select>×</label>'
                        '<input aria-label="共同控制步" type="range" min="0" max="0" value="0"><output>步0</output></div>'
                        f'<div class="columns">{"".join(cards)}</div></section>')
    css = """body{font:15px system-ui,sans-serif;background:#f4f5f7;color:#172434;margin:24px auto;max-width:1500px;padding:0 18px}
h1,h2,h3{line-height:1.3}section,header{background:white;border-radius:12px;padding:20px;margin-bottom:24px}video{width:100%;background:#111;aspect-ratio:1}
.columns{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:18px 0}
button,select{padding:7px 12px}input[type=range]{flex:1;min-width:120px}table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:6px;border-bottom:1px solid #ddd;text-align:left;overflow-wrap:anywhere}th{font-weight:500}.pending{min-height:240px;background:#f3f4f6;padding:18px}p{line-height:1.6}.note{color:#5a3520}@media(max-width:850px){.columns{grid-template-columns:1fr}}"""
    script = """const clocks=[];
document.querySelectorAll('section').forEach(section=>{
  const videos=[...section.querySelectorAll('video')],slider=section.querySelector('input'),label=section.querySelector('output');
  const max=Math.max(0,...videos.map(v=>Number(v.dataset.last)));
  slider.max=max;let step=0,playing=false,rate=1,origin=0,originStep=0;
  const seek=()=>{slider.value=step;label.textContent=`步${step} / ${max}`;
    videos.forEach(v=>{if(v.readyState){const t=Math.min(step,Number(v.dataset.last))/10;
      if(Math.abs(v.currentTime-t)>.015)v.currentTime=t;}});};
  const pause=()=>{playing=false;};
  const play=()=>{if(!max)return;if(step>=max)step=0;playing=true;origin=performance.now();originStep=step;};
  const reset=()=>{pause();step=0;seek();};
  const tick=now=>{if(playing){step=Math.min(max,originStep+Math.floor((now-origin)/1000*10*rate));seek();if(step>=max)pause();}requestAnimationFrame(tick);};
  section.querySelector('[data-action=play]').onclick=play;
  section.querySelector('[data-action=pause]').onclick=pause;
  section.querySelector('[data-action=reset]').onclick=reset;
  section.querySelector('select').onchange=e=>{rate=Number(e.target.value);origin=performance.now();originStep=step;};
  slider.oninput=()=>{pause();step=Number(slider.value);seek();};
  videos.forEach(v=>v.addEventListener('loadedmetadata',seek));
  clocks.push({play,pause,reset});seek();requestAnimationFrame(tick);
});
document.querySelectorAll('header [data-all]').forEach(button=>button.onclick=()=>clocks.forEach(c=>c[button.dataset.all]()));
document.addEventListener('visibilitychange',()=>{if(document.hidden)clocks.forEach(c=>c.pause());});"""
    limits = "".join(f'<li>{escape(limit)}</li>' for limit in summary["limits"])
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>位置对照 · {escape(summary["status"])}</title><style>{css}</style><header><h1>同位置 · 三模型仿真对照</h1>'
            f'<p>状态：{escape(summary["status"])}；已核验 {summary["completed_runs"]} / {len(summary["runs"])} 个运行。</p>'
            '<p>按控制步同步：原视频为初始画面＋每步一帧，10FPS回放，不代表墙钟或真实物理速度。短片到末尾保留最后真实画面，不循环。</p>'
            '<p class="note">视频仅在本机，通过相对路径读取；远端仓库不含原视频。请等待 RUNNING 格完成后重新生成本页。</p>'
            '<div class="toolbar"><button data-all="play">全部播放</button><button data-all="pause">全部暂停</button><button data-all="reset">全部从头</button></div>'
            f'<details><summary>指标及边界</summary><ul>{limits}</ul><p>表中诊断仅为代理；On不等于松爪撤手完成。动作范数为归一化OSC输入，位移为相邻新鲜观测毫米。</p></details></header>'
            + "".join(sections) + f'<script>{script}</script></html>')


def median(stats):
    return stats["p10_p50_p90"][1] if stats["n"] else None


def format_value(value):
    if value is None:
        return "—"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / ".local/position-study-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "evidence/position-study-v1")
    args = parser.parse_args()
    summary = build_summary(ROOT, args.study.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (args.output / "index.html").write_text(render_html(summary, ROOT, args.output.resolve()))
    print(json.dumps({k: summary[k] for k in ["status", "completed_runs", "pending_runs", "invalid_or_error_runs"]}))


if __name__ == "__main__":
    main()
