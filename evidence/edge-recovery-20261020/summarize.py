"""Audit matched rollout artifacts and report development results without promotion."""
import csv
import hashlib
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_negative_recovery import analyze  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan = json.loads((HERE / "plan.json").read_text())
    resolution = json.loads((HERE / "candidate-resolution.json").read_text())
    arms = {"parent": plan["parent"], "candidate": resolution["checkpoint"]}
    assert sha(ROOT / plan["parent"] / "params/manifest.ocdbt") == plan["parent_params_manifest_sha256"]
    assert sha(ROOT / resolution["checkpoint"] / "params/manifest.ocdbt") == resolution["params_manifest_sha256"]
    records, cards = [], []
    for arm, checkpoint in arms.items():
        for seed in plan["policy_seeds"]:
            for case in plan["cases"]:
                directory = ROOT / "evaluations/edge-recovery-20261020" / arm / f"seed-{seed}" / case["case"]
                source = ROOT / case["source_scene"]
                assert sha(source / "scene.json") == case["scene_sha256"]
                assert sha(source / "initial_state.npy") == case["initial_state_file_sha256"]
                assert (directory / "initial_state.npy").read_bytes() == (source / "initial_state.npy").read_bytes()
                result = json.loads((directory / "result.json").read_text())
                assert not result.get("exception") and result["video_frames"] == result["control_steps"] + 1
                metadata = result["model_metadata"]
                assert Path(metadata["checkpoint"]).resolve() == ROOT / checkpoint
                assert metadata["params_manifest_sha256"] == sha(ROOT / checkpoint / "params/manifest.ocdbt")
                assert metadata["diagnostic_rng_reset_per_connection"] == seed
                with (directory / "actions.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                metrics = analyze(rows, case["max_steps"])
                assert metrics["control_steps"] == result["control_steps"]
                records.append({"arm": arm, "seed": seed, "case": case["case"], "role": case["role"],
                                "directory": str(directory.relative_to(ROOT)), "metrics": metrics,
                                "wall_duration_s": result["wall_duration_s"],
                                "artifacts": {name: sha(directory / name) for name in
                                              ("actions.csv", "result.json", "initial_state.npy", "scene.json")}})
                video_files = list(directory.glob("*.mp4"))
                assert len(video_files) == 1
                target = HERE / "videos" / f"{arm}-{case['case']}{'-seed-1' if seed else ''}.mp4"
                target.parent.mkdir(exist_ok=True)
                if not target.exists():
                    target.write_bytes(video_files[0].read_bytes())
                assert target.read_bytes() == video_files[0].read_bytes()
                cards.append(f'<h3>{html.escape(arm)} / seed {seed} / {html.escape(case["case"])}</h3>'
                             f'<p>严格通过：{metrics["followup"]["strict_success"]}；控制步：{metrics["control_steps"]}</p>'
                             f'<video controls preload="metadata" width="768" src="videos/{target.name}"></video>')
    totals = {}
    for arm in arms:
        totals[arm] = {}
        for role in {case["role"] for case in plan["cases"]}:
            group = [r for r in records if r["arm"] == arm and r["role"] == role]
            totals[arm][role] = {"trials": len(group), "strict_successes": sum(
                r["metrics"]["followup"]["strict_success"] is True for r in group),
                "completed_followup_gate_failures": sum(r["metrics"]["followup"]["strict_success"] is False for r in group),
                "no_release_at_budget": sum(r["metrics"]["right_censored_without_release_at_budget"] for r in group),
                "release_followup_incomplete": sum(r["metrics"]["first_ten_step_release_control_step"] is not None and not r["metrics"]["followup"]["quantity_complete"] for r in group)}
    summary = {"status": "COMPLETE_DEVELOPMENT_COMPARISON", "promoted": False,
               "arms": arms, "totals": totals, "records": records, "limits": plan["limits"],
               "user_feedback": "Prior parent GUI Demo review: first four of five recordings completed placement; last negative-edge case reached 800steps without placement. New candidate not yet manually tested by user.",
               "interpretation": "Fresh-start development cases, not independent generalization; compare matched policy seeds."}
    (HERE / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (HERE / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>新增人工干预微调对比</title>'
        '<style>body{max-width:1100px;margin:32px auto;font:18px sans-serif;line-height:1.6}video{max-width:100%}</style>'
        '<h1>新增人工干预微调：训练前后对比</h1><p>布局0的1个负向边缘目标位置、4个已知开发对照，各2个推理种子。'
        '严格通过要求连续10步放置、松爪、撤手后再观察50步，碗漂移不超过5毫米。'
        '这些是开发诊断，不是独立泛化成功率；全部从保存的物理初始状态启动，不是从人工接管中途续跑。</p>'
        '<p><strong>请分别阅读：</strong>本轮全部案例从保存的初始布局启动；'
        '目标位置及对照均用于开发诊断。'
        '此前旧模型的五条用户录屏中，前四条完成放置，最后一条负向边缘位置达到800步仍失败；新模型结果以本页配对评测为准。</p>'
        '<p><strong>结论：</strong>目标位置严格通过1/2→2/2；总计7/10→8/10。仍有一条盘上闭爪停滞；另一条放置、撤手后出现一次闭爪指令，碗仍稳定在盘上，但未通过预先固定的持续开爪门限。配对成功的五条全部增加控制步，因此不推荐直接替代旧模型。</p><p><a href="review/index.html">新增人工干预片段</a> · <a href="summary.json">完整指标与校验</a></p><pre>'
        + html.escape(json.dumps(totals, ensure_ascii=False, indent=2)) + '</pre>' + ''.join(cards))
    print(json.dumps(totals), flush=True)


if __name__ == "__main__":
    main()
