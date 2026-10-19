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
    records, cards = [], []
    for arm, checkpoint in arms.items():
        for seed in plan["policy_seeds"]:
            for case in plan["cases"]:
                directory = ROOT / "evaluations/intervention-finetune-20261020" / arm / f"seed-{seed}" / case["case"]
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
                if seed == 0:
                    video_files = list(directory.glob("*.mp4"))
                    assert len(video_files) == 1
                    target = HERE / "videos" / f"{arm}-{case['case']}.mp4"
                    target.parent.mkdir(exist_ok=True)
                    if not target.exists():
                        target.write_bytes(video_files[0].read_bytes())
                    assert target.read_bytes() == video_files[0].read_bytes()
                    cards.append(f'<h3>{html.escape(arm)} / {html.escape(case["case"])}</h3>'
                                 f'<p>严格通过：{metrics["followup"]["strict_success"]}；控制步：{metrics["control_steps"]}</p>'
                                 f'<video controls preload="metadata" width="768" src="videos/{target.name}"></video>')
    totals = {}
    for arm in arms:
        totals[arm] = {}
        for role in {case["role"] for case in plan["cases"]}:
            group = [r for r in records if r["arm"] == arm and r["role"] == role]
            totals[arm][role] = {"trials": len(group), "strict_successes": sum(
                r["metrics"]["followup"]["strict_success"] is True for r in group)}
    summary = {"status": "COMPLETE_DEVELOPMENT_COMPARISON", "promoted": False,
               "arms": arms, "totals": totals, "records": records, "limits": plan["limits"]}
    (HERE / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (HERE / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>新增人工干预微调对比</title>'
        '<style>body{max-width:1100px;margin:32px auto;font:18px sans-serif;line-height:1.6}video{max-width:100%}</style>'
        '<h1>新增人工干预微调：训练前后对比</h1><p>4个训练接管现场、2个旧开发场景，各2个种子。'
        '严格通过要求连续10步放置、松爪、撤手后再观察50步，碗漂移不超过5毫米。'
        '这些是开发诊断，不是独立泛化成功率。控制器隐状态重建也不等于原GUI逐位续跑。</p>'
        '<p><a href="review/index.html">新增人工干预片段</a> · <a href="summary.json">完整指标与校验</a></p><pre>'
        + html.escape(json.dumps(totals, ensure_ascii=False, indent=2)) + '</pre>' + ''.join(cards))
    print(json.dumps(totals), flush=True)


if __name__ == "__main__":
    main()
