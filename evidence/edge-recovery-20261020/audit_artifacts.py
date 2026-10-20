"""Verify archived rollout metrics without loading a model or running simulation."""
import csv
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
from summarize_negative_recovery import analyze  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan = json.loads((HERE / "plan.json").read_text())
    summary = json.loads((HERE / "summary.json").read_text())
    child = json.loads((HERE / "candidate-resolution.json").read_text())
    archive = json.loads((HERE / "artifact-archive.json").read_text())
    expected = {(arm, seed, case["case"]) for arm in ("parent", "candidate")
                for seed in plan["policy_seeds"] for case in plan["cases"]}
    actual = {(r["arm"], r["seed"], r["case"]) for r in summary["records"]}
    assert expected == actual and len(actual) == len(summary["records"]) == len(archive) == 20
    for case in plan["cases"]:
        source = HERE / "frozen-scenes" / case["case"]
        assert sha(source / "scene.json") == case["scene_sha256"]
        assert sha(source / "initial_state.npy") == case["initial_state_file_sha256"]
    for rec, stored in zip(summary["records"], archive, strict=True):
        directory = HERE / "rollouts" / rec["arm"] / f"seed-{rec['seed']}" / rec["case"]
        assert directory == HERE / stored["directory"]
        for name, digest in rec["artifacts"].items():
            assert sha(directory / name) == digest
        assert sha(HERE / stored["video"]) == stored["video_sha256"]
        case = next(c for c in plan["cases"] if c["case"] == rec["case"])
        assert sha(directory / "initial_state.npy") == case["initial_state_file_sha256"]
        with (directory / "actions.csv").open() as stream:
            metrics = analyze(list(csv.DictReader(stream)), case["max_steps"])
        assert metrics == rec["metrics"]
        result = json.loads((directory / "result.json").read_text())
        assert result.get("exception") is None
        assert result["video_frames"] == result["control_steps"] + 1 == metrics["control_steps"] + 1
        metadata = result["model_metadata"]
        checkpoint = summary["arms"][rec["arm"]]
        assert metadata["checkpoint"].endswith("/" + checkpoint)
        digest = plan["parent_params_manifest_sha256"] if rec["arm"] == "parent" else child["params_manifest_sha256"]
        assert metadata["params_manifest_sha256"] == digest
        assert metadata["diagnostic_rng_reset_per_connection"] == rec["seed"]
    counts = {arm: sum(r["metrics"]["followup"]["strict_success"] is True
                      for r in summary["records"] if r["arm"] == arm)
              for arm in summary["arms"]}
    assert counts == {"parent": 7, "candidate": 8}
    result = {"status": "PASS", "rollouts": 20, "videos": 20,
              "frozen_initial_scenes": 5, "strict_successes_of_ten": counts,
              "summary_sha256": sha(HERE / "summary.json"),
              "plan_sha256": sha(HERE / "plan.json")}
    (HERE / "audit-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
