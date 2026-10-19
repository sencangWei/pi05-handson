"""Bounded, sequential evaluation of the predeclared two-arm matrix."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
PLAN = json.loads((ROOT / "evidence/retrain-20261019/experiment-plan.json").read_text())
OUT = ROOT / "evaluations/retrain-20261019"
STATUS = ROOT / "evidence/retrain-20261019/evaluation-progress.json"
ENV = os.environ.copy()
ENV.update(HF_HUB_OFFLINE="1", XLA_PYTHON_CLIENT_PREALLOCATE="false",
           XLA_PYTHON_CLIENT_MEM_FRACTION=".55", OMP_NUM_THREADS="4",
           TOKENIZERS_PARALLELISM="false", MUJOCO_GL="egl",
           LIBERO_CONFIG_PATH=str(ROOT / ".local/libero"))
ENV.pop("JAX_PLATFORMS", None)


def save(record):
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    tmp.replace(STATUS)


def main():
    assert not OUT.exists(), "Refuse to overwrite previous evaluations"
    OUT.mkdir(parents=True)
    progress = {"status": "RUNNING", "completed": [], "commands": []}
    save(progress)
    try:
        for arm, experiment in PLAN["arms"].items():
            checkpoint = ROOT / "checkpoints/pi05_handson_lora" / experiment / "1999"
            assert (checkpoint / "params/manifest.ocdbt").is_file(), str(checkpoint)
            for seed in PLAN["followup_evaluation"]["policy_seeds"]:
                log = ROOT / f"logs/retrain-eval-{arm}-seed{seed}-server.log"
                command = [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "scripts/serve_position_study.py"),
                           "--checkpoint", str(checkpoint), "--port", "8756", "--seed", str(seed)]
                progress["commands"].append(command)
                with log.open("xb") as stream:
                    server = subprocess.Popen(command, cwd=ROOT, env=ENV, stdout=stream, stderr=subprocess.STDOUT)
                progress["active_server"] = {"pid": server.pid, "arm": arm, "seed": seed, "log": str(log)}
                save(progress)
                try:
                    deadline = time.monotonic() + 600
                    while "server listening on" not in log.read_text(errors="replace"):
                        if server.poll() is not None:
                            raise RuntimeError(f"Server exited {server.returncode}: {log}")
                        if time.monotonic() >= deadline:
                            raise TimeoutError(str(log))
                        time.sleep(2)
                    for case in PLAN["followup_evaluation"]["physical_cases"]:
                        destination = OUT / f"{arm}__seed{seed}__{case}"
                        command = [str(ROOT / ".venv-sim/bin/python"), "-u", str(ROOT / "scripts/replay_saved_scene.py"),
                                   "--source-scene", str(ROOT / ".local/position-study-v1/sources" / case),
                                   "--output", str(destination), "--checkpoint", str(checkpoint), "--port", "8756",
                                   "--max-steps", "800", "--replan-steps", "1",
                                   "--stop-after-stable-release", "--post-release-steps", "50"]
                        progress["active_trial"] = destination.name
                        progress["commands"].append(command)
                        save(progress)
                        with (ROOT / f"logs/retrain-eval-{destination.name}.log").open("xb") as stream:
                            subprocess.run(command, cwd=ROOT, env=ENV, stdout=stream,
                                           stderr=subprocess.STDOUT, check=True, timeout=600)
                        result = json.loads((destination / "result.json").read_text())
                        if result.get("exception") is not None:
                            raise RuntimeError(f"Trial exception: {destination}")
                        progress["completed"].append({"run": destination.name,
                                                      "control_steps": result["control_steps"],
                                                      "first_release": result.get("first_ten_step_release_retreat_control_steps")})
                        save(progress)
                finally:
                    if server.poll() is None:
                        server.terminate()
                        try:
                            server.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            server.kill()
                            server.wait(timeout=5)
                progress.pop("active_server", None)
        progress.pop("active_trial", None)
        progress["status"] = "COMPLETE"
        save(progress)
    except BaseException as exc:
        progress["status"] = "FAILED"
        progress["exception"] = repr(exc)
        save(progress)
        raise


if __name__ == "__main__":
    main()
