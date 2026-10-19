"""Run the frozen public-only/current model comparison without touching live8755."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path(__file__).resolve().parent
OUTPUT = ROOT / "evaluations/precorrection-comparison-20261019"
PLAN = json.loads((EVIDENCE / "plan.json").read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(value):
    path = EVIDENCE / "progress.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main():
    for name, expected in PLAN["source_sha256"].items():
        assert sha(ROOT / "scripts" / name) == expected, name
    for arm, checkpoint in PLAN["arms"].items():
        assert sha(ROOT / checkpoint / "params/manifest.ocdbt") == PLAN["params_manifest_sha256"][arm]
    for spec in PLAN["evaluation_cases"]:
        source = ROOT / spec["source_scene"]
        assert sha(source / "scene.json") == spec["scene_sha256"]
        assert sha(source / "initial_state.npy") == spec["initial_state_file_sha256"]
    OUTPUT.mkdir(exist_ok=False)
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", XLA_PYTHON_CLIENT_PREALLOCATE="false",
               XLA_PYTHON_CLIENT_MEM_FRACTION=".55", OMP_NUM_THREADS="4",
               TOKENIZERS_PARALLELISM="false", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
               LIBERO_CONFIG_PATH=str(ROOT / ".local/libero"))
    env.pop("JAX_PLATFORMS", None)
    progress = {"status": "RUNNING", "plan_sha256": sha(EVIDENCE / "plan.json"),
                "runner_sha256": sha(Path(__file__)), "commands": [], "completed": []}
    write(progress)
    try:
        for arm, checkpoint in PLAN["arms"].items():
            for seed in PLAN["policy_seeds"]:
                with socket.socket() as probe:
                    if probe.connect_ex(("127.0.0.1", 8756)) == 0:
                        raise RuntimeError("8756 occupied; external owner is not stopped")
                command = [str(ROOT / ".venv/bin/python"), "-u", "scripts/serve_position_study.py",
                           "--checkpoint", str(ROOT / checkpoint), "--port", "8756", "--seed", str(seed)]
                log = ROOT / f"logs/precorrection-comparison-{arm}-seed{seed}-server.log"
                with log.open("xb") as stream:
                    server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                progress.update(active_server_pid=server.pid, active_arm=arm, active_seed=seed)
                progress["commands"].append(command)
                write(progress)
                try:
                    deadline = time.monotonic()+600
                    while "server listening on" not in log.read_text(errors="replace"):
                        if server.poll() is not None:
                            raise RuntimeError(f"Diagnostic server exited: {log}")
                        if time.monotonic() > deadline:
                            raise TimeoutError(str(log))
                        time.sleep(1)
                    for spec in PLAN["evaluation_cases"]:
                        destination = OUTPUT / arm / f"seed-{seed}" / spec["case"]
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        command = [str(ROOT / ".venv-sim/bin/python"), "-u", "scripts/replay_saved_scene.py",
                                   "--source-scene", str(ROOT / spec["source_scene"]),
                                   "--output", str(destination), "--checkpoint", str(ROOT / checkpoint),
                                   "--port", "8756", "--max-steps", str(spec["max_steps"]),
                                   "--replan-steps", "1", "--stop-after-stable-release", "--post-release-steps", "50"]
                        progress["commands"].append(command)
                        progress["active_trial"] = str(destination.relative_to(OUTPUT))
                        write(progress)
                        with (ROOT / f"logs/precorrection-comparison-{arm}-seed{seed}-{spec['case']}.log").open("xb") as stream:
                            subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                           stderr=subprocess.STDOUT, check=True, timeout=600)
                        result = json.loads((destination / "result.json").read_text())
                        if result.get("exception"):
                            raise RuntimeError(result["exception"])
                        completed = {"trial": progress["active_trial"], "steps": result["control_steps"],
                                     "first_release": result.get("first_ten_step_release_retreat_control_steps")}
                        progress["completed"].append(completed)
                        write(progress)
                        print(json.dumps(completed), flush=True)
                finally:
                    if server.poll() is None:
                        server.terminate()
                        try:
                            server.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            server.kill()
                            server.wait(timeout=5)
                progress.pop("active_server_pid", None)
        progress["status"] = "COMPLETE"
        progress.pop("active_trial", None)
        write(progress)
    except BaseException as error:
        progress.update(status="FAILED", exception=repr(error))
        write(progress)
        raise


if __name__ == "__main__":
    main()
