"""Bounded, sequential evaluation of the predeclared two-arm matrix."""
import argparse
import json
import socket
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
PLAN = json.loads((ROOT / "evidence/negative-recovery-20261019/experiment-plan.json").read_text())
OUT = ROOT / "evaluations/negative-recovery-20261019"
STATUS = None
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
    global STATUS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=list(PLAN["arms"]))
    parser.add_argument("--holding-only", action="store_true",
                        help="Rerun only archived invalid holding trials, with canonical controller initialization")
    args = parser.parse_args()
    suffix = "-holding-repair" if args.holding_only else ""
    STATUS = ROOT / f"evidence/negative-recovery-20261019/evaluation-{args.arm}{suffix}-progress.json"
    arm_output = OUT / args.arm
    arm_output.mkdir(parents=True, exist_ok=args.holding_only)
    progress = {"status": "RUNNING", "completed": [], "commands": []}
    save(progress)
    try:
        for arm, experiment in [(args.arm, PLAN["arms"][args.arm])]:
            checkpoint = ROOT / "checkpoints/pi05_handson_lora" / experiment / "1999"
            assert (checkpoint / "params/manifest.ocdbt").is_file(), str(checkpoint)
            for seed in PLAN["policy_seeds"]:
                log = ROOT / f"logs/negative-recovery-eval-{arm}-seed{seed}{suffix}-server.log"
                command = [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "scripts/serve_position_study.py"),
                           "--checkpoint", str(checkpoint), "--port", "8756", "--seed", str(seed)]
                progress["commands"].append(command)
                with socket.socket() as probe:
                    if probe.connect_ex(("127.0.0.1", 8756)) == 0:
                        raise RuntimeError("Diagnostic port8756 occupied; external process will not be stopped")
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
                    for spec in PLAN["evaluation_cases"]:
                        if args.holding_only and spec["role"] != "training_holding_state":
                            continue
                        case = spec["case"]
                        source = ROOT / spec["source_scene"]
                        import hashlib
                        assert hashlib.sha256((source/"scene.json").read_bytes()).hexdigest() == spec["scene_sha256"]
                        assert hashlib.sha256((source/"initial_state.npy").read_bytes()).hexdigest() == spec["initial_state_file_sha256"]
                        destination = arm_output / f"seed-{seed}" / case
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        command = [str(ROOT / ".venv-sim/bin/python"), "-u", str(ROOT / "scripts/replay_saved_scene.py"),
                                   "--source-scene", str(source),
                                   "--output", str(destination), "--checkpoint", str(checkpoint), "--port", "8756",
                                   "--max-steps", str(spec["max_steps"]), "--replan-steps", "1",
                                   "--stop-after-stable-release", "--post-release-steps", "50"]
                        if spec["role"] == "training_holding_state":
                            command.append("--initialize-saved-controller")
                        progress["active_trial"] = str(destination.relative_to(OUT))
                        progress["commands"].append(command)
                        save(progress)
                        with (ROOT / f"logs/negative-recovery-eval-{arm}-seed{seed}-{case}{suffix}.log").open("xb") as stream:
                            subprocess.run(command, cwd=ROOT, env=ENV, stdout=stream,
                                           stderr=subprocess.STDOUT, check=True, timeout=600)
                        result = json.loads((destination / "result.json").read_text())
                        if result.get("exception") is not None:
                            raise RuntimeError(f"Trial exception: {destination}")
                        progress["completed"].append({"run": str(destination.relative_to(OUT)),
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
