"""Run the frozen intervention comparison on an owned, isolated policy server."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("parent", "candidate"), required=True)
    args = parser.parse_args()
    plan = json.loads((HERE / "plan.json").read_text())
    if args.arm == "parent":
        checkpoint = ROOT / plan["parent"]
    else:
        resolution = json.loads((HERE / "candidate-resolution.json").read_text())
        assert resolution["parent"] == plan["parent"]
        assert resolution["new_updates"] == plan["new_updates"] == 1000
        checkpoint = ROOT / resolution["checkpoint"]
    assert (checkpoint / "params/manifest.ocdbt").is_file()
    for case in plan["cases"]:
        source = ROOT / case["source_scene"]
        assert sha(source / "scene.json") == case["scene_sha256"]
        assert sha(source / "initial_state.npy") == case["initial_state_file_sha256"]
    output = ROOT / "evaluations/intervention-finetune-20261020" / args.arm
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.update(HF_HUB_OFFLINE="1", XLA_PYTHON_CLIENT_PREALLOCATE="false",
               XLA_PYTHON_CLIENT_MEM_FRACTION=".55", OMP_NUM_THREADS="4",
               TOKENIZERS_PARALLELISM="false", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
               LIBERO_CONFIG_PATH=str(ROOT / ".local/libero"))
    env.pop("JAX_PLATFORMS", None)
    progress = {"status": "RUNNING", "arm": args.arm, "checkpoint": str(checkpoint),
                "plan_sha256": sha(HERE / "plan.json"), "completed": [], "commands": []}

    def save():
        target = HERE / f"{args.arm}-progress.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(progress, indent=2) + "\n")
        temporary.replace(target)

    save()
    try:
        for seed in plan["policy_seeds"]:
            with socket.socket() as probe:
                assert probe.connect_ex(("127.0.0.1", 8756)) != 0, "Diagnostic port occupied"
            command = [str(ROOT / ".venv/bin/python"), "-u", "scripts/serve_position_study.py",
                       "--checkpoint", str(checkpoint), "--port", "8756", "--seed", str(seed)]
            server_log = ROOT / f"logs/intervention-20261020-{args.arm}-seed{seed}-server.log"
            with server_log.open("xb") as stream:
                server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            progress.update(active_server_pid=server.pid, active_seed=seed)
            progress["commands"].append(command)
            save()
            try:
                deadline = time.monotonic() + 300
                while "server listening on" not in server_log.read_text(errors="replace"):
                    if server.poll() is not None:
                        raise RuntimeError(f"Owned diagnostic server exited: {server_log}")
                    if time.monotonic() > deadline:
                        raise TimeoutError(str(server_log))
                    time.sleep(1)
                for case in plan["cases"]:
                    destination = output / f"seed-{seed}" / case["case"]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    command = [str(ROOT / ".venv-sim/bin/python"), "-u", "scripts/replay_saved_scene.py",
                               "--source-scene", str(ROOT / case["source_scene"]),
                               "--output", str(destination), "--checkpoint", str(checkpoint),
                               "--port", "8756", "--max-steps", str(case["max_steps"]),
                               "--replan-steps", "1", "--stop-after-stable-release",
                               "--post-release-steps", "50"]
                    if case["initialize_saved_controller"]:
                        command.append("--initialize-saved-controller")
                    progress["commands"].append(command)
                    progress["active_case"] = case["case"]
                    save()
                    log = ROOT / f"logs/intervention-20261020-{args.arm}-seed{seed}-{case['case']}.log"
                    with log.open("xb") as stream:
                        subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, check=True, timeout=300)
                    result = json.loads((destination / "result.json").read_text())
                    assert not result.get("exception"), result.get("exception")
                    row = {"case": case["case"], "seed": seed, "steps": result["control_steps"],
                           "first_release": result.get("first_ten_step_release_retreat_control_steps"),
                           "directory": str(destination.relative_to(ROOT))}
                    progress["completed"].append(row)
                    save()
                    print(json.dumps(row), flush=True)
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
        progress.pop("active_case", None)
        save()
    except BaseException as error:
        progress.update(status="FAILED", exception=repr(error))
        save()
        raise


if __name__ == "__main__":
    main()
