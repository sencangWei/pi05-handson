"""Bounded real-policy/physics/rendering scheduler comparison; no native viewer or Tk."""
import argparse
import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "verified"
CHECKPOINT = ROOT / "checkpoints/pi05_handson_lora/negative_quadrants_20261019/1999"
SOURCE = ROOT / "evidence/live-speed-20261019/scene-002_init-024"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "openpi/third_party/libero"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_arm(arm):
    import numpy as np
    from openpi_client import image_tools
    from simulation_lab import LiberoTask, PolicyConnection, SimulationLab, TASK_LANGUAGE
    from responsive_lab import ResponsiveSimulationLab

    class HeadlessView:
        def __init__(self, env):
            self.paused, self.single_step = True, False
            self.control_dt = env.env.control_timestep

        def is_running(self):
            return True

        def lock(self):
            return contextlib.nullcontext()

        def sync(self):
            pass

        def close(self):
            pass

    class MeasuredConnection(PolicyConnection):
        def __init__(self):
            super().__init__(8756)
            self.latencies = []

        def infer(self, query):
            started = time.monotonic()
            response = super().infer(query)
            self.latencies.append(time.monotonic() - started)
            return response

    client = MeasuredConnection()
    cls = ResponsiveSimulationLab if arm == "responsive" else SimulationLab
    lab = cls(OUT / arm, LiberoTask(), client, HeadlessView)
    scene = json.loads((SOURCE / "scene.json").read_text())
    lab.reset(scene["layout_id"], scene["bowl_edit"]["dx_m"], scene["bowl_edit"]["dy_m"])
    np.testing.assert_array_equal(lab.env.get_sim_state(), np.load(SOURCE / "initial_state.npy"))
    try:
        if arm == "warmup":
            query = {"prompt": TASK_LANGUAGE, "observation/state": np.zeros(8)}
            for key, camera in [("observation/image", "agentview_image"),
                                ("observation/wrist_image", "robot0_eye_in_hand_image")]:
                query[key] = image_tools.convert_to_uint8(image_tools.resize_with_pad(
                    np.ascontiguousarray(lab.obs[camera][::-1, ::-1]), 224, 224))
            for _ in range(2):
                client.infer(query)
            write(OUT / "warmup.json", {"inference_calls": 2, "physics_actions": 0})
            return
        if arm == "responsive":
            lab._check_completion = lambda action, done: None  # Fixed400steps, not a task-completion test.
        lab.record.result.update(benchmark="fixed400steps scheduler comparison, no native GUI",
                                 stop_policy="fixed400steps; auto-completion disabled only in this benchmark")
        metrics = {"regenerate": [], "physics_step": []}
        physical_states = []
        for name, key in [("regenerate_obs_from_state", "regenerate"), ("step", "physics_step")]:
            original = getattr(lab.env, name)

            def measured(*args, function=original, bucket=metrics[key], metric_key=key, **kwargs):
                started = time.monotonic()
                result = function(*args, **kwargs)
                if metric_key == "physics_step":
                    physical_states.append(lab.env.get_sim_state().copy())
                bucket.append(time.monotonic() - started)
                return result
            setattr(lab.env, name, measured)
        poll = .01 if arm == "responsive" else .03
        lab.toggle_run()
        started = time.monotonic()
        polls = 0
        while lab.record.result["control_steps"] < 400:
            lab.tick()
            polls += 1
            if lab.finished:
                raise RuntimeError("Unexpected early finish")
            if time.monotonic() - started > 240:
                raise TimeoutError(arm)
            if lab.record.result["control_steps"] % 50 == 0:
                write(OUT / "progress.json", {"arm": arm, "steps": lab.record.result["control_steps"]})
            time.sleep(poll)
        duration = time.monotonic() - started
        costs = {k: {"calls": len(v), "total_s": sum(v), "mean_s": float(np.mean(v))}
                 for k, v in metrics.items()}
        report = {"arm": arm, "control_steps": 400, "timed_loop_wall_s": duration,
                  "steps_per_second": 400 / duration, "poll_callbacks": polls,
                  "client_inference_calls": len(client.latencies),
                  "client_inference_total_s": sum(client.latencies),
                  "client_inference_mean_s": float(np.mean(client.latencies)),
                  "main_thread_operations": costs,
                  "initial_state_sha256": lab.record.result["initial_state_sha256"],
                  "model_metadata": lab.record.result["model_metadata"],
                  "scope": "Real policy, MuJoCo physics and EGL two-camera observations; manual poll cadence, no Tk widgets/native viewer/compositor. Warmup and final save/encoding excluded. Auto completion disabled for equal400step budget."}
        np.save(OUT / arm / "physics-states.npy", np.asarray(physical_states), allow_pickle=False)
        write(OUT / f"{arm}-metrics.json", report)
        lab.save("latency_benchmark_budget_400")
    finally:
        lab.close()


def orchestrate():
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", 8756)) != 0, "External port8756 owner is never stopped"
    env = os.environ.copy()
    env.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", HF_HUB_OFFLINE="1",
               LIBERO_CONFIG_PATH=str(ROOT / ".local/libero"), XLA_PYTHON_CLIENT_PREALLOCATE="false",
               XLA_PYTHON_CLIENT_MEM_FRACTION=".55", OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
    env.pop("JAX_PLATFORMS", None)
    OUT.mkdir(exist_ok=False)
    log = ROOT / "logs/responsive-loop-benchmark-v2-server.log"
    command = [str(ROOT / ".venv/bin/python"), "-u", "scripts/serve_position_study.py",
               "--checkpoint", str(CHECKPOINT), "--port", "8756", "--seed", "0"]
    with log.open("xb") as stream:
        server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    write(OUT / "launch.json", {"server_command": command, "server_pid": server.pid,
                              "source": str(SOURCE), "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "source_scene_sha256": hashlib.sha256((SOURCE / "scene.json").read_bytes()).hexdigest()})
    try:
        deadline = time.monotonic() + 300
        while "server listening on" not in log.read_text(errors="replace"):
            assert server.poll() is None, "Diagnostic server exited"
            if time.monotonic() >= deadline:
                raise TimeoutError("Diagnostic server startup")
            time.sleep(1)
        for arm in ["warmup", "legacy", "responsive"]:
            with (ROOT / f"logs/responsive-loop-benchmark-v2-{arm}.log").open("xb") as stream:
                subprocess.run([str(ROOT / ".venv-sim/bin/python"), "-u", str(Path(__file__).resolve()),
                                "--arm", arm], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                               check=True, timeout=300)
        import numpy as np
        paths = [next((OUT / arm).glob("scene-*/actions.csv")) for arm in ["legacy", "responsive"]]
        tables = [list(csv.DictReader(path.open())) for path in paths]
        columns = [f"action_{i}" for i in range(7)] + [f"state_{i}" for i in range(8)]
        arrays = [np.array([[float(row[c]) for c in columns] for row in table]) for table in tables]
        assert arrays[0].shape == arrays[1].shape == (400, 15)
        physics = [np.load(OUT / arm / "physics-states.npy") for arm in ["legacy", "responsive"]]
        assert physics[0].shape == physics[1].shape and physics[0].shape[0] == 400
        equal = np.array_equal(*arrays) and np.array_equal(*physics)
        a, b = [json.loads((OUT / f"{arm}-metrics.json").read_text()) for arm in ["legacy", "responsive"]]
        report = {"status": "PASS" if equal else "TRAJECTORY_MISMATCH", "identical_actions_and_positions": equal,
                  "maximum_absolute_difference": float(np.max(np.abs(arrays[0] - arrays[1]))),
                  "all400_physical_states_equal": bool(np.array_equal(*physics)),
                  "physical_state_max_abs_diff": float(np.max(np.abs(physics[0] - physics[1]))),
                  "comparison_quantity": "Raw action7 plus fresh pre-state8 and every full physical integration state; cached post-observable fields are not physical integration states.",
                  "legacy": a, "responsive": b,
                  "wall_time_reduction_fraction": 1 - b["timed_loop_wall_s"] / a["timed_loop_wall_s"],
                  "speedup": a["timed_loop_wall_s"] / b["timed_loop_wall_s"],
                  "limits": "One order, one saved state, one policy seed. This validates scheduler overhead, not learned action efficiency or end-to-end native GUI speed."}
        write(OUT / "comparison.json", report)
        print(json.dumps({k: report[k] for k in ["status", "identical_actions_and_positions", "speedup", "wall_time_reduction_fraction"]}), flush=True)
        if not equal:
            raise RuntimeError("Changed physical trajectory; do not deploy without investigation")
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["warmup", "legacy", "responsive"])
    args = parser.parse_args()
    run_arm(args.arm) if args.arm else orchestrate()
