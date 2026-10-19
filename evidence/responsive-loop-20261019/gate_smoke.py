"""Real-physics completion-gate check using synthetic open/hold commands, not a VLA demo."""
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "openpi/third_party/libero"))
from generate_tracking_teacher import restore_controller  # noqa: E402 - local script bootstrap
from responsive_lab import ResponsiveSimulationLab  # noqa: E402
from simulation_lab import LiberoTask  # noqa: E402


class View:
    def __init__(self, env):
        self.paused, self.single_step, self.control_dt = True, False, env.env.control_timestep

    def is_running(self):
        return True

    def lock(self):
        return contextlib.nullcontext()

    def sync(self):
        pass

    def close(self):
        pass


class SettledTask:
    layout_count = 1

    def __init__(self):
        self.original = LiberoTask()
        self.metadata = self.original.metadata

    def create(self, layout):
        env, _ = self.original.create(24)
        source = ROOT / "evidence/live-speed-20261019/scene-002_init-024/final_state.npy"
        obs = env.regenerate_obs_from_state(np.load(source))
        restore_controller(env)
        self.source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        return env, obs


class SyntheticOpenHold:
    def infer(self, query):
        return {"actions": np.tile([0., 0, 0, 0, 0, 0, -1.], (10, 1))}, {
            "model_stage": "SYNTHETIC_OPEN_HOLD_NOT_LEARNED_POLICY"}

    def close(self):
        pass


task = SettledTask()
output = Path(__file__).resolve().parent / "synthetic-gate-smoke"
lab = ResponsiveSimulationLab(output, task, SyntheticOpenHold(), View)
try:
    lab.reset(0)
    lab.record.result.update(purpose="Synthetic real-physics gate smoke, not learned-policy success",
                             source_final_state_sha256=task.source_sha256)
    lab.toggle_run()
    deadline = time.monotonic() + 20
    while not lab.finished:
        lab.tick()
        assert time.monotonic() < deadline, "Completion gate did not stop"
        time.sleep(.001)
    result = lab.record.result
    assert result["stop_reason"] == "stable_release_complete"
    assert result["control_steps"] == 60 and lab.view.paused
    assert result["release_and_retreat_success"] is True
    print(json.dumps({"status": "PASS", "physics_control_steps": 60, "learned_model_calls": 0,
                      "paused": True, "stop_reason": result["stop_reason"]}), flush=True)
finally:
    lab.close()
