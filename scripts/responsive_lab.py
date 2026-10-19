"""Responsive interactive loop; the teacher-pinned legacy module stays unchanged."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np

from simulation_lab import TARGET, SimulationLab, state_digest, write_json


class ResponsiveSimulationLab(SimulationLab):
    def __init__(self, *args, **kwargs):
        self._request_started = None
        self._request_timing = {}
        self._timings = []
        self._timing_record = None
        self._last_action = None
        self._stable_count, self._stable_bowl = 0, None
        super().__init__(*args, **kwargs)
        path = self.output / "session.json"
        session = json.loads(path.read_text())
        session["source_sha256"]["responsive_lab.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        session["control_loop"] = "responsive_v1"
        write_json(path, session)

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self._timings, self._last_action = [], None
        self._timing_record = self.record
        self._stable_count, self._stable_bowl = 0, None
        self.record.result.update(
            control_loop="responsive_v1", timing_file="timing.csv",
            stop_policy="60 consecutive stable released/retreated steps, max800, or user save",
            stable_completion_definition="On, open command, gap>=.06m, EEF-bowl>=.10m for60 consecutive steps, bowl drift<=.005m",
        )
        scene_path = self.record.directory / "scene.json"
        if scene_path.exists():
            scene = json.loads(scene_path.read_text())
            for key in ("control_loop", "timing_file", "stop_policy", "stable_completion_definition"):
                scene[key] = self.record.result[key]
            write_json(scene_path, scene)

    def _submit(self, query, request_hash):
        self._request_started = time.monotonic()
        timing = self._request_timing = {"started": self._request_started}
        super()._submit(query, request_hash)

        def completed(future):
            timing["received"] = time.monotonic()
            if not future.cancelled() and future.exception() is None:
                response, _ = future.result()
                timing["server_infer_ms"] = response.get("server_timing", {}).get("infer_ms")
        self.pending[0].add_done_callback(completed)

    def _can_skip_render(self):
        if self.closed or self.view is None or self.finished or not self.view.is_running():
            return False
        if self.view.paused != self.was_paused or self.view.single_step:
            return False
        with self.view.lock():
            if state_digest(self.env.get_sim_state()) != self.known_state_hash:
                return False
        waiting = self.pending is not None and not self.pending[0].done()
        idle = self.pending is None and ((self.view.paused and not self.step_requested)
                                        or time.monotonic() < self.next_step_time)
        return waiting or idle

    def tick(self):
        if self._can_skip_render():
            self.view.sync()
            return
        started = time.monotonic()
        before = self.record.result["control_steps"] if self.record else 0
        pending = self.pending
        pre_eef = np.array(self.obs["robot0_eef_pos"]) if self.obs is not None else None
        super().tick()
        if self.record is None or self.record.result["control_steps"] == before:
            return
        # Pace the entire request/step cycle, rather than sleeping again after inference.
        self.next_step_time = (self._request_started if self._request_started is not None else started) + self.view.control_dt
        response, _ = pending[0].result()
        action = np.asarray(response["actions"])[0]
        ended = time.monotonic()
        timing = self._request_timing
        received = timing.get("received")
        row = {
            "control_step": self.record.result["control_steps"],
            "action_interval_s": ended - self._last_action[0]
            if self._last_action is not None and self._last_action[1] == self.generation else None,
            "client_inference_s": received - timing["started"] if received is not None else None,
            "server_infer_ms": timing.get("server_infer_ms"),
            "ready_to_step_s": max(0., started - received) if received is not None else None,
            "apply_tick_s": ended - started,
            "eef_move_mm": float(np.linalg.norm(self.obs["robot0_eef_pos"] - pre_eef) * 1000),
        }
        path = self.record.directory / "timing.csv"
        exists = path.exists()
        with path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        self._timings.append(row)
        self._last_action = ended, self.generation
        self._check_completion(action, bool(self.record.result.get("final_benchmark_success")))
        if self.record.saved:
            self._store_timing_summary()
            write_json(self.record.directory / "result.json", self.record.result)

    def _check_completion(self, action, done):
        if getattr(self, "correction_active", False):
            return
        bowl = np.asarray(self.obs[TARGET + "_pos"])
        gap = abs(float(self.obs["robot0_gripper_qpos"][0] - self.obs["robot0_gripper_qpos"][1]))
        released = done and action[6] < 0 and gap >= .06 and np.linalg.norm(self.obs["robot0_eef_pos"] - bowl) >= .10
        if not released:
            self._stable_count, self._stable_bowl = 0, None
            return
        if self._stable_bowl is None or np.linalg.norm(bowl - self._stable_bowl) > .005:
            self._stable_count, self._stable_bowl = 0, bowl.copy()
        self._stable_count += 1
        if self._stable_count == 60:
            # LIBERO post-step observables may be cached; confirm current geometry once.
            with self.view.lock():
                fresh = self.env.regenerate_obs_from_state(self.env.get_sim_state())
            fresh_gap = abs(float(fresh["robot0_gripper_qpos"][0] - fresh["robot0_gripper_qpos"][1]))
            if (fresh_gap < .06 or np.linalg.norm(fresh["robot0_eef_pos"] - fresh[TARGET + "_pos"]) < .10
                    or np.linalg.norm(fresh[TARGET + "_pos"] - self._stable_bowl) > .005):
                self._stable_count, self._stable_bowl = 0, None
                return
            self.obs = fresh
            self.record.result.update(release_and_retreat_success=True, stable_completion_steps=60)
            self.save("stable_release_complete")
            self.status = "已确认放稳、松爪及撤手60步，已暂停并保存"

    def _store_timing_summary(self):
        if self.record is None:
            return
        rows = self._timings if self.record is self._timing_record else []
        means = {}
        for key in ("action_interval_s", "client_inference_s", "server_infer_ms", "ready_to_step_s", "apply_tick_s", "eef_move_mm"):
            values = [row[key] for row in rows if row[key] is not None]
            means[key] = statistics.mean(values) if values else None
        self.record.result["timing_summary"] = {"steps": len(rows), "means": means,
            "scope": "Client RPC includes network/server work; apply_tick includes observations, physics and recording. Intervals can include user single-step delays."}

    def save(self, reason="user_saved"):
        self._store_timing_summary()
        return super().save(reason)

    def state(self):
        state = super().state()
        state["waiting"] = self.pending is not None and not self.pending[0].done() and not self.finished
        recent = self._timings[-20:] if self.record is self._timing_record else []
        intervals = [row["action_interval_s"] for row in recent if row["action_interval_s"] is not None and row["action_interval_s"] > 0]
        infer = [row["client_inference_s"] for row in recent if row["client_inference_s"] is not None]
        if intervals:
            text = f"最近步频：{1 / statistics.mean(intervals):.1f}步/秒"
            if infer:
                text += f"；推理等待：{statistics.mean(infer) * 1000:.0f}ms"
            text += f"；末端移动：{statistics.median(row['eef_move_mm'] for row in recent):.2f}mm/步"
            state["performance_text"] = text
        return state
