"""Explicit human takeover and correction data preparation for the local lab."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from correction_data import CorrectionRecorder
from lab_workbench import LabWorkbench
from simulation_lab import TASK_LANGUAGE, TARGET, quaternion_axis_angle, state_digest


class CorrectionWorkbench(LabWorkbench):
    def __init__(self, *args, **kwargs):
        self.correction = None
        self.correction_status = "暂停模型后接管；只录制你执行的纠正动作"
        self.manual_grip = -1.0
        self.manual_state_hash = None
        super().__init__(*args, **kwargs)

    @property
    def correction_active(self):
        return self.correction is not None and self.correction.active

    def _require_no_correction(self):
        if self.correction_active:
            raise ValueError("请先保存正确示范或结束为草稿，再运行模型、更换场景或开始训练")

    @staticmethod
    def _state8(obs):
        state = np.concatenate((obs["robot0_eef_pos"], quaternion_axis_angle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"]))
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("Invalid robot observation")
        return state

    def camera_images(self):
        if self.obs is None:
            return None
        return {name: np.ascontiguousarray(self.obs[key][::-1, ::-1]) for name, key in (
            ("image", "agentview_image"), ("wrist_image", "robot0_eye_in_hand_image"))}

    def begin_correction(self, split="train"):
        self._ensure_not_training()
        self._require_no_correction()
        if split not in ("train", "holdout"):
            raise ValueError("请选择训练或留出示范")
        if self.env is None or self.view is None or not self.view.is_running():
            raise ValueError("请先加载场景；可在模型放置失败时暂停接管")
        self.save("human_takeover")
        self._new_connection()
        self.pending = None
        self.view.paused, self.view.single_step, self.step_requested = True, False, False
        with self.view.lock():
            state = self.env.get_sim_state().copy()
            self.obs = self.env.regenerate_obs_from_state(state)
        scene = self.record.result
        scenario = {key: scene.get(key) for key in ("layout_id", "bowl_edit", "task_metadata")}
        metadata = {"task": TASK_LANGUAGE, "split": split,
                    "scenario_key": hashlib.sha256(json.dumps(scenario, sort_keys=True).encode()).hexdigest(),
                    "source_scene": str(self.record.directory), "scenario": scenario,
                    "source_model_metadata": scene.get("model_metadata"),
                    "control_dt": float(self.view.control_dt),
                    "camera_orientation": "same double-axis flip as original course inference",
                    "action_contract": "OSC_POSE normalized delta xyz/rotation; +1 close, -1 open",
                    "initial_sim_state_sha256": state_digest(state),
                    "software_source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                               for name in ("correction_workbench.py", "correction_data.py", "lab_panel.py")}}
        directory = self.project / "corrections" / f"human_{time.time_ns()}"
        recorder = CorrectionRecorder(directory, metadata)
        np.save(directory / "initial_sim_state.npy", state, allow_pickle=False)
        self.correction = recorder
        self.manual_state_hash = state_digest(state)
        self.manual_grip = -1.0 if np.abs(self.obs["robot0_gripper_qpos"]).sum() >= .06 else 1.0
        self.correction_status = "已接管：小步调整位置，放稳后张开、撤手，再原地保持 10 步"
        self.event("correction_started", directory=str(directory), split=split, scenario_key=metadata["scenario_key"])

    def _manual_steps(self, action, repeats):
        self._ensure_not_training()
        if not self.correction_active or self.env is None or self.view is None or not self.view.is_running():
            raise ValueError("请先点击开始接管录制")
        if type(repeats) is not int or not 1 <= repeats <= 20:
            raise ValueError("每次执行步数须为 1～20")
        action = np.asarray(action, dtype=float)
        if action.shape != (7,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError("Invalid normalized manual action")
        self.view.paused, self.view.single_step, self.step_requested = True, False, False
        try:
            for _ in range(repeats):
                with self.view.lock():
                    before = self.env.get_sim_state().copy()
                    if state_digest(before) != self.manual_state_hash:
                        raise ValueError("录制期间物理状态被原生窗口修改；本段只保留草稿，请重新接管")
                    obs = self.env.regenerate_obs_from_state(before)
                    self.obs = obs
                    images = self.camera_images()
                    state8 = self._state8(obs)
                    after, _, done, _ = self.env.step(action.tolist())
                    self.correction.append(state8, action, images, self._state8(after),
                                           after[f"{TARGET}_pos"], bool(done))
                    self.obs = after
                    self.manual_state_hash = state_digest(self.env.get_sim_state())
                    self.known_state_hash = self.manual_state_hash
                self.view.handle.sync()
            self.correction_status = f"已记录 {self.correction.count} 个人工控制步；暂停等待下一次按钮操作"
            self.event("manual_action", action=action.tolist(), repeats=repeats, frames=self.correction.count)
        except Exception as error:
            if self.correction_active:
                self.correction.finish(False)
            self.correction_status = f"录制暂停并保留草稿：{error}"
            self.event("correction_error", error=str(error))
            raise

    def manual_jog(self, axis, sign, scale, repeats):
        if type(axis) is not int or axis not in range(6) or type(sign) is not int or sign not in (-1, 1):
            raise ValueError("请选择 XYZ 或 Rx/Ry/Rz 的正负方向")
        if isinstance(scale, bool) or not np.isfinite(scale) or not .05 <= scale <= .3:
            raise ValueError("动作幅度须为 0.05～0.3")
        action = np.zeros(7)
        action[axis], action[6] = sign * scale, self.manual_grip
        self._manual_steps(action, repeats)

    def manual_gripper(self, opened, repeats=10):
        if type(opened) is not bool:
            raise ValueError("请明确选择张开或闭合")
        action = np.zeros(7)
        action[6] = -1 if opened else 1
        self._manual_steps(action, repeats)
        self.manual_grip = float(action[6])

    def manual_hold(self, repeats=10):
        action = np.zeros(7)
        action[6] = self.manual_grip
        self._manual_steps(action, repeats)

    def finish_correction(self, approved):
        if not self.correction_active:
            raise ValueError("当前没有录制中的纠正示范")
        if type(approved) is not bool:
            raise ValueError("请明确选择合格或草稿")
        with self.view.lock():
            final_state = self.env.get_sim_state().copy()
            if approved and state_digest(final_state) != self.manual_state_hash:
                self.correction.finish(False)
                self.correction_status = "保存前检测到原生窗口修改物理状态，本段只保留草稿"
                self.event("correction_error", error=self.correction_status)
                raise ValueError(self.correction_status)
            self.obs = self.env.regenerate_obs_from_state(final_state)
            images = self.camera_images()
        directory = self.correction.finish(approved)
        np.save(directory / "final_sim_state.npy", final_state, allow_pickle=False)
        import imageio.v2 as imageio
        imageio.imwrite(directory / "final-agentview.png", images["image"])
        imageio.imwrite(directory / "final-wrist.png", images["wrist_image"])
        self.correction_status = "已保存并通过观察门；仍需人工检查动作过程" if approved else "已保留草稿；不会导入训练"
        self.event("correction_finished", directory=str(directory), approved=approved)
        return directory

    def correction_state(self):
        counts = {"train": 0, "holdout": 0}
        for path in (self.project / "corrections").glob("*/metadata.json"):
            try:
                data = json.loads(path.read_text())
                if data.get("status") == "approved" and data.get("split") in counts:
                    counts[data["split"]] += 1
            except (OSError, ValueError):
                pass
        return {"active": self.correction_active, "frames": self.correction.count if self.correction else 0,
                "directory": str(self.correction.directory) if self.correction else "",
                "approved_train": counts["train"], "approved_holdout": counts["holdout"],
                "status": self.correction_status}

    def corrections_bundles(self):
        found = []
        for path in (self.project / "evidence").glob("corrections-*-manifest.json"):
            try:
                if json.loads(path.read_text()).get("ready"):
                    found.append(str(path.resolve()))
            except (OSError, ValueError):
                pass
        return sorted(found, reverse=True)

    def prepare_corrections(self):
        self._require_no_correction()
        def prepare():
            import os
            env = os.environ.copy()
            env["JAX_PLATFORMS"] = "cpu"
            command = [str(self.project / ".venv/bin/python"), "-u", "scripts/prepare_corrections.py",
                       "--project-root", str(self.project), "--name", f"human_{time.time_ns()}"]
            result = subprocess.run(command, cwd=self.project, env=env, capture_output=True, text=True, timeout=600)
            log = self.project / "logs" / f"correction_prepare_{time.time_ns()}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(result.stdout + result.stderr)
            if result.returncode:
                raise ValueError(f"混合数据尚未生成：{(result.stderr or result.stdout)[-1200:]}；日志 {log}")
            return str(log)
        self._background("correction_prepare", prepare)
        self.correction_status = "正在校验并生成独立混合数据；训练尚未启动"

    def _collect_background(self):
        future = self.background.get("correction_prepare")
        if future is not None and future.done():
            del self.background["correction_prepare"]
            try:
                log = future.result()
                self.correction_status = f"混合数据已生成；到模型与训练页刷新数据列表。日志：{log}"
                self.event("corrections_prepared", log=log)
            except Exception as error:
                self.correction_status = str(error)
                self.event("corrections_prepare_error", error=str(error))
        super()._collect_background()

    def start_training(self, parent, steps, batch_size, learning_rate, uniform=False, correction_manifest=None):
        self._require_no_correction()
        if correction_manifest is None:
            return super().start_training(parent, steps, batch_size, learning_rate, uniform)
        status = self.jobs.snapshot()
        if status["training"]["running"] or (status["port_busy"] and not status["model"]["owned"]):
            raise ValueError("请先结束训练或在原终端退出外部模型服务")
        if not uniform:
            raise ValueError("纠正混合数据使用均匀采样，请选择均匀")
        manifest = Path(correction_manifest).resolve()
        if str(manifest) not in self.corrections_bundles():
            raise ValueError("请先准备并选择本项目的纠正数据版本")
        parent, steps, batch_size, learning_rate = self.jobs.validate_training(parent, steps, batch_size, learning_rate)
        self._archive_scene("start_correction_training")
        self.jobs.stop_model()
        job = self.jobs.start_correction_training(parent, steps, batch_size, learning_rate, manifest)
        self.job_state = self.jobs.snapshot()
        self.status = "纠正微调已由你启动；原数据和父 checkpoint 保留"
        self.event("correction_training_started", manifest=str(manifest), parent=str(parent), log=job["log"])

    def reset(self, *args, **kwargs):
        self._require_no_correction()
        return super().reset(*args, **kwargs)

    def start_model(self, checkpoint):
        self._require_no_correction()
        return super().start_model(checkpoint)

    def stop_model(self):
        self._require_no_correction()
        return super().stop_model()

    def toggle_run(self):
        self._require_no_correction()
        return super().toggle_run()

    def single_step(self):
        self._require_no_correction()
        return super().single_step()

    def save(self, reason="user_saved"):
        self._require_no_correction()
        return super().save(reason)

    def tick(self):
        if self.correction_active and self.view is not None and not self.view.is_running():
            self.finish_correction(False)
        super().tick()

    def state(self):
        return {**super().state(), "correction_active": self.correction_active}

    def close(self):
        if self.correction_active:
            self.finish_correction(False)
        super().close()
