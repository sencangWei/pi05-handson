"""Explicit human takeover and correction data preparation for the local lab."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from correction_data import MAX_FRAMES, CorrectionRecorder, digest, load_frames, write_json
from lab_workbench import LabWorkbench
from simulation_lab import TASK_LANGUAGE, TARGET, TrialRecord, quaternion_axis_angle, state_digest


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

    @staticmethod
    def _restore_controllers(env, frames, control_dt):
        for robot in getattr(env.env, "robots", []):
            substeps = int(env.env.control_timestep / env.env.model_timestep)
            if not np.isclose(control_dt, env.env.control_timestep) or substeps < 1:
                raise ValueError("草稿控制周期与当前环境不一致")
            low, high = np.full(2, -1.), np.full(2, 1.)
            for frame in frames:
                delta = np.array([-1., 1.]) * robot.gripper.speed * np.sign(frame["action"][6]) * substeps
                low, high = np.clip(low + delta, -1, 1), np.clip(high + delta, -1, 1)
            if not np.allclose(low, high, atol=1e-12, rtol=0):
                raise ValueError("草稿动作不足以确定夹爪累计控制状态；保留原场景")
            robot.controller.update(force=True)
            robot.controller.reset_goal()
            robot.gripper.current_action = low.copy()

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
                    "action_contract": "OSC_POSE normalized world delta xyz/axis-angle rotation; +1 close, -1 open",
                    "manual_motion_profile": {"normal_translation": .6, "precision_translation": .1,
                                              "ui_rotation": .1, "translation_limit": .8,
                                              "recording": "actual normalized actions unchanged; preset switch records no step"},
                    "initial_sim_state_sha256": state_digest(state),
                    "software_source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                               for name in ("correction_workbench.py", "correction_data.py", "lab_panel.py",
                                                            "live_preview.py", "viewer_controls.py", "manual_joystick.py")}}
        directory = self.project / "corrections" / f"human_{time.time_ns()}"
        recorder = CorrectionRecorder(directory, metadata)
        np.save(directory / "initial_sim_state.npy", state, allow_pickle=False)
        self.correction = recorder
        self.manual_state_hash = state_digest(state)
        self.manual_grip = -1.0 if np.abs(self.obs["robot0_gripper_qpos"]).sum() >= .06 else 1.0
        self.correction_status = "已接管：空中搬运用普通移动，接近抓放用精细移动；放稳后张开、撤手，再保持 10 步"
        self.event("correction_started", directory=str(directory), split=split, scenario_key=metadata["scenario_key"])

    def resume_correction(self, directory):
        """Continue a validated draft in a new directory, retaining its human action prefix."""
        self._ensure_not_training()
        self._require_no_correction()
        source = Path(directory).expanduser().resolve()
        if not source.is_relative_to(self.project / "corrections"):
            raise ValueError("请选择本项目 corrections 下的草稿目录")
        metadata = json.loads((source / "metadata.json").read_text())
        if metadata.get("status") != "draft" or metadata.get("task") != TASK_LANGUAGE:
            raise ValueError("仅能继续本任务的草稿，合格示范保持不变")
        frames = load_frames(source, metadata)
        if not 0 < len(frames) < MAX_FRAMES:
            raise ValueError("草稿须包含动作且尚未达到单段上限")
        state = np.load(source / "final_sim_state.npy", allow_pickle=False)
        initial = np.load(source / "initial_sim_state.npy", allow_pickle=False)
        if state.ndim != 1 or not np.isfinite(state).all():
            raise ValueError("草稿的最终仿真状态无效")
        if state_digest(initial) != metadata.get("initial_sim_state_sha256"):
            raise ValueError("草稿的初始仿真状态校验失败")
        if metadata.get("final_sim_state_sha256") not in (None, state_digest(state)):
            raise ValueError("草稿的最终仿真状态校验失败")
        scenario = metadata["scenario"]
        layout = scenario["layout_id"]
        if type(layout) is not int or not 0 <= layout < self.layout_count:
            raise ValueError("草稿布局编号无效")
        if scenario.get("task_metadata") != getattr(self.task, "metadata", {}):
            raise ValueError("草稿与当前任务资源不一致")
        candidate, _ = self.task.create(layout)
        try:
            if candidate.get_sim_state().shape != state.shape:
                raise ValueError("草稿仿真状态维度与当前环境不一致")
            obs = candidate.regenerate_obs_from_state(state)
            if (not np.allclose(candidate.get_sim_state(), state, atol=1e-12, rtol=0)
                    # Step observations may precede the final physics substep; allow 1 mm / 1 mrad.
                    or not np.allclose(self._state8(obs), frames[-1]["post_state"], atol=1e-3, rtol=0)
                    or not np.allclose(obs[f"{TARGET}_pos"], frames[-1]["post_bowl_pos"], atol=1e-3, rtol=0)):
                raise ValueError("草稿最终位置与最后记录帧不一致，保留原场景")
            grip = float(frames[-1]["action"][6])
            self._restore_controllers(candidate, frames, metadata["control_dt"])
            target = self.project / "corrections" / f"human_{time.time_ns()}"
            recorder = CorrectionRecorder(target, metadata)
            for item in metadata["files"]:
                shutil.copyfile(source / item["path"], target / item["path"])
                if digest(target / item["path"]) != item["sha256"]:
                    raise ValueError("续录副本校验失败；原草稿保留")
            shutil.copyfile(source / "initial_sim_state.npy", target / "initial_sim_state.npy")
            recorder.count = len(frames)
            recorder.metadata.update(frame_count=recorder.count, files=metadata["files"],
                                     observation_gates=None, final_sim_state_sha256=None,
                                     resume_provenance={"source_directory": str(source),
                                         "source_metadata_sha256": digest(source / "metadata.json"),
                                         "inherited_frames": recorder.count, "state_sha256": state_digest(state),
                                         "recorded_observation_max_abs_diff": float(np.max(np.abs(self._state8(obs) - frames[-1]["post_state"]))),
                                         "controller_state": "goals reset at restored pose; gripper accumulator reconstructed from prefix",
                                         "recording_software_source_sha256": {
                                             name: digest(Path(__file__).parent / name)
                                             for name in ("correction_workbench.py", "correction_data.py", "lab_panel.py",
                                                          "live_preview.py", "viewer_controls.py", "manual_joystick.py")},
                                         "bit_exact_controller_resume": False})
            write_json(target / "metadata.json", recorder.metadata)
            np.save(target / "resume_sim_state.npy", state, allow_pickle=False)
            np.save(target / "final_sim_state.npy", state, allow_pickle=False)
            scene = {**scenario, "task_language": TASK_LANGUAGE, "formal_benchmark": False,
                     "interactive_experiment": True, "physics_horizon_ignored": True,
                     "resumed_human_draft": str(source)}
            record = TrialRecord(self.output / f"scene-{self.scene_number + 1:03d}_init-{layout:03d}_resume-{time.time_ns()}",
                                 scene, obs, state)
            record.save("human_draft_resume", state, obs)
            view = self.viewer_factory(candidate)
            try:
                self.save("resume_correction")
            except Exception:
                view.close()
                raise
        except Exception:
            candidate.close()
            raise
        self._close_scene()
        self._new_connection()
        self.pending = None
        self.env, self.obs, self.layout_id = candidate, obs, layout
        self.scene_number += 1
        self.record, self.view = record, view
        self.view.paused, self.view.single_step, self.step_requested = True, False, False
        self.finished = True
        self.correction = recorder
        self.manual_grip = grip
        self.manual_state_hash = self.known_state_hash = state_digest(state)
        self.status = "已恢复草稿场景并暂停；人工按钮继续录制"
        self.correction_status = f"已恢复 {recorder.count} 步，可继续；原草稿保留，尚未保存为合格"
        self.event("correction_resumed", directory=str(target), **recorder.metadata["resume_provenance"])

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
                if self.correction.count >= MAX_FRAMES:
                    raise ValueError(f"单段人工纠正已达 {MAX_FRAMES} 步上限，本段保留草稿；请开启新段")
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
                self.view.sync()
            self.correction_status = f"已记录 {self.correction.count} 个人工控制步；暂停等待下一次按钮操作"
            self.event("manual_action", action=action.tolist(), repeats=repeats, frames=self.correction.count)
        except Exception as error:
            if self.correction_active:
                try:
                    self.finish_correction(False)
                except Exception as snapshot_error:
                    if self.correction_active:
                        self.correction.finish(False)
                    self.event("correction_snapshot_error", error=str(snapshot_error))
            self.correction_status = f"录制暂停并保留草稿：{error}"
            self.event("correction_error", error=str(error))
            raise

    def manual_jog(self, axis, sign, scale, repeats):
        if type(axis) is not int or axis not in range(6) or type(sign) is not int or sign not in (-1, 1):
            raise ValueError("请选择 XYZ 或 Rx/Ry/Rz 的正负方向")
        limit = .8 if axis < 3 else .3
        if isinstance(scale, bool) or not np.isfinite(scale) or not .05 <= scale <= limit:
            raise ValueError(f"平移幅度须为 0.05～0.8，旋转幅度须为 0.05～0.3（本次上限 {limit}）")
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

    def manual_translate(self, x, y, scale):
        """Record one view-relative table-plane step; Z and orientation stay fixed."""
        if (not np.isfinite([x, y, scale]).all() or isinstance(scale, bool)
                or not .05 <= scale <= .8 or np.linalg.norm([x, y]) > 1 + 1e-6):
            raise ValueError("摇杆幅度须在单位圆内，平移幅度须为 0.05～0.8")
        if not self.correction_active or self.view is None or not self.view.is_running():
            raise ValueError("请先开始人工接管")
        right, up = self.view.translation_basis()
        delta = x * right + y * up
        delta /= max(1., float(np.linalg.norm(delta)))
        action = np.zeros(7)
        action[:2], action[6] = delta * scale, self.manual_grip
        self._manual_steps(action, 1)

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
        self.correction.metadata["final_sim_state_sha256"] = state_digest(final_state)
        write_json(directory / "metadata.json", self.correction.metadata)
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

    def prepare_corrections(self, base_manifest=None):
        self._require_no_correction()
        base = None
        selection = None
        if base_manifest is not None:
            selected = Path(base_manifest).resolve()
            bundles = self.corrections_bundles()
            if str(selected) not in bundles or selected.parent != (self.project / "evidence").resolve():
                raise ValueError("请选择本项目有效的公开扩充或混合训练数据版本")
            manifest = json.loads(selected.read_text())
            if manifest.get("training_sampling_profile"):
                raise ValueError("恢复重点版本已复用现有数据，无需重建；新增示范请从原始混合数据版本另行筛选")
            if manifest.get("dataset_kind") == "expanded_public_v1":
                base = selected
            elif manifest.get("dataset_kind") == "expanded_public_corrections_v1":
                relative = manifest.get("public_base_manifest")
                if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
                    raise ValueError("公开扩充基座路径无效")
                base = (self.project / relative).resolve()
                if "curation_selection" in manifest:
                    relative_selection = manifest["curation_selection"]
                    if (not isinstance(relative_selection, str) or Path(relative_selection).is_absolute()
                            or ".." in Path(relative_selection).parts):
                        raise ValueError("精选片段清单路径无效")
                    selection = (self.project / relative_selection).resolve()
                    if selection.parent != (self.project / "evidence").resolve() or not selection.is_file():
                        raise ValueError("精选片段清单缺失或不属于本项目")
            else:
                raise ValueError("所选版本没有公开扩充基座")
            if (str(base) not in bundles or base.parent != (self.project / "evidence").resolve()
                    or json.loads(base.read_text()).get("dataset_kind") != "expanded_public_v1"):
                raise ValueError("公开扩充基座缺失或不属于本项目")
        def prepare():
            import os
            env = os.environ.copy()
            env["JAX_PLATFORMS"] = "cpu"
            command = [str(self.project / ".venv/bin/python"), "-u", "scripts/prepare_corrections.py",
                       "--project-root", str(self.project), "--name", f"human_{time.time_ns()}"]
            if base is not None:
                command.extend(["--base-manifest", str(base)])
            if selection is not None:
                command.extend(["--selection", str(selection)])
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
            raise ValueError("所选数据版本使用均匀采样，请选择均匀")
        manifest = Path(correction_manifest).resolve()
        if str(manifest) not in self.corrections_bundles():
            raise ValueError("请先准备并选择本项目的训练数据版本")
        dataset_kind = json.loads(manifest.read_text()).get("dataset_kind", "corrections")
        parent, steps, batch_size, learning_rate = self.jobs.validate_training(parent, steps, batch_size, learning_rate)
        self._archive_scene("start_correction_training")
        self.jobs.stop_model()
        job = self.jobs.start_correction_training(parent, steps, batch_size, learning_rate, manifest)
        self.job_state = self.jobs.snapshot()
        self.status = {
            "expanded_public_v1": "公开示范扩充微调已由你启动；人工纠正未混入，原数据和父 checkpoint 保留",
            "expanded_public_corrections_v1": "公开示范与人工纠正混合微调已由你启动；原数据和父 checkpoint 保留",
        }.get(dataset_kind, "纠正微调已由你启动；原数据和父 checkpoint 保留")
        self.event("correction_training_started", manifest=str(manifest), parent=str(parent), log=job["log"],
                   dataset_kind=dataset_kind)

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
