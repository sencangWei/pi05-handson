"""Interactive LIBERO experiments: native control panel plus MuJoCo viewer."""
import argparse
from concurrent.futures import Future
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

from evaluate_libero import (
    DUMMY_ACTION, TASK_LANGUAGE, prepare_output, quaternion_axis_angle,
    update_benchmark_progress, write_json,
)

ROOT = Path(__file__).resolve().parents[1]
TARGET = "akita_black_bowl_1"
INTERACTIVE_STEP_LIMIT = 800


def state_digest(state):
    import numpy as np
    return hashlib.sha256(np.asarray(state).tobytes()).hexdigest()


def bowl_penetrations(env, tolerance_m=0.001):
    """Return deep external contacts after forward; normal resting contacts are tolerated."""
    target_geoms = set(env.env.objects_dict[TARGET].contact_geoms)
    result = []
    for contact in env.sim.data.contact[:env.sim.data.ncon]:
        names = [env.sim.model.geom_id2name(int(g)) for g in (contact.geom1, contact.geom2)]
        is_target = [name in target_geoms for name in names]
        if sum(is_target) == 1 and float(contact.dist) < -tolerance_m:
            result.append({"other_geom": names[0] if not is_target[0] else names[1],
                           "depth_m": -float(contact.dist)})
    return result


def move_bowl(env, obs, dx, dy):
    """Edit a freshly reset scene; preserve height and rotation and refresh camera observations."""
    import numpy as np
    if not all(math.isfinite(v) and abs(v) <= 0.1 for v in (dx, dy)):
        raise ValueError("碗的位置偏移需在 -10 到 10 cm 范围内")
    before = np.asarray(obs[f"{TARGET}_pos"]).copy()
    after = before + [dx, dy, 0]
    if np.linalg.norm(obs["robot0_gripper_qpos"]) < 0.04 or np.linalg.norm(after - obs["robot0_eef_pos"]) < 0.12:
        raise ValueError("请在机械臂开爪、远离碗的初始布局中调整位置")
    joints = env.env.objects_dict[TARGET].joints
    if len(joints) != 1:
        raise ValueError("目标物体不是单个自由关节")
    joint = joints[0]
    original_qpos = np.asarray(env.sim.data.get_joint_qpos(joint)).copy()
    qvel = np.asarray(env.sim.data.get_joint_qvel(joint)).copy()
    qpos = original_qpos.copy()
    if qpos.shape != (7,) or qvel.shape != (6,):
        raise ValueError("目标自由关节的维度不匹配")
    qpos[:2] += [dx, dy]
    env.sim.data.set_joint_qpos(joint, qpos)
    env.sim.data.set_joint_qvel(joint, np.zeros_like(qvel))
    fresh = env.regenerate_obs_from_state(env.get_sim_state())
    overlaps = bowl_penetrations(env)
    if overlaps:
        deepest = max(overlaps, key=lambda item: item["depth_m"])
        env.sim.data.set_joint_qpos(joint, original_qpos)
        env.sim.data.set_joint_qvel(joint, qvel)
        env.regenerate_obs_from_state(env.get_sim_state())
        other = "盘子" if "plate" in (deepest["other_geom"] or "") else "其他物体"
        raise ValueError(f"碗与{other}发生重叠（约{deepest['depth_m'] * 1000:.1f}毫米）；请把碗挪远后再加载")
    if not np.allclose(fresh[f"{TARGET}_pos"], after, atol=1e-6, rtol=0):
        raise ValueError("碗的位置更新与请求不一致")
    return fresh, {"before_xyz": before.tolist(), "after_xyz": after.tolist(), "dx_m": dx, "dy_m": dy}


class LiberoTask:
    def __init__(self):
        from libero.libero import benchmark, get_libero_path
        import torch
        suite = benchmark.get_benchmark_dict()["libero_spatial"]()
        matches = [suite.get_task(i) for i in range(suite.n_tasks)
                   if suite.get_task(i).language.strip().casefold() == TASK_LANGUAGE]
        if len(matches) != 1:
            raise ValueError("未找到唯一匹配的黑碗放置任务")
        self.task = matches[0]
        self.bddl = Path(get_libero_path("bddl_files")) / self.task.problem_folder / self.task.bddl_file
        path = Path(get_libero_path("init_states")) / self.task.problem_folder / self.task.init_states_file
        self.initial_states = torch.load(path, weights_only=False)  # Pinned, trusted local LIBERO file.
        self.layout_count = len(self.initial_states)
        self.metadata = {"bddl_sha256": hashlib.sha256(self.bddl.read_bytes()).hexdigest(),
                         "reference_states_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def create(self, layout):
        from libero.libero.envs import OffScreenRenderEnv
        env = OffScreenRenderEnv(bddl_file_name=str(self.bddl), camera_heights=256, camera_widths=256,
                                 ignore_done=True)
        try:
            env.seed(42)
            env.reset()
            obs = env.set_init_state(self.initial_states[layout])
            for _ in range(10):
                obs, _, _, _ = env.step(DUMMY_ACTION)
            return env, obs
        except Exception:
            env.close()
            raise


class PolicyConnection:
    """Bounded WebSocket requests, executed by the lab's background inference thread."""
    def __init__(self, port):
        self.port, self.ws, self.metadata = port, None, None
        self.closed = False

    def metadata_only(self):
        from openpi_client import msgpack_numpy
        from websockets.sync.client import connect
        try:
            if self.closed:
                raise RuntimeError("推理连接已关闭")
            if self.ws is None:
                self.ws = connect(f"ws://127.0.0.1:{self.port}", open_timeout=3, close_timeout=1,
                                  compression=None, max_size=None)
                if self.closed:
                    raise RuntimeError("推理连接已关闭")
                self.metadata = msgpack_numpy.unpackb(self.ws.recv(timeout=3))
            return self.metadata
        except Exception:
            self._disconnect()
            raise

    def infer(self, query):
        from openpi_client import msgpack_numpy
        try:
            self.metadata_only()
            ws = self.ws
            ws.send(msgpack_numpy.Packer().pack(query))
            response = ws.recv(timeout=60)
            if isinstance(response, str):
                raise RuntimeError(response)
            return msgpack_numpy.unpackb(response), self.metadata
        except Exception:
            self._disconnect()
            raise

    def _disconnect(self):
        ws, self.ws = self.ws, None
        if ws is not None:
            ws.close()

    def close(self):
        self.closed = True
        self._disconnect()


class TrialRecord:
    def __init__(self, directory, scene, obs, state):
        import numpy as np
        import imageio.v2 as imageio
        self.directory = directory
        directory.mkdir()
        self.result = {**scene, "control_steps": 0, "inference_calls": 0,
                       "success": False, "first_benchmark_success_control_steps": None,
                       "post_success_steps_executed": 0, "final_benchmark_success": None,
                       "release_and_retreat_success": None, "model_metadata": None}
        self.frames = [np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])]
        self.handle = (directory / "actions.csv").open("x", newline="")
        columns = ["step", "reward", "done"] + [f"state_{i}" for i in range(8)] + [f"action_{i}" for i in range(7)]
        columns += [f"post_eef_{i}" for i in range(3)] + [f"post_bowl_{i}" for i in range(3)]
        columns += [f"post_finger_{i}" for i in range(2)]
        self.writer = csv.DictWriter(self.handle, fieldnames=columns)
        self.writer.writeheader()
        self.saved = False
        np.save(directory / "initial_state.npy", state, allow_pickle=False)
        imageio.imwrite(directory / "initial-agentview.png", self.frames[0])
        imageio.imwrite(directory / "initial-wrist.png", np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]))
        write_json(directory / "scene.json", scene)

    def append(self, state, action, obs, reward, done):
        import numpy as np
        row = {"step": self.result["control_steps"], "reward": float(reward), "done": bool(done)}
        for name, values in [("state", state), ("action", action), ("post_eef", obs["robot0_eef_pos"]),
                             ("post_bowl", obs[f"{TARGET}_pos"]), ("post_finger", obs["robot0_gripper_qpos"])]:
            row.update({f"{name}_{i}": float(value) for i, value in enumerate(values)})
        self.writer.writerow(row)
        self.handle.flush()
        self.frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
        self.result["control_steps"] += 1

    def save(self, reason, state, obs=None):
        import imageio.v2 as imageio
        import numpy as np
        if self.saved:
            return
        self.handle.flush()
        self.result.update(stop_reason=reason, final_state_sha256=state_digest(state))
        np.save(self.directory / "final_state.npy", state, allow_pickle=False)
        picture = self.frames[-1] if obs is None else np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        imageio.imwrite(self.directory / "final-agentview.png", picture)
        # Write JSON first so a failed encoder still leaves the control record recoverable.
        write_json(self.directory / "result.json", self.result)
        if self.result["control_steps"]:
            imageio.mimwrite(self.directory / "video.mp4", self.frames, fps=10)
            self.result.update(video="video.mp4", video_frames=len(self.frames), video_fps=10)
        write_json(self.directory / "result.json", self.result)
        self.handle.close()
        self.frames.clear()
        self.saved = True


class SimulationLab:
    def __init__(self, output, task, policy, viewer_factory=None):
        from live_preview import LivePreview
        self.output = prepare_output(output)
        self.task, self.layout_count, self.policy = task, task.layout_count, policy
        self.viewer_factory = viewer_factory or LivePreview
        self.env = self.view = self.obs = self.record = None
        self.pending = None
        self.generation, self.scene_number = 0, 0
        self.layout_id, self.finished, self.closed = None, False, False
        self.status = "请选择布局，然后点击加载"
        self.step_requested, self.next_step_time = False, 0.0
        self.was_paused = True
        self.known_state_hash = None
        self.model_name = "运行后读取模型信息"
        self.events = (self.output / "operations.jsonl").open("x")
        write_json(self.output / "session.json", {
            "task": TASK_LANGUAGE, "layout_count": task.layout_count,
            "source_sha256": {name: hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest()
                              for name in ("simulation_lab.py", "lab_panel.py", "lab_jobs.py", "correction_workbench.py",
                                           "correction_data.py", "prepare_corrections.py", "train_corrections.py")},
            "formal_benchmark": False,
        })

    def event(self, kind, **details):
        self.events.write(json.dumps({"time_ns": time.time_ns(), "scene": self.scene_number,
                                      "kind": kind, **details}, ensure_ascii=False, allow_nan=False) + "\n")
        self.events.flush()

    def _close_scene(self):
        if self.view is not None:
            self.view.close()
            self.view = None
        if self.env is not None:
            self.env.close()
            self.env = None

    def reset(self, layout_id, dx_m=0.0, dy_m=0.0):
        if not 0 <= layout_id < self.layout_count:
            raise ValueError("布局编号超出范围")
        if not all(math.isfinite(v) and abs(v) <= 0.1 for v in (dx_m, dy_m)):
            raise ValueError("位置偏移需在 ±10 cm 范围内")
        self.save("reset")  # Failures propagate: keep the old scene until it is saved.
        self._close_scene()
        self.generation += 1
        self.record, self.obs, self.layout_id = None, None, None
        self.finished, self.step_requested = False, False
        try:
            self.env, self.obs = self.task.create(layout_id)
            self.obs, edit = move_bowl(self.env, self.obs, dx_m, dy_m)
            self.layout_id = layout_id
            self.scene_number += 1
            state = self.env.get_sim_state().copy()
            scene = {"layout_id": layout_id, "seed": 42, "bowl_edit": edit,
                     "task_metadata": getattr(self.task, "metadata", {}),
                     "initial_state_sha256": state_digest(state), "task_language": TASK_LANGUAGE,
                     "interactive_experiment": True, "formal_benchmark": False,
                     "untracked_native_gui_edits_possible": True, "replan_steps": 1,
                     "max_steps": INTERACTIVE_STEP_LIMIT, "post_success_steps": None, "physics_horizon_ignored": True,
                     "stop_policy": "max_steps_or_user_save; On is recorded without early stopping",
                     "success_definition": "Official LIBERO On predicate; release/retreat requires inspection",
                     "video_timing": "initial frame plus one per control step, replay at 10 FPS; pauses omitted",
                     "policy_rng": "Server RNG advances across resets; reset does not reseed model sampling"}
            directory = self.output / f"scene-{self.scene_number:03d}_init-{layout_id:03d}"
            self.record = TrialRecord(directory, scene, self.obs, state)
            self.view = self.viewer_factory(self.env)
            self.was_paused = True
            self.known_state_hash = state_digest(state)
            self.status = "已加载，暂停中；可运行模型或单步"
            self.event("reset", **scene)
        except Exception as error:
            if self.record is not None and self.env is not None:
                self.record.result["setup_error"] = str(error)
                self.record.save("scene_load_error", self.env.get_sim_state(), self.obs)
            self._close_scene()
            self.finished = True
            raise

    def toggle_run(self):
        if self.view is None or self.finished:
            raise ValueError("请先加载或重置布局")
        self.view.paused = not self.view.paused
        self.step_requested = False
        self.view.single_step = False
        if self.view.paused:
            self.generation += 1
        self.was_paused = self.view.paused
        self.status = "已暂停" if self.view.paused else "运行中"
        self.event("pause" if self.view.paused else "run")

    def single_step(self):
        if self.view is None or self.finished:
            raise ValueError("请先加载或重置布局")
        if not self.view.paused:
            self.generation += 1
        self.view.paused, self.step_requested = True, True
        self.view.single_step = False
        self.was_paused = True
        self.event("single_step")

    def _submit(self, query, request_hash):
        future = Future()
        self.pending = (future, self.generation, request_hash)
        policy = self.policy

        def work():
            try:
                future.set_result(policy.infer(query))
            except Exception as error:
                future.set_exception(error)
        threading.Thread(target=work, daemon=True, name="lab-policy-request").start()

    def tick(self):
        """Main Tk thread owns every scene edit and physics step; inference is asynchronous."""
        import numpy as np
        if self.closed or self.view is None:
            return
        if not self.view.is_running():
            self.save("viewer_closed")
            self._close_scene()
            return
        self.view.sync()
        if self.finished:
            self.view.paused = True
            return
        native_single_step = self.view.single_step
        self.view.single_step = False
        if self.view.paused and not self.was_paused:
            self.generation += 1
            self.step_requested = bool(native_single_step)
            self.event("native_pause")
        else:
            self.step_requested |= native_single_step
        self.was_paused = self.view.paused
        wants_step = not self.view.paused or self.step_requested
        try:
            with self.view.lock():
                current = self.env.get_sim_state().copy()
                current_hash = state_digest(current)
                if current_hash != self.known_state_hash:
                    self.event("native_state_change", state_sha256=current_hash)
                    np.save(self.record.directory / f"native-edit-{time.time_ns()}.npy", current, allow_pickle=False)
                self.known_state_hash = current_hash
                self.obs = self.env.regenerate_obs_from_state(current)
            if self.pending is None:
                if wants_step and time.monotonic() >= self.next_step_time:
                    from openpi_client import image_tools
                    def picture(key):
                        return image_tools.convert_to_uint8(image_tools.resize_with_pad(
                            np.ascontiguousarray(self.obs[key][::-1, ::-1]), 224, 224))
                    state = np.concatenate((self.obs["robot0_eef_pos"],
                                            quaternion_axis_angle(self.obs["robot0_eef_quat"]),
                                            self.obs["robot0_gripper_qpos"]))
                    if state.shape != (8,) or not np.isfinite(state).all():
                        raise ValueError("机器人观测维度或数值不正确")
                    query = {"observation/image": picture("agentview_image"),
                             "observation/wrist_image": picture("robot0_eye_in_hand_image"),
                             "observation/state": state, "prompt": TASK_LANGUAGE}
                    self._submit(query, current_hash)
                return
            future, generation, request_hash = self.pending
            if not future.done():
                return
            if generation != self.generation or request_hash != current_hash:
                self.pending = None
                self.event("discard_stale_prediction")
                return
            if not wants_step:
                return
            self.pending = None
            response, metadata = future.result()
            chunk = np.asarray(response["actions"])
            if chunk.ndim != 2 or chunk.shape[0] < 1 or chunk.shape[1] != 7 or not np.isfinite(chunk).all():
                raise ValueError("模型返回的动作维度或数值不正确")
            previous = self.record.result["model_metadata"]
            if previous is not None and metadata != previous:
                raise ValueError("推理模型已变化，请重置布局后开始新实验")
            self.record.result["model_metadata"] = metadata
            self.model_name = metadata.get("model_stage", metadata.get("checkpoint", "已连接模型"))
            state = np.concatenate((self.obs["robot0_eef_pos"], quaternion_axis_angle(self.obs["robot0_eef_quat"]),
                                    self.obs["robot0_gripper_qpos"]))
            with self.view.lock():
                if self.view.paused and not self.step_requested:
                    self.event("discard_prediction_after_pause")
                    return
                if state_digest(self.env.get_sim_state()) != request_hash:
                    self.event("discard_stale_prediction")
                    return
                self.obs, reward, done, _ = self.env.step(chunk[0].tolist())
                self.known_state_hash = state_digest(self.env.get_sim_state())
            self.record.result["inference_calls"] += 1
            self.record.append(state, chunk[0], self.obs, reward, done)
            self.step_requested = False
            self.next_step_time = time.monotonic() + self.view.control_dt
            update_benchmark_progress(self.record.result, done, 0)
            self.status = "已收到 On 信号，继续执行至观察上限；松爪撤手待复核" if self.record.result["success"] else "模型执行中"
            if self.record.result["control_steps"] >= INTERACTIVE_STEP_LIMIT:
                self.save("step_limit")
        except Exception as error:
            self.view.paused = True
            self.step_requested = False
            self.status = f"已暂停：{error}"
            self.event("error", error=str(error))
            raise

    def save(self, reason="user_saved"):
        if self.view is not None:
            self.view.paused = True
        self.step_requested = False
        self.finished = True
        if self.record is not None and not self.record.saved:
            with self.view.lock() if self.view is not None else contextlib.nullcontext():
                state = self.env.get_sim_state().copy()
                self.obs = self.env.regenerate_obs_from_state(state)
            self.record.save(reason, state, self.obs)
            self.event("saved", reason=reason, directory=str(self.record.directory))
        self.status = ("已到观察上限并保存；On 信号不代表完成松爪撤手，请复核录像"
                       if reason == "step_limit" else "本轮已保存；可更换布局或重置后继续")

    def state(self):
        result = self.record.result if self.record is not None else {}
        waiting = self.pending is not None and not self.pending[0].done() and not self.finished
        return {"status": "等待模型返回；可暂停或重置布局" if waiting else self.status, "ready": self.view is not None,
                "running": self.view is not None and not self.view.paused and not self.finished,
                "finished": self.finished, "layout_id": self.layout_id,
                "control_steps": result.get("control_steps", 0), "on_reached": result.get("success", False),
                "model_name": self.model_name, "output_dir": str(self.output),
                "model_path": (result.get("model_metadata") or {}).get("checkpoint", "")}

    def close(self):
        if not self.closed:
            self.save("exit")
            self._close_scene()
            self.policy.close()
            self.event("exit")
            self.events.close()
            self.closed = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8755)
    parser.add_argument("--resume-correction", type=Path, help="恢复纠正草稿并暂停，不执行后续动作")
    args = parser.parse_args()
    # Cover direct CLI launches as well as desktop launches before opening any GPU scene.
    import fcntl
    lock_path = ROOT / ".local/lab-panel.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            from lab_launcher import notify
            notify("仿真实验室已在运行，请切换到已有窗口。")
            return
        run_gui(args.port, args.resume_correction)


def run_gui(port, resume_correction=None):
    # Select GLX before importing robosuite/OpenGL; initialize no model or training job.
    os.environ["MUJOCO_GL"] = os.environ["PYOPENGL_PLATFORM"] = "glx"
    os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".local/libero"))
    import tkinter as tk
    import sys
    sys.path.insert(0, str(ROOT / "openpi/third_party/libero"))
    from lab_panel import LabPanel
    from correction_workbench import CorrectionWorkbench
    output = ROOT / "evaluations" / f"interactive_lab_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}"
    task = LiberoTask()
    lab = CorrectionWorkbench(output, task, port=port)
    root = tk.Tk()
    panel = LabPanel(root, lab)
    if resume_correction is not None:
        panel.notebook.select(panel.correction_page)
        panel._perform(lambda: lab.resume_correction(resume_correction))
    try:
        root.mainloop()
    finally:
        lab.close()


if __name__ == "__main__":
    main()
