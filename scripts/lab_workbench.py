"""User-triggered model/training jobs and local result review for the lab panel."""
from concurrent.futures import Future
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time
import urllib.request
import webbrowser

from evaluate_libero import write_json
from export_training_curves import parse_metrics
from lab_jobs import LabJobs
from demo_recording import DemoRecording
from responsive_lab import ResponsiveSimulationLab
from simulation_lab import ROOT, PolicyConnection


class LabWorkbench(ResponsiveSimulationLab):
    def __init__(self, output, task, *, port=8755, jobs=None, root=ROOT, **kwargs):
        self.project = Path(root).resolve()
        self.demo = DemoRecording(self.project)
        self.port = port
        self.jobs = jobs or LabJobs(self.project, port)
        self.connection = "尚未检查连接；可连接外部服务或在本页启动模型"
        self.background = {}
        self.job_state = self.jobs.snapshot()
        self.last_job_poll = 0.0
        super().__init__(output, task, PolicyConnection(port), **kwargs)

    def checkpoints(self):
        return self.jobs.checkpoints()

    def _new_connection(self):
        self.generation += 1
        self.background.pop("connection", None)
        self.model_name = "新模型尚未检查连接"
        self.policy.close()
        self.policy = PolicyConnection(self.port)
        self.connection = "连接已清除，请检查或运行新模型"

    def _archive_scene(self, reason):
        self.save(reason)
        self._close_scene()
        self._new_connection()

    def start_model(self, checkpoint):
        status = self.jobs.snapshot()
        if status["training"]["running"]:
            raise ValueError("请先等待训练完成或停止本软件的训练")
        if status["port_busy"] and not status["model"]["owned"]:
            raise ValueError("8755 端口有外部服务。可直接检查并连接；切换模型须先在原终端退出外部服务")
        checkpoint = self.jobs.validate_checkpoint(checkpoint)
        self._archive_scene("switch_model")
        self.jobs.stop_model()
        self.jobs.start_model(checkpoint)
        self.job_state = self.jobs.snapshot()
        self.connection = "模型正在后台加载，可在下方查看日志，随后点击检查连接"
        self.event("model_started", checkpoint=str(checkpoint))

    def stop_model(self):
        if not self.jobs.snapshot()["model"]["owned"]:
            raise ValueError("当前服务由外部启动，请在其原终端退出；本软件不会停止它")
        self._archive_scene("stop_model")
        self.jobs.stop_model()
        self.connection = "本软件的模型服务已停止"
        self.event("model_stopped")

    def start_training(self, parent, steps, batch_size, learning_rate, uniform=False):
        status = self.jobs.snapshot()
        if status["training"]["running"]:
            raise ValueError("已有训练正在运行")
        if status["port_busy"] and not status["model"]["owned"]:
            raise ValueError("训练前请在原终端退出外部推理服务，以释放显存；本软件只停止自己启动的服务")
        parent, steps, batch_size, learning_rate = self.jobs.validate_training(parent, steps, batch_size, learning_rate)
        self._archive_scene("start_training")
        self.jobs.stop_model()
        job = self.jobs.start_training(parent, steps, batch_size, learning_rate, uniform)
        self.job_state = self.jobs.snapshot()
        self.event("training_started", parent=str(parent), steps=int(steps), batch_size=int(batch_size),
                   learning_rate=float(learning_rate), uniform=bool(uniform), log=job["log"])
        self.status = "训练已由你启动；可查看日志和刷新曲线快照"

    def stop_training(self):
        status = self.jobs.snapshot()["training"]
        if not status["owned"]:
            raise ValueError("没有本软件启动的运行中训练")
        self.jobs.stop_training()
        self.job_state = self.jobs.snapshot()
        self.status = "训练已停止，已有 checkpoint 保留；未保存的更新不会恢复"
        self.event("training_stopped")

    def _ensure_not_training(self):
        self.job_state = self.jobs.snapshot()
        if self.job_state["training"]["running"]:
            raise ValueError("训练期间请等待完成后再运行模型")

    def reset(self, layout_id, dx_m=0.0, dy_m=0.0):
        self._ensure_not_training()
        super().reset(layout_id, dx_m, dy_m)

    def toggle_run(self):
        self._ensure_not_training()
        super().toggle_run()

    def single_step(self):
        self._ensure_not_training()
        super().single_step()

    def _background(self, name, operation):
        if name in self.background and not self.background[name].done():
            return
        future = Future()
        self.background[name] = future

        def work():
            try:
                future.set_result(operation())
            except Exception as error:
                future.set_exception(error)
        threading.Thread(target=work, daemon=True, name=f"lab-{name}").start()

    def check_connection(self):
        self.connection = "正在检查模型身份，不执行动作"

        def probe():
            policy = PolicyConnection(self.port)
            try:
                return policy.metadata_only()
            finally:
                policy.close()
        self._background("connection", probe)

    def _collect_background(self):
        for name, future in list(self.background.items()):
            if not future.done():
                continue
            del self.background[name]
            try:
                value = future.result()
                if name == "connection":
                    label = value.get("model_stage", value.get("checkpoint", "模型服务"))
                    self.connection = f"已连接：{label}"
                    self.model_name = label
                    self.event("connection_checked", metadata=value)
                elif name == "curves":
                    self.status = f"曲线快照已更新：{value['source']}"
                    webbrowser.open(value["url"])
                    self.event("curve_snapshot", **value)
            except Exception as error:
                if name == "connection":
                    self.connection = f"未连接：{error}"
                else:
                    self.status = f"操作未完成：{error}"
                self.event("background_error", operation=name, error=str(error))

    def tick(self):
        try:
            super().tick()
        finally:
            self._collect_background()
            self.demo.poll()

    def toggle_demo(self, panel_id):
        if self.demo.recording:
            self.demo.stop()
        else:
            if self.view is None or not self.view.is_running():
                raise ValueError("请先加载仿真布局")
            self.demo.start(panel_id, self.record.directory)
        self.event("demo_recording", recording=self.demo.recording, message=self.demo.message)

    def delete_last_demo(self):
        self.demo.delete_last()
        self.event("demo_deleted", message=self.demo.message)

    def save(self, reason="user_saved"):
        if self.demo.recording:
            self.demo.stop()
        return super().save(reason)

    def state(self):
        value = super().state()
        value["training_running"] = self.job_state["training"]["running"]
        value["current_result_path"] = str(self.record.directory / "result.json") if self.record else ""
        value.update(demo_recording=self.demo.recording, demo_available=self.demo.last is not None,
                     demo_status=self.demo.message)
        return value

    def workbench_state(self):
        self._collect_background()
        if time.monotonic() - self.last_job_poll >= 0.8:
            self.job_state = self.jobs.snapshot()
            self.last_job_poll = time.monotonic()
        return {"jobs": self.job_state, "connection": self.connection,
                "model_log": self.jobs.tail("model"), "training_log": self.jobs.tail("training")}

    def dataset_info(self):
        path = self.project / "evidence/data_manifest.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def capture(self):
        if self.env is None or self.view is None or self.record is None:
            raise ValueError("请先加载场景")
        import imageio.v2 as imageio
        import numpy as np
        with self.view.lock():
            obs = self.env.regenerate_obs_from_state(self.env.get_sim_state())
        prefix = f"capture-{time.time_ns()}"
        for name, key in [("agentview", "agentview_image"), ("wrist", "robot0_eye_in_hand_image")]:
            imageio.imwrite(self.record.directory / f"{prefix}-{name}.png", np.ascontiguousarray(obs[key][::-1, ::-1]))
        self.status = "两个相机的截图已保存到本轮目录"
        self.event("capture", prefix=prefix)

    def reset_camera(self):
        if self.view is None:
            raise ValueError("请先加载场景")
        with self.view.lock():
            self.view.handle.cam.lookat[:] = [0, 0, 1.0]
            self.view.handle.cam.distance = 1.6
            self.view.handle.cam.azimuth = 130
            self.view.handle.cam.elevation = -30
        self.event("camera_reset")

    def top_camera(self):
        if self.view is None:
            raise ValueError("请先加载场景")
        self.view.top_view()
        self.event("camera_top")

    def _result_path(self, path):
        path = Path(path).expanduser().resolve()
        if not path.is_relative_to(self.project / "evaluations") or not path.is_file() or not path.name.endswith("result.json"):
            raise ValueError("请选择项目评测目录中的结果文件")
        return path

    @staticmethod
    def _digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def results(self):
        found = []
        paths = sorted((self.project / "evaluations").rglob("*result.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)[:200]
        for path in paths:
            try:
                path = self._result_path(path)
                data = json.loads(path.read_text())
                metadata = data.get("model_metadata") or {}
                if not metadata and (path.parent / "summary.json").is_file():
                    metadata = json.loads((path.parent / "summary.json").read_text()).get("model_metadata") or {}
                review_path = path.with_name(path.stem + ".review.json")
                review, note = "待复核", ""
                if review_path.is_file():
                    reviewed = json.loads(review_path.read_text())
                    note = reviewed.get("note", "")
                    if reviewed.get("source_result_sha256") != self._digest(path):
                        review = "复核过期"
                    elif reviewed.get("release_and_retreat_pass") is not None:
                        review = "收尾成功" if reviewed["release_and_retreat_pass"] else "收尾失败"
                video = (path.parent / (data.get("video") or "video.mp4")).resolve()
                if not video.is_relative_to(self.project / "evaluations"):
                    video = None
                found.append({"result_path": str(path), "directory": str(path.parent),
                              "video": str(video) if video is not None and video.is_file() else None,
                              "layout": data.get("layout_id", data.get("initial_state_id")),
                              "model": metadata.get("model_stage", metadata.get("checkpoint", "未知模型")),
                              "on": bool(data.get("success")), "steps": data.get("control_steps", 0),
                              "review": review, "note": note})
            except (OSError, ValueError, TypeError):
                continue
        return found

    def review_result(self, result_path, passed, note):
        path = self._result_path(result_path)
        if passed is not None and not isinstance(passed, bool):
            raise ValueError("复核结果必须为成功、失败或待复核")
        write_json(path.with_name(path.stem + ".review.json"), {
            "source_result_sha256": self._digest(path), "reviewed_at_ns": time.time_ns(),
            "release_and_retreat_pass": passed, "note": str(note).strip()[:2000],
            "review_type": "manual_local_user", "changes_official_benchmark_result": False,
        })
        self.event("manual_review", result=str(path), passed=passed)

    @staticmethod
    def _open(path):
        subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def play_result(self, result_path):
        path = self._result_path(result_path)
        data = json.loads(path.read_text())
        video = (path.parent / (data.get("video") or "video.mp4")).resolve()
        if not video.is_relative_to(self.project / "evaluations") or not video.is_file():
            raise ValueError("这轮没有已保存的视频，请先结束并保存")
        self._open(video)

    def open_result_directory(self, result_path):
        self._open(self._result_path(result_path).parent)

    def open_curves(self):
        def export():
            source = self.jobs.snapshot()["training"].get("log", "")
            if source and Path(source).is_file() and Path(source).resolve().parent == self.project / "logs":
                path = Path(source)
            else:
                candidates = sorted((self.project / "logs").glob("*training*.log"), key=lambda f: f.stat().st_mtime)
                if not candidates:
                    raise ValueError("没有训练日志")
                path = candidates[-1]
            rows = parse_metrics(path.read_text(errors="replace"))
            if not rows:
                raise ValueError("日志尚无 Step 指标，训练开始输出后再刷新曲线")
            from tensorboard.compat.proto.event_pb2 import Event
            from tensorboard.compat.proto.summary_pb2 import Summary
            from tensorboard.summary.writer.event_file_writer import EventFileWriter
            output = self.project / ".local/tensorboard" / f"gui_snapshot_{time.time_ns()}"
            writer = EventFileWriter(str(output))
            try:
                for step, values in rows:
                    writer.add_event(Event(wall_time=time.time(), step=step, summary=Summary(value=[
                        Summary.Value(tag=f"train/{key}", simple_value=value) for key, value in values.items()])))
                writer.flush()
            finally:
                writer.close()
            write_json(output / "snapshot.json", {"source": str(path), "source_sha256": self._digest(path),
                                                   "points": len(rows), "wall_time": "export time"})
            self.jobs.start_tensorboard()
            url = "http://127.0.0.1:6006"
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                try:
                    with urllib.request.urlopen(url, timeout=0.5):
                        break
                except OSError:
                    time.sleep(0.2)
            return {"source": str(path), "url": url, "output": str(output), "points": len(rows)}
        self._background("curves", export)
