"""Record the visible lab panel and its owned MuJoCo window; recycle saved demos."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

KIND = "pi05-lab-screen-demo-v1"


def capture_region(panel_id):
    from Xlib import X, display
    connection = display.Display()
    try:
        root = connection.screen().root
        clients = root.get_full_property(connection.intern_atom("_NET_CLIENT_LIST"), X.AnyPropertyType)
        windows = [connection.create_resource_object("window", int(w)) for w in clients.value]
        panel = connection.create_resource_object("window", panel_id)
        ids = {w.id for w in windows}
        while panel.id not in ids:
            parent = panel.query_tree().parent
            if parent.id == root.id:
                raise ValueError("未找到仿真面板窗口")
            panel = parent
        viewers = []
        for window in windows:
            owner = window.get_full_property(connection.intern_atom("_NET_WM_PID"), X.AnyPropertyType)
            if owner is not None and int(owner.value[0]) == os.getpid():
                if (window.get_wm_name() or "").startswith("MuJoCo"):
                    viewers.append(window)
        if len(viewers) != 1:
            raise ValueError("请先加载布局，保持一个 MuJoCo 三维窗口")
        def bounds(window):
            if window.get_attributes().map_state != X.IsViewable:
                raise ValueError("请显示面板及三维窗口，不要最小化")
            geometry = window.get_geometry()
            position = root.translate_coords(window, 0, 0)
            prop = window.get_full_property(connection.intern_atom("_NET_FRAME_EXTENTS"), X.AnyPropertyType)
            left, right, top, bottom = prop.value if prop is not None else (0, 0, 0, 0)
            return (position.x - int(left), position.y - int(top),
                    position.x + geometry.width + int(right), position.y + geometry.height + int(bottom))
        windows = (panel, viewers[0])
        rectangles = [bounds(window) for window in windows]
        screen = root.get_geometry()
        workarea = root.get_full_property(connection.intern_atom("_NET_WORKAREA"), X.AnyPropertyType)
        desktop = root.get_full_property(connection.intern_atom("_NET_CURRENT_DESKTOP"), X.AnyPropertyType)
        index = 4 * int(desktop.value[0]) if desktop is not None else 0
        work = workarea.value[index:index + 4] if workarea is not None else (0, 0, screen.width, screen.height)
        work_x, work_y, work_width, work_height = map(int, work)
        a, b = rectangles
        if max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3]):
            # Put both requested windows side by side rather than recording one hiding the other.
            panel_width = min(a[2] - a[0], max(720, work_width // 3))
            viewer_width = min(b[2] - b[0], work_width - panel_width - 24)
            if viewer_width < 400:
                raise ValueError("屏幕不足以同时显示面板和三维窗口，请先调整显示分辨率")
            height = min(max(a[3] - a[1], b[3] - b[1]), work_height - 72)
            for window, x, width in [(panel, work_x + 12, panel_width),
                                     (viewers[0], work_x + panel_width + 12, viewer_width)]:
                window.configure(x=x, y=work_y + 12, width=width, height=height)
            connection.sync()
            time.sleep(.2)
            rectangles = [bounds(window) for window in windows]
            a, b = rectangles
            if max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3]):
                raise ValueError("请把面板和三维窗口并排显示后再录制")
        x, y = min(r[0] for r in rectangles), min(r[1] for r in rectangles)
        right, bottom = max(r[2] for r in rectangles), max(r[3] for r in rectangles)
        if x < 0 or y < 0 or right > screen.width or bottom > screen.height:
            raise ValueError("请把面板和三维窗口完整移入屏幕后再录制")
        return {"x": x, "y": y, "width": right - x, "height": bottom - y,
                "window_ids": [panel.id, viewers[0].id]}
    finally:
        connection.close()


class DemoRecording:
    def __init__(self, project):
        self.project = Path(project).resolve()
        self.directory = self.project / "demos"
        self.process = self.log = self.current = None
        self.message = "录制面板和三维窗口：请并排放好；录制中保持窗口位置，避免遮挡。"
        self.last = self._latest()

    def _latest(self):
        if self.directory.is_symlink():
            raise ValueError("Demo 目录不能是符号链接")
        for path in sorted(self.directory.glob("demo-*/demo.json"), reverse=True):
            if path.is_symlink() or path.parent.is_symlink():
                continue
            try:
                data = json.loads(path.read_text())
                if data.get("kind") == KIND and data.get("status") == "saved":
                    return path.parent
            except (OSError, ValueError):
                continue
        return None

    @property
    def recording(self):
        return self.process is not None

    def start(self, panel_id, source_scene):
        if self.recording:
            raise ValueError("Demo 已在录制")
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise ValueError("录制需要本机 ffmpeg 和 ffprobe")
        region = capture_region(panel_id)
        if self.directory.is_symlink():
            raise ValueError("Demo 目录不能是符号链接")
        self.directory.mkdir(exist_ok=True)
        directory = self.directory / f"demo-{time.time_ns()}"
        directory.mkdir()
        metadata = {"kind": KIND, "status": "recording", "source_scene": str(source_scene),
                    "started_at_ns": time.time_ns(), "fps": 15, "region": region,
                    "timing": "wall clock, including pauses", "video": "demo.mp4"}
        (directory / "demo.json").write_text(json.dumps(metadata, indent=2) + "\n")
        log = (directory / "encoder.log").open("xb")
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-f", "x11grab",
                   "-framerate", "15", "-video_size", f"{region['width']}x{region['height']}",
                   "-i", f"{os.environ['DISPLAY']}+{region['x']},{region['y']}", "-an",
                   "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-threads", "2",
                   "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                   str(directory / "demo.mp4")]
        try:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        except Exception:
            log.close()
            raise
        self.process, self.log, self.current = process, log, directory
        self.message = f"正在录制面板＋三维窗口；再次点击停止保存：{directory.name}"

    def stop(self):
        if not self.recording:
            return
        process, directory = self.process, self.current
        try:
            if process.poll() is None:
                try:
                    process.stdin.write(b"q\n")
                    process.stdin.flush()
                except BrokenPipeError:
                    pass
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=3)
            metadata = json.loads((directory / "demo.json").read_text())
            probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                    "stream=codec_name,width,height:format=duration", "-of", "json",
                                    str(directory / "demo.mp4")], capture_output=True, text=True, timeout=8)
            video = json.loads(probe.stdout) if probe.returncode == 0 else {}
            valid = process.returncode == 0 and video.get("streams") and float(video.get("format", {}).get("duration", 0)) > 0
            metadata.update(status="saved" if valid else "failed", stopped_at_ns=time.time_ns(), probe=video)
            (directory / "demo.json").write_text(json.dumps(metadata, indent=2) + "\n")
            if not valid:
                raise ValueError(f"Demo 未正常完成，原文件保留，请查看 {directory / 'encoder.log'}")
            self.last = directory
            self.message = f"Demo 已保存：{directory / 'demo.mp4'}"
        finally:
            process.stdin.close()
            self.log.close()
            self.process = self.log = self.current = None

    def poll(self):
        if self.recording and self.process.poll() is not None:
            try:
                self.stop()
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                self.message = str(error)

    def delete_last(self):
        if self.recording:
            raise ValueError("请先停止录制，再删除上一个 Demo")
        source = self._latest()
        if source is None:
            raise ValueError("没有可删除的 Demo")
        target = self.project / ".local/demo-trash" / f"{source.name}-{time.time_ns()}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.parent.resolve().is_relative_to(self.project):
            raise ValueError("Demo 回收目录超出项目范围")
        source.rename(target)
        self.last = self._latest()
        self.message = f"已删除 {source.name}；可从 {target} 找回"

