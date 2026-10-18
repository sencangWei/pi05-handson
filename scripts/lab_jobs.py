"""Course-only job ownership; closing the GUI deliberately leaves jobs running."""
import errno
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import threading
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("model", "training", "tensorboard")


class LabJobs:
    def __init__(self, root=ROOT, port=8755):
        self.root = Path(root).resolve()
        if isinstance(port, bool) or not isinstance(port, int) or not 0 < port < 65536:
            raise ValueError("Invalid serving port")
        self.port = port
        self.registry = self.root / ".local/lab-jobs.json"
        self._lock = threading.RLock()
        self._children = {}
        try:
            records = json.loads(self.registry.read_text())
            self._jobs = {k: v for k, v in records.items() if k in KINDS and isinstance(v, dict)}
        except (OSError, ValueError, AttributeError):
            self._jobs = {}

    def _save(self):
        self.registry.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.registry.with_name(f".lab-jobs-{os.getpid()}-{time.time_ns()}.tmp")
        temporary.write_text(json.dumps(self._jobs, indent=2) + "\n")
        temporary.replace(self.registry)

    def validate_checkpoint(self, checkpoint):
        base = self.root / "checkpoints/pi05_handson_lora"
        path = Path(checkpoint).expanduser()
        path = (path if path.is_absolute() else self.root / path).resolve()
        try:
            relative = path.relative_to(base)
        except ValueError:
            raise ValueError("Checkpoint must belong to this course") from None
        if len(relative.parts) != 2 or not path.name.isdecimal():
            raise ValueError("Expected a course experiment and numeric checkpoint step")
        for required in ("params/manifest.ocdbt", "assets/course/libero_black_bowl/norm_stats.json"):
            if not (path / required).is_file():
                raise ValueError(f"Incomplete checkpoint: {path / required}")
        return path

    def checkpoints(self):
        found = []
        for candidate in (self.root / "checkpoints/pi05_handson_lora").glob("*/[0-9]*"):
            try:
                found.append(str(self.validate_checkpoint(candidate)))
            except ValueError:
                pass
        return sorted(set(found), key=lambda path: (-(Path(path) / "params/manifest.ocdbt").stat().st_mtime_ns, path))

    @staticmethod
    def _port_busy(port):
        # Bind probing does not send a malformed HTTP request to a websocket server.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
            except OSError as exc:
                if exc.errno in (errno.EADDRINUSE, errno.EACCES):
                    return True
                raise
        return False

    def _owned_process(self, kind):
        record = self._jobs.get(kind, {})
        try:
            process = psutil.Process(record["pid"])
            if (process.uids().real == os.getuid()
                    and process.create_time() == record["create_time"]
                    and process.cmdline() == record["cmdline"]
                    and process.is_running() and process.status() != psutil.STATUS_ZOMBIE):
                return process
        except (KeyError, TypeError, ValueError, psutil.Error):
            pass
        return None

    def _state(self, kind, busy=False):
        record = self._jobs.get(kind, {})
        owned = self._owned_process(kind) is not None
        child = self._children.get(kind)
        exit_code = child.poll() if child is not None else record.get("exit_code")
        state = {"running": owned or busy, "owned": owned,
                 "pid": record.get("pid") if owned else None, "log": record.get("log", "")}
        if exit_code is not None:
            state["exit_code"] = exit_code
        return state

    def snapshot(self):
        with self._lock:
            busy = self._port_busy(self.port)
            return {"model": self._state("model", busy), "training": self._state("training"),
                    "tensorboard": self._state("tensorboard", self._port_busy(6006)), "port_busy": busy}

    def tail(self, kind, limit=8000):
        if kind not in KINDS or not isinstance(limit, int) or limit < 0:
            raise ValueError("Invalid job kind or log limit")
        with self._lock:
            path = Path(self._jobs.get(kind, {}).get("log", ""))
            try:
                # Registry paths are restricted even when the registry was edited externally.
                if path.resolve().parent != (self.root / "logs").resolve() or not limit:
                    return ""
                with path.open("rb") as handle:
                    handle.seek(0, os.SEEK_END)
                    handle.seek(max(0, handle.tell() - limit * 4))
                    return handle.read().decode("utf-8", errors="replace")[-limit:]
            except OSError:
                return ""

    def _launch(self, kind, command):
        log = self.root / "logs" / f"ui_{kind}_{time.time_ns()}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        for key, value in {"HF_HUB_OFFLINE": "1", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                           "XLA_PYTHON_CLIENT_MEM_FRACTION": ".55", "OMP_NUM_THREADS": "4"}.items():
            environment.setdefault(key, value)
        if kind in ("model", "training") and environment.get("JAX_PLATFORMS", "").strip().lower() == "cpu":
            environment.pop("JAX_PLATFORMS")
        with log.open("ab") as output:
            child = subprocess.Popen(command, cwd=self.root, env=environment, stdout=output,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        self._children[kind] = child
        record = {"pid": child.pid, "log": str(log), "cmdline": command}
        try:
            record["create_time"] = psutil.Process(child.pid).create_time()
        except psutil.Error:
            record["exit_code"] = child.poll()
        self._jobs[kind] = record
        self._save()
        return self._state(kind)

    def start_model(self, checkpoint):
        with self._lock:
            checkpoint = self.validate_checkpoint(checkpoint)
            if self._owned_process("training"):
                raise RuntimeError("Stop training before loading a model")
            if self._owned_process("model"):
                raise RuntimeError("A managed model is already running")
            if self._port_busy(self.port):
                raise RuntimeError("Serving port is occupied; external processes are never stopped")
            return self._launch("model", [str(self.root / ".venv/bin/python"), "-u",
                str(self.root / "scripts/serve_checkpoint.py"), "--checkpoint", str(checkpoint), "--port", str(self.port)])

    @staticmethod
    def _positive(value, name, integer=False):
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{name} must be positive and finite") from None
        if isinstance(value, bool) or not math.isfinite(number) or number <= 0 or (integer and not number.is_integer()):
            raise ValueError(f"{name} must be positive and finite" + (" integer" if integer else ""))
        return int(number) if integer else number

    def validate_training(self, parent, steps, batch_size, learning_rate):
        return (self.validate_checkpoint(parent), self._positive(steps, "Steps", integer=True),
                self._positive(batch_size, "Batch size", integer=True),
                self._positive(learning_rate, "Learning rate"))

    def start_training(self, parent, steps, batch_size, learning_rate, uniform=False):
        with self._lock:
            parent, steps, batch_size, learning_rate = self.validate_training(parent, steps, batch_size, learning_rate)
            if self._owned_process("training"):
                raise RuntimeError("Training is already running")
            if self._owned_process("model") or self._port_busy(self.port):
                raise RuntimeError("Stop serving before training; external processes are never stopped")
            experiment = f"ui_{'uniform' if uniform else 'focus'}_{time.time_ns()}"
            command = [str(self.root / ".venv/bin/python"), "-u", str(self.root / "scripts/train_release_focus.py"),
                       "--parent-checkpoint", str(parent), "--steps", str(steps), "--batch-size", str(batch_size),
                       "--learning-rate", str(learning_rate), "--experiment", experiment]
            if uniform:
                command.append("--uniform-control")
            return self._launch("training", command)

    def start_correction_training(self, parent, steps, batch_size, learning_rate, manifest):
        with self._lock:
            parent, steps, batch_size, learning_rate = self.validate_training(parent, steps, batch_size, learning_rate)
            manifest = Path(manifest).resolve()
            if manifest.parent != self.root / "evidence" or not manifest.name.startswith("corrections-") or not manifest.name.endswith("-manifest.json"):
                raise ValueError("Expected this course's correction manifest")
            if not json.loads(manifest.read_text()).get("ready"):
                raise ValueError("Correction dataset is not ready")
            if self._owned_process("training") or self._owned_process("model") or self._port_busy(self.port):
                raise RuntimeError("Stop managed model/training first; external services are never stopped")
            experiment = f"ui_correct_{time.time_ns()}"
            return self._launch("training", [str(self.root / ".venv/bin/python"), "-u",
                str(self.root / "scripts/train_corrections.py"), "--manifest", str(manifest),
                "--parent-checkpoint", str(parent), "--steps", str(steps), "--batch-size", str(batch_size),
                "--learning-rate", str(learning_rate), "--experiment", experiment])

    def start_tensorboard(self):
        with self._lock:
            if self._owned_process("tensorboard") or self._port_busy(6006):
                return self._state("tensorboard", self._port_busy(6006))
            return self._launch("tensorboard", [str(self.root / ".venv-sim/bin/python"), "-m", "tensorboard.main",
                "--logdir", str(self.root / ".local/tensorboard"), "--host", "127.0.0.1", "--port", "6006"])

    def _stop(self, kind):
        with self._lock:
            process = self._owned_process(kind)
            if process is not None:
                try:
                    process.terminate()
                    try:
                        exit_code = process.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        process = self._owned_process(kind)
                        if process is None:
                            return self._state(kind)
                        process.kill()
                        exit_code = process.wait(timeout=3)
                    self._jobs[kind]["exit_code"] = exit_code
                    self._save()
                except psutil.NoSuchProcess:
                    pass
            return self._state(kind)

    def stop_model(self):
        return self._stop("model")

    def stop_training(self):
        return self._stop("training")
