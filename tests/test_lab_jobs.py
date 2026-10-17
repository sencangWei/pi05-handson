import json
from contextlib import ExitStack
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import lab_jobs


class FakeProcess:
    def __init__(self, pid, command):
        self.pid, self.command = pid, command
        self.created, self.uid, self.alive = 123.5, os.getuid(), True
        self.terminate = Mock(side_effect=self._finish)
        self.kill = Mock(side_effect=self._finish)

    def _finish(self):
        self.alive = False

    def create_time(self):
        return self.created

    def cmdline(self):
        return self.command

    def uids(self):
        return SimpleNamespace(real=self.uid)

    def is_running(self):
        return self.alive

    def status(self):
        return "running"

    def wait(self, timeout):
        return 0


class LabJobsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.checkpoint = self.root / "checkpoints/pi05_handson_lora/demo/42"
        for name in ("params/manifest.ocdbt", "assets/course/libero_black_bowl/norm_stats.json"):
            path = self.checkpoint / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
        self.processes = {}
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.popen = stack.enter_context(patch.object(lab_jobs.subprocess, "Popen", side_effect=self.launch))
        stack.enter_context(patch.object(lab_jobs.psutil, "Process", side_effect=self.lookup))
        self.port_busy = stack.enter_context(patch.object(lab_jobs.LabJobs, "_port_busy", return_value=False))
        self.jobs = lab_jobs.LabJobs(self.root)

    def launch(self, command, **kwargs):
        pid = 100 + len(self.processes)
        process = FakeProcess(pid, command)
        self.processes[pid] = process
        return SimpleNamespace(pid=pid, poll=lambda: None if process.alive else 0)

    def lookup(self, pid):
        if pid not in self.processes:
            raise lab_jobs.psutil.NoSuchProcess(pid)
        return self.processes[pid]

    def test_checkpoint_discovery_and_escape_rejection(self):
        self.assertEqual(self.jobs.checkpoints(), [str(self.checkpoint)])
        for checkpoint in (self.root, self.checkpoint.parent, self.checkpoint.parent / "42suffix"):
            with self.assertRaises(ValueError):
                self.jobs.start_model(checkpoint)
        with tempfile.TemporaryDirectory() as outside:
            escape = self.checkpoint.parent / "50"
            escape.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                self.jobs.start_model(escape)
        (self.checkpoint / "assets/course/libero_black_bowl/norm_stats.json").unlink()
        with self.assertRaises(ValueError):
            self.jobs.start_training(self.checkpoint, 10, 1, 0.001)
        self.popen.assert_not_called()

    def test_foreign_ports_are_reported_and_never_launched_or_stopped(self):
        self.port_busy.return_value = True
        with self.assertRaises(RuntimeError):
            self.jobs.start_model(self.checkpoint)
        with self.assertRaises(RuntimeError):
            self.jobs.start_training(self.checkpoint, 10, 1, 0.001)
        state = self.jobs.start_tensorboard()
        self.assertTrue(state["running"])
        self.assertFalse(state["owned"])
        self.assertEqual(self.jobs.stop_model()["owned"], False)
        self.assertTrue(self.jobs.snapshot()["port_busy"])
        self.popen.assert_not_called()

    def test_checkpoints_sort_by_manifest_mtime_then_path(self):
        checkpoints = [self.checkpoint]
        for name in ("2", "3"):
            checkpoint = self.checkpoint.parent / name
            for required in ("params/manifest.ocdbt", "assets/course/libero_black_bowl/norm_stats.json"):
                path = checkpoint / required
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")
            checkpoints.append(checkpoint)
        for checkpoint, modified in zip(checkpoints, (1_000_000, 2_000_000, 2_000_000)):
            os.utime(checkpoint / "params/manifest.ocdbt", ns=(modified, modified))
        self.assertEqual(self.jobs.checkpoints(), [str(checkpoints[1]), str(checkpoints[2]), str(checkpoints[0])])

    def test_model_arguments_environment_and_loading_exclusion(self):
        with patch.dict(os.environ, {"JAX_PLATFORMS": "cpu"}, clear=True):
            state = self.jobs.start_model(self.checkpoint)
        command, = self.popen.call_args.args
        self.assertEqual(command, [str(self.root / ".venv/bin/python"), "-u",
            str(self.root / "scripts/serve_checkpoint.py"), "--checkpoint", str(self.checkpoint), "--port", "8755"])
        options = self.popen.call_args.kwargs
        self.assertNotIn("JAX_PLATFORMS", options["env"])
        self.assertEqual(options["env"]["HF_HUB_OFFLINE"], "1")
        self.assertEqual(options["env"]["XLA_PYTHON_CLIENT_MEM_FRACTION"], ".55")
        self.assertNotIn("shell", options)
        self.assertTrue(state["owned"])
        self.assertTrue(self.jobs.snapshot()["model"]["running"])
        with self.assertRaises(RuntimeError):
            self.jobs.start_training(self.checkpoint, 10, 1, 0.001)
        with self.assertRaises(RuntimeError):
            self.jobs.start_model(self.checkpoint)

    def test_training_arguments_and_invalid_numbers(self):
        for steps, batch, rate in ((0, 1, 1e-4), (1.2, 1, 1e-4), (1, True, 1e-4),
                                   (1, 1, float("nan")), (1, 1, float("inf")), (1, 1, -0.1)):
            with self.assertRaises(ValueError):
                self.jobs.start_training(self.checkpoint, steps, batch, rate)
        self.popen.assert_not_called()
        self.jobs.start_training(self.checkpoint, 120, 4, 0.00005, uniform=True)
        command = self.popen.call_args.args[0]
        self.assertEqual(command[2:11], [str(self.root / "scripts/train_release_focus.py"),
            "--parent-checkpoint", str(self.checkpoint), "--steps", "120", "--batch-size", "4", "--learning-rate", "5e-05"])
        self.assertEqual(command[11], "--experiment")
        self.assertTrue(command[12].startswith("ui_uniform_"))
        self.assertEqual(command[-1], "--uniform-control")
        with self.assertRaises(RuntimeError):
            self.jobs.start_model(self.checkpoint)
        with self.assertRaises(RuntimeError):
            self.jobs.start_training(self.checkpoint, 120, 4, 1e-4)

    def test_reattach_then_stop_verified_process_and_preserve_files(self):
        started = self.jobs.start_model(self.checkpoint)
        reattached = lab_jobs.LabJobs(self.root)
        self.assertTrue(reattached.snapshot()["model"]["owned"])
        stopped = reattached.stop_model()
        self.processes[started["pid"]].terminate.assert_called_once()
        self.assertFalse(stopped["running"])
        self.assertEqual(stopped["exit_code"], 0)
        self.assertTrue(self.checkpoint.exists())
        self.assertTrue(Path(started["log"]).exists())
        self.assertEqual(json.loads(reattached.registry.read_text())["model"]["exit_code"], 0)

    def test_mismatched_identity_never_stops_or_owns_reused_pid(self):
        started = self.jobs.start_training(self.checkpoint, 10, 1, 0.001)
        process = self.processes[started["pid"]]
        for attribute, changed in (("created", 999), ("command", ["foreign"]), ("uid", os.getuid() + 1)):
            previous = getattr(process, attribute)
            setattr(process, attribute, changed)
            reattached = lab_jobs.LabJobs(self.root)
            self.assertFalse(reattached.snapshot()["training"]["owned"])
            reattached.stop_training()
            process.terminate.assert_not_called()
            process.kill.assert_not_called()
            setattr(process, attribute, previous)

    def test_kill_rechecks_identity_after_timeout(self):
        started = self.jobs.start_model(self.checkpoint)
        process = self.processes[started["pid"]]
        process.terminate.side_effect = None
        def reused(timeout):
            process.created += 1
            raise lab_jobs.psutil.TimeoutExpired(timeout)
        process.wait = reused
        self.jobs.stop_model()
        process.terminate.assert_called_once()
        process.kill.assert_not_called()

    def test_stubborn_verified_process_is_killed_with_bounded_wait(self):
        started = self.jobs.start_training(self.checkpoint, 10, 1, 0.001)
        process = self.processes[started["pid"]]
        process.terminate.side_effect = None
        process.wait = Mock(side_effect=[lab_jobs.psutil.TimeoutExpired(5), 0])
        self.jobs.stop_training()
        process.kill.assert_called_once()
        self.assertEqual([call.kwargs["timeout"] for call in process.wait.call_args_list], [5, 3])

    def test_tensorboard_arguments_and_log_tail(self):
        state = self.jobs.start_tensorboard()
        self.assertEqual(self.popen.call_args.args[0], [str(self.root / ".venv-sim/bin/python"),
            "-m", "tensorboard.main", "--logdir", str(self.root / ".local/tensorboard"),
            "--host", "127.0.0.1", "--port", "6006"])
        Path(state["log"]).write_text("prefix你好结尾")
        self.assertEqual(self.jobs.tail("tensorboard", 2), "结尾")
        self.assertEqual(self.jobs.tail("tensorboard", 0), "")
        self.jobs.start_tensorboard()
        self.assertEqual(self.popen.call_count, 1)

    def test_port_probe_only_binds_without_connect(self):
        with patch.object(lab_jobs.socket, "socket") as socket_class:
            probe = socket_class.return_value.__enter__.return_value
            # Use the original descriptor because normal tests replace this method.
            self.assertFalse(PORT_PROBE(8755))
            probe.bind.assert_called_once_with(("127.0.0.1", 8755))
            probe.connect.assert_not_called()
            probe.bind.side_effect = OSError(lab_jobs.errno.EADDRINUSE, "occupied")
            self.assertTrue(PORT_PROBE(8755))


PORT_PROBE = lab_jobs.LabJobs._port_busy


if __name__ == "__main__":
    unittest.main()
