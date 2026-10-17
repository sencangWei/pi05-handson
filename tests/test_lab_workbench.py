"""Workbench integration without starting models, training or a simulator."""
from concurrent.futures import Future
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lab_workbench import LabWorkbench
from lab_jobs import LabJobs
from test_simulation_lab import FakeEnv, FakeView, FakeRecord


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "checkpoints/pi05_handson_lora/focus/1999"
        for name in ("params/manifest.ocdbt", "assets/course/libero_black_bowl/norm_stats.json"):
            path = self.checkpoint / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
        self.jobs = Mock(spec=LabJobs)
        self.status = {"model": {"running": False, "owned": False},
                       "training": {"running": False, "owned": False}, "port_busy": False}
        self.jobs.snapshot.side_effect = lambda: self.status
        self.jobs.validate_checkpoint.side_effect = LabJobs(self.root).validate_checkpoint
        self.jobs.validate_training.side_effect = LabJobs(self.root).validate_training
        self.jobs.start_training.return_value = {"log": "fake-training.log"}
        self.calls = []
        def create(layout):
            env = FakeEnv(layout, self.calls)
            return env, env.regenerate_obs_from_state(env.get_sim_state())
        self.record_patch = patch("simulation_lab.TrialRecord", FakeRecord)
        self.record_patch.start()
        self.lab = LabWorkbench(self.root / "evaluations/session", SimpleNamespace(layout_count=3, create=create),
                                root=self.root, jobs=self.jobs, viewer_factory=FakeView)

    def tearDown(self):
        self.lab.close()
        self.record_patch.stop()
        self.temp.cleanup()

    def result(self, **data):
        path = self.root / "evaluations/old/result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"success": True, "initial_state_id": 3, "control_steps": 149, **data}))
        return path

    def test_direct_gui_entry_is_single_instance_before_task_or_gpu_load(self):
        import fcntl
        import simulation_lab
        lock_path = self.root / ".local/lab-panel.lock"
        lock_path.parent.mkdir()
        with lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(simulation_lab, "ROOT", self.root), patch.object(sys, "argv", ["simulation_lab.py"]):
                with patch.object(simulation_lab, "run_gui") as gui, patch("lab_launcher.notify") as notify:
                    simulation_lab.main()
                    gui.assert_not_called()
                    notify.assert_called_once()
        def run(port):
            with lock_path.open("a+") as other:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with patch.object(simulation_lab, "ROOT", self.root), patch.object(sys, "argv", ["simulation_lab.py"]):
            with patch.object(simulation_lab, "run_gui", side_effect=run) as gui:
                simulation_lab.main()
                gui.assert_called_once_with(8755)

    def test_foreign_service_never_stopped_or_replaced(self):
        self.status["port_busy"] = True
        for action in (lambda: self.lab.start_model(self.checkpoint),
                       lambda: self.lab.start_training(self.checkpoint, 2, 1, 1e-4), self.lab.stop_model):
            with self.assertRaises(ValueError):
                action()
        self.jobs.stop_model.assert_not_called()
        self.jobs.start_model.assert_not_called()
        self.jobs.start_training.assert_not_called()

    def test_invalid_inputs_preserve_scene_and_running_model(self):
        self.lab.reset(0)
        old = self.lab.env
        with self.assertRaises(ValueError):
            self.lab.start_model(self.root / "missing")
        with self.assertRaises(ValueError):
            self.lab.start_training(self.checkpoint, 0, 1, 1e-4)
        self.assertIs(self.lab.env, old)
        self.jobs.stop_model.assert_not_called()

    def test_explicit_training_saves_scene_before_job_and_forwards_parameters(self):
        self.lab.reset(0)
        record = self.lab.record
        self.lab.start_training(self.checkpoint, 7, 2, 1e-4, uniform=True)
        self.assertTrue(record.saved)
        self.assertIsNone(self.lab.env)
        self.assertEqual(self.calls[:2], ["view_close", "env_close"])
        self.jobs.start_training.assert_called_once_with(self.checkpoint, 7, 2, 1e-4, True)

    def test_training_motion_guard_reads_fresh_job_state(self):
        self.lab.reset(0)
        self.status["training"] = {"running": True, "owned": True}
        for action in (self.lab.toggle_run, self.lab.single_step, lambda: self.lab.reset(1)):
            with self.assertRaises(ValueError):
                action()
        self.assertTrue(self.lab.view.paused)
        self.assertFalse(self.lab.env.applied)

    def test_switch_model_archives_scene_and_closes_old_connection(self):
        self.lab.reset(0)
        self.lab.policy = Mock()
        old_policy = self.lab.policy
        self.lab.start_model(self.checkpoint)
        old_policy.close.assert_called_once()
        self.jobs.stop_model.assert_called_once()
        self.jobs.start_model.assert_called_once_with(self.checkpoint)
        self.assertTrue(self.lab.record.saved)
        self.assertIsNone(self.lab.view)

    def test_result_without_video_is_visible_and_review_preserves_official_result(self):
        path = self.result(video=None)
        before = path.read_bytes()
        self.assertEqual(self.lab.results()[0]["layout"], 3)
        self.lab.review_result(path, False, "夹爪未松开")
        row = self.lab.results()[0]
        self.assertEqual(row["review"], "收尾失败")
        self.assertTrue(row["on"])
        self.assertEqual(before, path.read_bytes())
        self.lab.review_result(path, None, "待看")
        self.assertEqual(self.lab.results()[0]["review"], "待复核")

    def test_modified_result_expires_manual_review(self):
        path = self.result()
        self.lab.review_result(path, True, "checked")
        path.write_text(path.read_text() + "\n")
        self.assertEqual(self.lab.results()[0]["review"], "复核过期")

    def test_result_and_video_path_escape_are_refused(self):
        other = self.root / "result.json"
        other.write_text("{}")
        with self.assertRaises(ValueError):
            self.lab.review_result(other, True, "")
        outside = self.root / "outside.mp4"
        outside.touch()
        path = self.result(video="../../outside.mp4")
        self.assertIsNone(self.lab.results()[0]["video"])
        with self.assertRaises(ValueError):
            self.lab.play_result(path)

    def test_play_opens_saved_video_only_on_explicit_call(self):
        path = self.result(video="video.mp4")
        video = path.parent / "video.mp4"
        video.touch()
        with patch.object(self.lab, "_open") as opened:
            self.lab.results()
            opened.assert_not_called()
            self.lab.play_result(path)
            opened.assert_called_once_with(video)

    def test_background_metadata_probe_does_not_request_an_action(self):
        policy = Mock()
        policy.metadata_only.return_value = {"model_stage": "test-only"}
        with patch("lab_workbench.PolicyConnection", return_value=policy):
            self.lab.check_connection()
            self.lab.background["connection"].result(timeout=2)
            self.lab._collect_background()
        policy.metadata_only.assert_called_once()
        policy.infer.assert_not_called()
        policy.close.assert_called_once()
        self.assertEqual(self.lab.model_name, "test-only")

    def test_curve_snapshot_exports_real_step_values_without_starting_tensorboard(self):
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except ImportError:
            self.skipTest("TensorBoard unavailable")
        path = self.root / "logs/training-test.log"
        path.parent.mkdir()
        path.write_text("Step 0: loss=0.5, grad_norm=0.2, param_norm=1.0\nStep 10: loss=0.25, grad_norm=0.1, param_norm=1.0\n")
        with patch("lab_workbench.urllib.request.urlopen"), patch("lab_workbench.webbrowser.open"):
            self.lab.open_curves()
            result = self.lab.background["curves"].result(timeout=8)
            self.lab._collect_background()
        self.jobs.start_tensorboard.assert_called_once()
        accumulator = EventAccumulator(result["output"]).Reload()
        values = accumulator.Scalars("train/loss")
        self.assertEqual([(v.step, v.value) for v in values], [(0, 0.5), (10, 0.25)])
        snapshot = json.loads((Path(result["output"]) / "snapshot.json").read_text())
        self.assertEqual(snapshot["source_sha256"], self.lab._digest(path))

    def test_curve_completion_opens_browser_only_after_background_success(self):
        future = Future()
        self.lab.background["curves"] = future
        with patch("lab_workbench.webbrowser.open") as browser:
            self.lab._collect_background()
            browser.assert_not_called()
            future.set_result({"source": "fake.log", "url": "http://127.0.0.1:6006"})
            self.lab._collect_background()
            browser.assert_called_once_with("http://127.0.0.1:6006")


if __name__ == "__main__":
    unittest.main()
