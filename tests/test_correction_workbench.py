"""Real correction recorder and backend, with fake physics and process jobs."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from correction_workbench import CorrectionWorkbench
from lab_jobs import LabJobs
from test_simulation_lab import FakeEnv, FakeRecord, FakeView


class ManualEnv(FakeEnv):
    def regenerate_obs_from_state(self, state):
        obs = super().regenerate_obs_from_state(state)
        # Deliberate mutable alias: the stored BEFORE state must survive env.step.
        obs["robot0_eef_pos"] = self.eef
        obs["agentview_image"] = np.full((256, 256, 3), self.renders % 256, dtype=np.uint8)
        obs["robot0_eye_in_hand_image"] = np.full((256, 256, 3), 17, dtype=np.uint8)
        obs["robot0_gripper_qpos"] = np.array([.04, -.04]) if getattr(self, "opened", True) else np.array([.002, -.002])
        return obs

    def step(self, action):
        self.applied.append(action)
        self.eef += np.asarray(action[:3]) * .05
        self.opened = action[6] < 0
        self.steps += 1
        return self.regenerate_obs_from_state(self.get_sim_state()), 1.0, True, {}


class CorrectionBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.calls = []
        def create(layout):
            env = ManualEnv(layout, self.calls)
            return env, env.regenerate_obs_from_state(env.get_sim_state())
        self.jobs = Mock(spec=LabJobs)
        self.job_state = {"model": {"owned": False, "running": False},
                          "training": {"owned": False, "running": False}, "port_busy": False}
        self.jobs.snapshot.side_effect = lambda: self.job_state
        self.jobs.validate_training.side_effect = lambda *args: args
        self.jobs.start_correction_training.return_value = {"log": "fake.log"}
        self.patch_record = patch("simulation_lab.TrialRecord", FakeRecord)
        self.patch_record.start()
        self.lab = CorrectionWorkbench(self.root / "evaluations/session", SimpleNamespace(layout_count=3, create=create),
                                       root=self.root, jobs=self.jobs, viewer_factory=FakeView)
        self.lab.reset(0)
        self.lab.view.control_dt = .05

    def tearDown(self):
        self.lab.close()
        self.patch_record.stop()
        self.temp.cleanup()

    def test_takeover_archives_policy_and_gates_all_model_scene_actions(self):
        old_record = self.lab.record
        self.lab.begin_correction()
        self.assertTrue(old_record.saved)
        self.assertTrue(self.lab.finished)
        self.assertTrue(self.lab.correction_active)
        self.assertTrue(self.lab.view.paused)
        for action in (self.lab.toggle_run, self.lab.single_step, lambda: self.lab.reset(1),
                       lambda: self.lab.start_training("parent", 10, 1, 1e-4)):
            with self.assertRaises(ValueError):
                action()
        self.lab.view.paused = False  # Native SPACE cannot resume the model.
        self.lab.tick()
        self.assertFalse(self.lab.env.applied)
        self.assertTrue(self.lab.view.paused)

    def test_before_state_images_and_action_remain_aligned_after_mutating_step(self):
        before = self.lab.obs["robot0_eef_pos"].copy()
        self.lab.begin_correction()
        self.lab.manual_jog(0, 1, .1, 1)
        with np.load(self.lab.correction.directory / "frame-000000.npz", allow_pickle=False) as frame:
            np.testing.assert_allclose(frame["state"][:3], before)
            np.testing.assert_allclose(frame["post_state"][0] - frame["state"][0], .005, atol=1e-7)
            np.testing.assert_allclose(frame["action"], [.1, 0, 0, 0, 0, 0, -1])
            self.assertEqual(frame["image"].shape, (256, 256, 3))
        self.assertEqual(self.lab.record.result["control_steps"], 0)
        self.assertEqual(self.lab.correction.count, 1)

    def test_rejected_approval_can_continue_then_open_hold_and_save(self):
        self.lab.begin_correction()
        self.lab.manual_gripper(False)
        with self.assertRaises(ValueError):
            self.lab.finish_correction(True)
        self.assertTrue(self.lab.correction_active)
        self.lab.manual_gripper(True)
        self.lab.manual_hold()
        directory = self.lab.finish_correction(True)
        self.assertEqual(self.lab.correction.count, 30)
        self.assertEqual(json.loads((directory / "metadata.json").read_text())["status"], "approved")
        self.assertFalse(self.lab.correction_active)
        self.assertTrue((directory / "final-agentview.png").is_file())
        self.assertTrue((directory / "final-wrist.png").is_file())
        self.assertEqual(self.lab.correction_state()["approved_train"], 1)

    def test_native_edit_after_last_recorded_action_cannot_be_approved(self):
        self.lab.begin_correction()
        self.lab.manual_gripper(True)
        self.lab.manual_hold()
        self.lab.env.qpos[0] += .02
        with self.assertRaises(ValueError):
            self.lab.finish_correction(True)
        self.assertFalse(self.lab.correction_active)
        metadata = json.loads((self.lab.correction.directory / "metadata.json").read_text())
        self.assertEqual(metadata["status"], "draft")

    def test_native_physical_edit_drafts_instead_of_training_corrupted_labels(self):
        self.lab.begin_correction()
        self.lab.env.qpos[0] += .01
        with self.assertRaises(ValueError):
            self.lab.manual_hold()
        self.assertEqual(self.lab.correction.count, 0)
        self.assertFalse(self.lab.correction_active)
        metadata = json.loads((self.lab.correction.directory / "metadata.json").read_text())
        self.assertEqual(metadata["status"], "draft")
        self.assertFalse(self.lab.env.applied)

    def test_invalid_manual_parameters_never_step(self):
        self.lab.begin_correction()
        for action in (lambda: self.lab.manual_jog(7, 1, .1, 1),
                       lambda: self.lab.manual_jog(0, 1, float("nan"), 1),
                       lambda: self.lab.manual_jog(0, 1, .1, 21)):
            with self.assertRaises(ValueError):
                action()
        self.assertFalse(self.lab.env.applied)

    def test_close_preserves_unapproved_draft_and_never_stops_external_jobs(self):
        self.lab.begin_correction(split="holdout")
        directory = self.lab.correction.directory
        self.lab.manual_hold(1)
        self.lab.close()
        self.assertEqual(json.loads((directory / "metadata.json").read_text())["status"], "draft")
        self.jobs.stop_model.assert_not_called()
        self.jobs.stop_training.assert_not_called()

    def test_finished_policy_scene_can_be_taken_over(self):
        self.lab.save("step_limit")
        self.lab.begin_correction()
        self.lab.manual_gripper(True, repeats=1)
        self.assertEqual(self.lab.correction.count, 1)

    def test_correction_bundle_training_routes_to_separate_script(self):
        evidence = self.root / "evidence"
        evidence.mkdir()
        manifest = evidence / "corrections-test-manifest.json"
        manifest.write_text(json.dumps({"ready": True}))
        self.assertEqual(self.lab.corrections_bundles(), [str(manifest)])
        self.lab.start_training("parent", 10, 2, 1e-4, uniform=True, correction_manifest=manifest)
        self.jobs.start_correction_training.assert_called_once_with("parent", 10, 2, 1e-4, manifest)
        self.jobs.start_training.assert_not_called()

    def test_prepare_only_starts_bounded_cpu_data_command_on_explicit_request(self):
        result = SimpleNamespace(returncode=0, stdout="ready", stderr="")
        with patch("correction_workbench.subprocess.run", return_value=result) as run:
            run.assert_not_called()
            self.lab.prepare_corrections()
            self.lab.background["correction_prepare"].result(timeout=2)
            self.lab._collect_background()
        command = run.call_args.args[0]
        self.assertIn("scripts/prepare_corrections.py", command)
        self.assertEqual(run.call_args.kwargs["env"]["JAX_PLATFORMS"], "cpu")
        self.assertIn("混合数据已生成", self.lab.correction_status)


if __name__ == "__main__":
    unittest.main()
