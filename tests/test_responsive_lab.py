"""Control-loop regressions without a model, simulator or GPU."""
import csv
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import test_simulation_lab as legacy

from responsive_lab import ResponsiveSimulationLab


class ResponsiveTests(legacy.LabTests):
    def setUp(self):
        with patch.object(legacy, "SimulationLab", ResponsiveSimulationLab):
            super().setUp()

    def test_waiting_without_state_changes_does_not_rerender_cameras(self):
        self.lab.toggle_run()
        self.lab.tick()
        renders = self.lab.env.renders
        for _ in range(20):
            self.lab.tick()
        self.assertEqual(self.lab.env.renders, renders)
        self.assertEqual(self.lab.env.steps, 0)
        self.assertEqual(len(self.requests), 1)

    def test_unchanged_paused_scene_does_not_rerender_cameras(self):
        renders = self.lab.env.renders
        for _ in range(20):
            self.lab.tick()
        self.assertEqual(self.lab.env.renders, renders)

    def test_pending_single_step_does_not_rerender_or_apply_early(self):
        self.lab.single_step()
        self.lab.tick()
        renders = self.lab.env.renders
        for _ in range(20):
            self.lab.tick()
        self.assertEqual(self.lab.env.renders, renders)
        self.assertFalse(self.lab.env.applied)

    def test_native_change_while_waiting_is_still_recorded_and_discarded(self):
        self.lab.toggle_run()
        self.lab.tick()
        self.lab.env.qpos[0] += .01
        self.lab.tick()
        self.assertEqual(len(list(self.lab.record.directory.glob("native-edit-*.npy"))), 1)
        self.complete()
        self.lab.tick()
        self.assertFalse(self.lab.env.applied)

    def test_pacing_includes_inference_time_instead_of_adding_delay(self):
        self.lab.view.control_dt = .05
        self.lab.toggle_run()
        self.lab.tick()
        self.lab._request_started = 10.0
        self.complete()
        with patch("responsive_lab.time.monotonic", return_value=10.1):
            self.lab.tick()
        self.assertAlmostEqual(self.lab.next_step_time, 10.05)

    def test_timing_sidecar_preserves_original_action_contract(self):
        self.lab.toggle_run()
        self.lab.tick()
        self.complete()
        self.lab.tick()
        with (self.lab.record.directory / "timing.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertEqual(int(rows[0]["control_step"]), 1)
        self.assertEqual(len(self.lab.env.applied), 1)
        np.testing.assert_array_equal(self.lab.env.applied[0], np.ones(7))

    def test_completion_requires_sixty_stable_released_steps(self):
        self.lab.toggle_run()
        self.lab.obs["robot0_gripper_qpos"] = np.array([.04, -.04])
        action = np.array([0., 0, 0, 0, 0, 0, -1.])
        for _ in range(59):
            self.lab._check_completion(action, True)
        self.assertFalse(self.lab.finished)
        self.lab._check_completion(action, True)
        self.assertTrue(self.lab.finished)
        self.assertEqual(self.lab.record.reason, "stable_release_complete")

    def test_on_with_closed_command_never_auto_finishes(self):
        for _ in range(80):
            self.lab._check_completion(np.ones(7), True)
        self.assertFalse(self.lab.finished)

    def test_replaced_manual_record_does_not_inherit_policy_timings(self):
        self.lab.toggle_run()
        self.lab.tick()
        self.complete()
        self.lab.tick()
        manual = SimpleNamespace(result={})
        with patch.object(self.lab, "record", manual):
            self.lab._store_timing_summary()
            self.assertEqual(manual.result["timing_summary"]["steps"], 0)
            self.assertNotIn("performance_text", self.lab.state())

    def test_stable_streak_restarts_after_bowl_motion(self):
        action = np.array([0., 0, 0, 0, 0, 0, -1.])
        for _ in range(40):
            self.lab._check_completion(action, True)
        self.lab.obs[legacy.TARGET + "_pos"][0] += .006
        for _ in range(40):
            self.lab._check_completion(action, True)
        self.assertFalse(self.lab.finished)

    def test_completion_rechecks_fresh_geometry_before_stopping(self):
        action = np.array([0., 0, 0, 0, 0, 0, -1.])
        self.lab.env.eef = self.lab.env.qpos[:3] + [0., 0., .04]
        for _ in range(60):
            self.lab._check_completion(action, True)
        self.assertFalse(self.lab.finished)
