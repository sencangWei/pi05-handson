from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generate_tracking_teacher import GATE_NAMES, frame_action, quality_gates, reference_schedule, tracking_action, world_rotation_error


class TrackingMathTests(unittest.TestCase):
    def test_world_so3_left_error_noncommuting_rotations(self):
        current = Rotation.from_euler("z", 90, degrees=True)
        target = Rotation.from_rotvec([.2, 0, 0])*current
        error = world_rotation_error(target.as_rotvec(), current.as_rotvec())
        np.testing.assert_allclose(error, [.2, 0, 0], atol=1e-12)
        np.testing.assert_allclose((Rotation.from_rotvec(error)*current).as_matrix(), target.as_matrix(), atol=1e-12)

    def test_rotvec_equivalent_representations_have_zero_error(self):
        np.testing.assert_allclose(world_rotation_error([0, 0, 2*np.pi+.1], [0, 0, .1]), 0, atol=1e-12)

    def test_schedule_protects_boundaries_and_covers_reference(self):
        schedule = reference_schedule(796, (390, 550))
        self.assertEqual(len(schedule), 726)
        self.assertEqual(schedule[0], (0, 1))
        self.assertEqual(schedule[-1], (795, 1))
        self.assertTrue(all(rate == 1 for index, rate in schedule if not 400 <= index < 540))
        self.assertTrue(all(rate == 2 for index, rate in schedule if 400 <= index < 540))
        self.assertTrue(all(b[0]-a[0] == a[1] for a, b in zip(schedule, schedule[1:])))
        self.assertEqual(sum(390 <= i < 550 for i, _ in schedule), 90)

    def test_feedforward_feedback_scale_clip_and_gripper_preserved(self):
        actual = np.zeros(8)
        reference = actual.copy()
        reference[0] = .005
        reference[5] = .05
        source = np.array([.1, 0, 0, 0, 0, 0, -1.])
        command, error = tracking_action(source, reference, actual, 2, np.array([.05]*3+[.5]*3))
        np.testing.assert_allclose(command, [.3, 0, 0, 0, 0, .1, -1.], atol=1e-7)
        self.assertAlmostEqual(error[0], .005)
        reference[0] = 1
        command, _ = tracking_action(source, reference, actual, 2, np.array([.05]*3+[.5]*3))
        self.assertEqual(command[0], 1)

    def test_nonfinite_tracking_rejected(self):
        with self.assertRaises(ValueError):
            tracking_action(np.full(7, np.nan), np.zeros(8), np.zeros(8), 2, np.ones(6))

    def test_rate_one_keeps_all_source_reference_indices(self):
        schedule = reference_schedule(735, (341, 495), rate=1)
        self.assertEqual(schedule, [(i, 1) for i in range(735)])

    def test_feedback_cannot_modify_actions_outside_carry(self):
        original = np.array([.03, -.02, .1, .01, 0, 0, -1.], dtype=np.float32)
        reference, actual = np.ones(8), np.zeros(8)
        for index in [0, 340, 495, 734]:
            action, error = frame_action(original, reference, actual, 1, np.ones(6), index, (341, 495))
            np.testing.assert_array_equal(action, original)
            np.testing.assert_array_equal(error, np.zeros(6))
        carry_action, _ = frame_action(original, reference, actual, 1, np.ones(6), 341, (341, 495))
        self.assertFalse(np.array_equal(carry_action, original))
        with self.assertRaises(ValueError):
            frame_action(original, reference, actual, 2, np.ones(6), 0, (341, 495))


class QualityTests(unittest.TestCase):
    def metrics(self):
        return {"terminal_on_last_ten": True, "terminal_min_gap_m": .08,
                "terminal_open_command_last_ten": True, "final_retreat_m": .15,
                "carry_position_error_p95_m": .004, "carry_orientation_error_p95_deg": 2.,
                "carry_min_bowl_height_delta_to_reference_m": -.01, "carry_relative_bowl_eef_error_max_m": .01,
                "carry_control_step_speedup": 1.77, "continuous_physics_trajectory": True,
                "actions_finite_and_bounded": True, "reference_coverage_complete": True}

    def test_all_gates_required_not_terminal_only(self):
        base = self.metrics()
        self.assertEqual(set(quality_gates(base)), set(GATE_NAMES))
        self.assertTrue(all(quality_gates(base).values()))
        for key, value in [("carry_position_error_p95_m", .006), ("carry_orientation_error_p95_deg", 4),
                           ("carry_min_bowl_height_delta_to_reference_m", -.02),
                           ("carry_relative_bowl_eef_error_max_m", .02), ("carry_control_step_speedup", 1.4),
                           ("terminal_min_gap_m", .05), ("final_retreat_m", .09), ("terminal_on_last_ten", False)]:
            with self.subTest(key=key):
                self.assertFalse(all(quality_gates(dict(base, **{key: value})).values()))

    def test_speed_gate_is_role_aware_without_fake_rate_one_speedup(self):
        metrics = dict(self.metrics(), carry_control_step_speedup=1.)
        self.assertTrue(all(quality_gates(metrics, rate=1).values()))
        self.assertFalse(all(quality_gates(metrics, rate=2).values()))


if __name__ == "__main__":
    unittest.main()
