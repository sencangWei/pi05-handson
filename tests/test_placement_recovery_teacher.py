from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generate_placement_recovery_teacher import box_vertical_bounds, position_action, released, terminal_gates


def frame():
    return {"action": np.r_[np.zeros(6), -1.], "post_state": np.r_[0., 0., .15, np.zeros(3), .04, -.04],
            "post_bowl_pos": np.zeros(3), "done": np.array(True)}


class PlacementTeacherTests(unittest.TestCase):
    def test_translation_norm_sign_scale_and_rotation(self):
        a = position_action([.1, -.1, .1], 1, .15)
        self.assertAlmostEqual(float(np.linalg.norm(a[:3])), .15, places=7)
        np.testing.assert_array_equal(np.sign(a[:3]), [1, -1, 1])
        np.testing.assert_array_equal(a[3:6], 0)
        self.assertEqual(a[6], 1)
        np.testing.assert_allclose(position_action([.001, 0, 0], -1, .15)[:3], [.02, 0, 0])

    def test_invalid_motion_is_rejected(self):
        for error, limit in [([np.nan, 0, 0], .2), ([0, 0], .2), ([0, 0, 0], 2)]:
            with self.assertRaises(ValueError):
                position_action(error, -1, limit)

    def test_rotated_box_world_vertical_extent(self):
        r = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        self.assertEqual(box_vertical_bounds([0, 0, 1], r, [.1, .2, .3]), (.9, 1.1))

    def test_on_alone_closed_command_and_actual_gap_cannot_pass(self):
        f = frame()
        self.assertTrue(released(f))
        f["action"][6] = 1
        self.assertFalse(released(f))
        f["action"][6] = -1
        f["post_state"][6:8] = [.001, -.001]
        self.assertFalse(released(f))
        f["post_state"][6:8] = [.04, -.04]
        f["post_state"][2] = .05
        self.assertFalse(released(f))

    def test_exact_fifty_followup_and_continuity_required(self):
        frames = [frame() for _ in range(60)]
        self.assertTrue(all(terminal_gates(frames, 10, True, False).values()))
        self.assertFalse(all(terminal_gates(frames[:-1], 10, True, False).values()))
        self.assertFalse(all(terminal_gates(frames, 10, False, False).values()))
        self.assertFalse(all(terminal_gates(frames, 10, True, True).values()))

    def test_unstable_or_intermittent_followup_rejected(self):
        frames = [frame() for _ in range(60)]
        frames[20]["post_bowl_pos"][0] = .006
        self.assertFalse(terminal_gates(frames, 10, True, False)["followup_bowl_motion_le_5mm"])
        frames[20] = frame()
        frames[20]["done"] = np.array(False)
        self.assertFalse(terminal_gates(frames, 10, True, False)["followup_all_released"])


if __name__ == "__main__":
    unittest.main()
