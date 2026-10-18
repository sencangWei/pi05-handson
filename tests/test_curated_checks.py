from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from curated_checks import curated_clip_checks


def lift_frames():
    frames = []
    for number in range(20):
        state = np.array([0., 0., .9 + number * .005, 0., 0., 0., .001, -.001])
        post = state.copy()
        post[2] += .005
        frames.append({"state": state, "post_state": post, "post_bowl_pos": np.array([0., 0., .85]),
                       "action": np.array([0., 0., .1, 0., 0., 0., 1.]), "done": np.array(False)})
    return frames


def regrasp_frames():
    frames = []
    for number in range(25):
        height = .9 + max(0, number - 4) * .002
        bowl = np.array([0., 0., height])
        post = np.array([0., 0., height + .05, 0., 0., 0., .04, -.04])
        if number >= 5:
            post[6:] = [.01, -.01]
        before = post.copy()
        before[2] -= .002
        frames.append({"state": before, "post_state": post, "post_bowl_pos": bowl,
                       "action": np.array([0., 0., .1, 0., 0., 0., -1. if number < 5 else 1.]),
                       "done": np.array(False)})
    return frames


class CuratedCheckTests(unittest.TestCase):
    def test_partial_phases_pass_without_placement_success(self):
        for kind, frames in (("lift_clearance", lift_frames()), ("regrasp_lift", regrasp_frames())):
            result = curated_clip_checks(kind, frames)
            self.assertTrue(all(result["checks"].values()))
            self.assertFalse(result["official_task_success"])
            with self.assertRaises(ValueError):
                curated_clip_checks("placement_release", frames)

    def test_lifting_the_bowl_or_no_eef_clearance_is_not_disengagement(self):
        original = lift_frames()
        for fault in ("bowl_up", "no_lift", "not_clear", "unstable_bowl", "down_command"):
            frames = deepcopy(original)
            if fault == "bowl_up":
                frames[-1]["post_bowl_pos"][2] += .02
            elif fault == "no_lift":
                frames[-1]["post_state"][2] = frames[0]["state"][2]
            elif fault == "not_clear":
                frames[-1]["post_bowl_pos"] = frames[-1]["post_state"][:3].copy()
            elif fault == "unstable_bowl":
                frames[-2]["post_bowl_pos"][0] += .01
            else:
                frames[0]["action"][2] = -.1
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                curated_clip_checks("lift_clearance", frames)

    def test_empty_close_no_open_or_bowl_left_behind_fail_regrasp(self):
        original = regrasp_frames()
        for fault in ("empty_lift", "no_open", "no_close", "fingers_open", "slip", "eef_not_lifted"):
            frames = deepcopy(original)
            if fault == "empty_lift":
                for frame in frames:
                    frame["post_bowl_pos"][2] = .9
            elif fault == "no_open":
                frames[0]["action"][6] = 1.
            elif fault == "no_close":
                frames[-1]["action"][6] = -1.
            elif fault == "fingers_open":
                frames[-1]["post_state"][6:] = [.04, -.04]
            elif fault == "slip":
                frames[-1]["post_bowl_pos"][0] += .03
            else:
                frames[-1]["post_state"][2] = frames[5]["state"][2]
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                curated_clip_checks("regrasp_lift", frames)

    def test_invalid_frames_are_rejected(self):
        with self.assertRaises(ValueError):
            curated_clip_checks("unknown", lift_frames())
        with self.assertRaises(ValueError):
            curated_clip_checks("lift_clearance", lift_frames()[:9])
        frames = lift_frames()
        frames[-1]["post_state"][2] = np.nan
        with self.assertRaises(ValueError):
            curated_clip_checks("lift_clearance", frames)
        frames[-1]["post_state"] = np.zeros(7)
        with self.assertRaises(ValueError):
            curated_clip_checks("lift_clearance", frames)


if __name__ == "__main__":
    unittest.main()
