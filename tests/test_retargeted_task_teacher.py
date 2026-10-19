"""Pure controller/geometry contracts; physical quality is checked on recorded pilot frames."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from generate_retargeted_task_teacher import load_reference, servo_action


def test_translation_is_world_scaled_and_norm_bounded():
    actual = np.zeros(8)
    goal = np.r_[.05, .05, 0, 0, 0, 0]
    action = servo_action(goal, actual, -1, .4)
    assert np.linalg.norm(action[:3]) == pytest.approx(.4)
    assert action[0] == pytest.approx(action[1])
    assert action[6] == -1
    assert np.array_equal(action[3:6], np.zeros(3))


def test_rotation_uses_short_world_so3_error_across_pi():
    actual = np.r_[np.zeros(3), 0, 0, -np.pi+.01, 0, 0]
    goal = np.r_[np.zeros(3), 0, 0, np.pi-.01]
    action = servo_action(goal, actual, 1, .4)
    assert action[5] == pytest.approx(-.04, abs=1e-6)
    reconstructed = Rotation.from_rotvec(action[3:6]*.5)*Rotation.from_rotvec(actual[3:6])
    assert (Rotation.from_rotvec(goal[3:6])*reconstructed.inv()).magnitude() < 1e-6


@pytest.mark.parametrize("goal,actual,grip,limit", [
    (np.zeros(5), np.zeros(8), 1, .4),
    (np.full(6, np.nan), np.zeros(8), 1, .4),
    (np.zeros(6), np.full(8, np.inf), 1, .4),
    (np.zeros(6), np.zeros(8), 0, .4),
    (np.zeros(6), np.zeros(8), 1, 0),
    (np.zeros(6), np.zeros(8), 1, 1.1),
])
def test_invalid_servo_contract_rejected(goal, actual, grip, limit):
    with pytest.raises(ValueError):
        servo_action(goal, actual, grip, limit)


def test_reference_selects_sustained_grasp_and_exact_scene_shift(tmp_path):
    old, new = tmp_path/"old", tmp_path/"new"
    old.mkdir()
    new.mkdir()
    scene = {"task_metadata": {"hash": "same"}, "task_language": "task", "layout_id": 24,
             "bowl_edit": {"after_xyz": [.04, .37, .90]}}
    (old/"scene.json").write_text(json.dumps(scene))
    scene["bowl_edit"]["after_xyz"] = [-.03, .37, .90]
    (new/"scene.json").write_text(json.dumps(scene))
    (old/"result.json").write_text(json.dumps({"success": True}))
    keys = [*[f"state_{i}" for i in range(8)], "action_6", *[f"post_bowl_{i}" for i in range(3)], "reward"]
    with (old/"actions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        for i in range(50):
            writer.writerow({**{f"state_{j}": j+i for j in range(8)}, "action_6": 1 if i >= 10 else -1,
                             "post_bowl_0": .04, "post_bowl_1": .37,
                             "post_bowl_2": .95 if i >= 25 else .90, "reward": int(i >= 40)})
    states, grasp, lift, on, delta, parsed = load_reference(old, new)
    assert (grasp, lift, on) == (16, 25, 40)
    assert states.shape == (50, 8)
    np.testing.assert_allclose(delta, [-.07, 0, 0])
    assert parsed == scene
    (old/"result.json").write_text(json.dumps({"success": False}))
    with pytest.raises(ValueError, match="successful"):
        load_reference(old, new)
