"""Validate intervention boundaries without starting MuJoCo, a model, or hardware."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_libero import arguments, shift_ungrasped_target


class TargetShiftTests(unittest.TestCase):
    def environment(self):
        data = SimpleNamespace(qpos=np.array([0.01, 0.3, 0.83, 1.0, 0.0, 0.0, 0.0]),
                               qvel=np.ones(6), mutations=0)
        data.get_joint_qpos = lambda joint: data.qpos
        data.get_joint_qvel = lambda joint: data.qvel

        def set_position(joint, value):
            self.assertEqual(joint, "target_joint")
            data.qpos[:] = value
            data.mutations += 1

        def set_velocity(joint, value):
            self.assertEqual(joint, "target_joint")
            data.qvel[:] = value

        data.set_joint_qpos = set_position
        data.set_joint_qvel = set_velocity
        obs = {"akita_black_bowl_1_pos": data.qpos[:3].copy(),
               "robot0_eef_pos": np.array([-0.1, 0.1, 1.2]),
               "robot0_gripper_qpos": np.array([0.04, -0.04])}
        env = SimpleNamespace(sim=SimpleNamespace(data=data),
                              env=SimpleNamespace(objects_dict={
                                  "akita_black_bowl_1": SimpleNamespace(joints=["target_joint"])}))
        env.get_sim_state = lambda: data.qpos.copy()
        env.regenerate_obs_from_state = lambda state: dict(obs, akita_black_bowl_1_pos=state[:3].copy())
        return env, obs

    def test_only_world_x_changes_and_new_observation_is_returned(self):
        env, obs = self.environment()
        original = env.sim.data.qpos.copy()
        fresh, event = shift_ungrasped_target(env, obs, 0.05)
        np.testing.assert_allclose(env.sim.data.qpos[1:], original[1:])
        np.testing.assert_allclose(env.sim.data.qvel, 0)
        np.testing.assert_allclose(fresh["akita_black_bowl_1_pos"] - obs["akita_black_bowl_1_pos"], [0.05, 0, 0])
        np.testing.assert_allclose(event["actual_delta_xyz"], [0.05, 0, 0])

    def test_bowl_near_hand_is_rejected_before_any_mutation(self):
        env, obs = self.environment()
        obs["robot0_eef_pos"] = obs["akita_black_bowl_1_pos"].copy()
        with self.assertRaisesRegex(ValueError, "open gripper"):
            shift_ungrasped_target(env, obs, 0.05)
        self.assertEqual(env.sim.data.mutations, 0)

    def test_closed_gripper_is_rejected_before_any_mutation(self):
        env, obs = self.environment()
        obs["robot0_gripper_qpos"] = np.array([0.002, -0.002])
        with self.assertRaises(ValueError):
            shift_ungrasped_target(env, obs, 0.05)
        self.assertEqual(env.sim.data.mutations, 0)

    def test_existing_command_has_no_intervention(self):
        with patch.object(sys, "argv", ["evaluate_libero.py", "--output", "unused"]):
            self.assertIsNone(arguments().target_shift_step)


if __name__ == "__main__":
    unittest.main()
