"""Isolated control-flow checks. No MuJoCo, policy service, or training is started."""
from concurrent.futures import Future
import contextlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from simulation_lab import LiberoTask, SimulationLab, TARGET, TrialRecord, move_bowl


class FakeEnv:
    def __init__(self, layout, calls, hit_at=None):
        self.calls, self.hit_at = calls, hit_at
        self.qpos = np.array([layout * 0.01, 0, 0.83, 1, 0, 0, 0], dtype=float)
        self.qvel = np.ones(6)
        self.eef = np.array([-0.2, 0, 1.2])
        self.steps, self.renders = 0, 0
        self.applied = []
        data = SimpleNamespace(get_joint_qpos=lambda joint: self.qpos,
                               get_joint_qvel=lambda joint: self.qvel,
                               set_joint_qpos=lambda joint, value: self.qpos.__setitem__(slice(None), value),
                               set_joint_qvel=lambda joint, value: self.qvel.__setitem__(slice(None), value))
        self.sim = SimpleNamespace(data=data)
        self.env = SimpleNamespace(objects_dict={TARGET: SimpleNamespace(joints=["bowl"])})

    def get_sim_state(self):
        return np.concatenate((self.qpos, self.qvel, self.eef, [self.steps]))

    def regenerate_obs_from_state(self, state):
        self.renders += 1
        return {f"{TARGET}_pos": self.qpos[:3].copy(), "robot0_eef_pos": self.eef.copy(),
                "robot0_eef_quat": np.array([0, 0, 0, 1]),
                "robot0_gripper_qpos": np.array([0.04, -0.04]),
                "agentview_image": np.full((16, 16, 3), self.renders % 256, dtype=np.uint8),
                "robot0_eye_in_hand_image": np.zeros((16, 16, 3), dtype=np.uint8)}

    def step(self, action):
        self.applied.append(action)
        self.eef[0] += action[0] * 0.01
        self.steps += 1
        done = self.hit_at is not None and self.steps >= self.hit_at
        return self.regenerate_obs_from_state(self.get_sim_state()), float(done), done, {}

    def close(self):
        self.calls.append("env_close")


class FakeView:
    def __init__(self, env):
        self.env = env
        self.paused, self.single_step, self.control_dt = True, False, 0
        self.handle = SimpleNamespace(sync=lambda: None)

    def is_running(self):
        return True

    def lock(self):
        return contextlib.nullcontext()

    def close(self):
        self.env.calls.append("view_close")


class FakeRecord:
    def __init__(self, directory, scene, obs, state):
        self.directory = directory
        directory.mkdir()
        self.result = {**scene, "control_steps": 0, "inference_calls": 0, "success": False,
                       "first_benchmark_success_control_steps": None, "model_metadata": None}
        self.saved = False

    def append(self, state, action, obs, reward, done):
        self.result["control_steps"] += 1

    def save(self, reason, state, obs=None):
        self.saved = True
        self.reason = reason


class LabTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.calls, self.envs = [], []
        self.hit_at = None
        def create(layout):
            env = FakeEnv(layout, self.calls, self.hit_at)
            self.envs.append(env)
            return env, env.regenerate_obs_from_state(env.get_sim_state())
        self.patch = patch("simulation_lab.TrialRecord", FakeRecord)
        self.patch.start()
        task = SimpleNamespace(layout_count=3, create=create)
        policy = SimpleNamespace(close=lambda: self.calls.append("policy_close"))
        self.lab = SimulationLab(Path(self.temporary.name) / "session", task, policy, FakeView)
        self.lab.reset(0)
        self.requests = []

        def submit(query, digest):
            future = Future()
            self.requests.append((query, future))
            self.lab.pending = future, self.lab.generation, digest
        self.lab._submit = submit

    def tearDown(self):
        self.lab.close()
        self.patch.stop()
        self.temporary.cleanup()

    def complete(self, future=None, shape=(10, 7)):
        if future is None:
            future = self.requests[-1][1]
        future.set_result(({"actions": np.ones(shape)}, {"model_stage": "fake-test-only"}))

    def test_paused_never_moves_and_single_step_uses_new_camera_observation(self):
        self.lab.tick()
        self.assertFalse(self.requests)
        self.lab.single_step()
        self.lab.tick()
        query, _ = self.requests[-1]
        self.assertEqual(query["observation/image"].shape, (224, 224, 3))
        self.assertEqual(query["observation/image"][100, 100, 0], self.lab.env.renders)
        self.complete()
        self.lab.tick()
        self.lab.tick()
        self.assertEqual(len(self.lab.env.applied), 1)
        self.assertTrue(self.lab.view.paused)

    def test_pause_discards_pending_result_and_resume_predicts_again(self):
        self.lab.toggle_run()
        self.lab.tick()
        self.lab.toggle_run()
        self.complete()
        self.lab.tick()
        self.assertFalse(self.lab.env.applied)
        self.assertIsNone(self.lab.pending)
        self.lab.toggle_run()
        self.lab.tick()
        self.assertEqual(len(self.requests), 2)
        self.complete()
        self.lab.tick()
        self.assertEqual(len(self.lab.env.applied), 1)

    def test_native_pause_discards_inflight_prediction(self):
        self.lab.toggle_run()
        self.lab.tick()
        self.lab.view.paused = True  # SPACE in the native viewer.
        self.complete()
        self.lab.tick()
        self.assertFalse(self.lab.env.applied)
        self.assertIsNone(self.lab.pending)

    def test_explicit_pause_cancels_both_single_step_flags(self):
        self.lab.single_step()
        self.lab.tick()
        self.lab.toggle_run()
        self.lab.view.single_step = True
        self.lab.toggle_run()
        self.complete()
        self.lab.tick()
        self.assertTrue(self.lab.view.paused)
        self.assertFalse(self.lab.step_requested)
        self.assertFalse(self.lab.view.single_step)
        self.assertFalse(self.lab.env.applied)

    def test_native_pause_after_single_step_then_run_cancels_old_step_intent(self):
        self.lab.single_step()
        self.lab.tick()
        self.lab.toggle_run()
        self.lab.view.paused = True
        self.complete()
        self.lab.tick()
        self.lab.tick()
        self.assertTrue(self.lab.view.paused)
        self.assertFalse(self.lab.step_requested)
        self.assertEqual(len(self.requests), 1)
        self.assertFalse(self.lab.env.applied)

    def test_reset_saves_first_closes_view_before_env_and_discards_old_prediction(self):
        self.lab.toggle_run()
        self.lab.tick()
        old = self.lab.record
        future = self.requests[-1][1]
        self.lab.reset(1, 0.02, -0.03)
        self.assertTrue(old.saved)
        self.assertEqual(self.calls[:2], ["view_close", "env_close"])
        np.testing.assert_allclose(self.lab.obs[f"{TARGET}_pos"], [0.03, -0.03, 0.83])
        self.complete(future)
        self.lab.toggle_run()
        self.lab.tick()
        self.assertFalse(self.lab.env.applied)
        self.assertIsNone(self.lab.pending)

    def test_manual_state_change_invalidates_waiting_prediction(self):
        self.lab.toggle_run()
        self.lab.tick()
        self.lab.env.qpos[0] += 0.01
        self.complete()
        self.lab.tick()
        self.assertFalse(self.lab.env.applied)
        self.assertIsNone(self.lab.pending)
        self.assertEqual(len(list(self.lab.record.directory.glob("native-edit-*.npy"))), 1)

    def test_invalid_prediction_pauses_without_executing(self):
        self.lab.toggle_run()
        self.lab.tick()
        self.complete(shape=(10, 8))
        with self.assertRaises(ValueError):
            self.lab.tick()
        self.assertTrue(self.lab.view.paused)
        self.assertFalse(self.lab.env.applied)

    def test_save_failure_preserves_current_scene_on_reset(self):
        view, env = self.lab.view, self.lab.env
        with patch.object(self.lab.record, "save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.lab.reset(1)
        self.assertIs(self.lab.view, view)
        self.assertIs(self.lab.env, env)
        self.assertFalse(self.calls)

    def test_on_signal_is_recorded_but_does_not_stop_unreleased_interactive_trial(self):
        self.lab.env.hit_at = 1
        self.lab.toggle_run()
        for _ in range(41):
            self.lab.tick()
            self.complete()
            self.lab.tick()
        self.assertEqual(len(self.lab.env.applied), 41)
        self.assertEqual(self.lab.record.result["post_success_steps_executed"], 40)
        self.assertFalse(self.lab.finished)
        self.assertFalse(self.lab.record.saved)
        for _ in range(179):
            self.lab.tick()
            self.complete()
            self.lab.tick()
        self.assertEqual(len(self.lab.env.applied), 220)
        self.assertEqual(self.lab.record.result["post_success_steps_executed"], 219)
        self.assertTrue(self.lab.finished)
        self.assertTrue(self.lab.record.saved)
        self.assertEqual(self.lab.record.reason, "step_limit")
        self.assertIn("松爪", self.lab.status)
        self.lab.tick()
        self.assertEqual(len(self.lab.env.applied), 220)

    def test_position_edit_preserves_z_and_rotation_and_clears_velocity(self):
        env = self.lab.env
        before = env.qpos.copy()
        obs, _ = move_bowl(env, self.lab.obs, 0.05, -0.04)
        np.testing.assert_array_equal(env.qpos[2:], before[2:])
        np.testing.assert_allclose(env.qvel, 0)
        np.testing.assert_allclose(obs[f"{TARGET}_pos"], before[:3] + [0.05, -0.04, 0])
        for value in (0.2, float("nan")):
            with self.assertRaises(ValueError):
                move_bowl(env, obs, value, 0)

    def test_invalid_reset_does_not_close_existing_scene(self):
        with self.assertRaises(ValueError):
            self.lab.reset(3)
        self.assertFalse(self.calls)
        self.assertFalse(self.lab.record.saved)
        operations = [json.loads(line) for line in (self.lab.output / "operations.jsonl").read_text().splitlines()]
        self.assertEqual(operations[0]["kind"], "reset")


class InteractiveHorizonTests(unittest.TestCase):
    def test_interactive_env_can_cross_old_horizon_without_losing_task_success_signal(self):
        class HorizonEnv:
            def __init__(self, **kwargs):
                self.ignore_done = kwargs.get("ignore_done", False)
                self.timestep, self.done = 0, False
            def seed(self, seed):
                pass
            def reset(self):
                pass
            def set_init_state(self, state):
                return {}
            def close(self):
                pass
            def step(self, action):
                if self.done:
                    raise ValueError("executing action in terminated episode")
                self.timestep += 1
                self.done = self.timestep >= 1000 and not self.ignore_done
                # LIBERO returns task On success separately from its internal time limit.
                return {}, 1.0, True, {}
        factory = unittest.mock.Mock(side_effect=HorizonEnv)
        task = LiberoTask.__new__(LiberoTask)
        task.bddl, task.initial_states = Path("fake-task.bddl"), [np.zeros(2)]
        modules = {"libero.libero.envs": SimpleNamespace(OffScreenRenderEnv=factory)}
        with patch.dict(sys.modules, modules):
            env, _ = task.create(0)
        for _ in range(1000):
            _, _, task_done, _ = env.step([0] * 7)
        self.assertEqual(env.timestep, 1010)
        self.assertTrue(factory.call_args.kwargs["ignore_done"])
        self.assertTrue(task_done)
        self.assertFalse(env.done)


class RecordTests(unittest.TestCase):
    def test_real_csv_json_png_and_video_export_is_idempotent(self):
        import imageio.v2 as imageio
        with tempfile.TemporaryDirectory() as temporary:
            env = FakeEnv(0, [])
            obs = env.regenerate_obs_from_state(env.get_sim_state())
            path = Path(temporary) / "record"
            record = TrialRecord(path, {"interactive_experiment": True}, obs, env.get_sim_state())
            record.append(np.zeros(8), np.zeros(7), obs, 0.0, False)
            record.save("isolated_export_check", env.get_sim_state(), obs)
            original = (path / "result.json").read_bytes()
            record.save("must_not_overwrite", env.get_sim_state(), obs)
            self.assertEqual(original, (path / "result.json").read_bytes())
            self.assertTrue(record.handle.closed)
            self.assertEqual(record.frames, [])
            with imageio.get_reader(path / "video.mp4") as reader:
                self.assertEqual(sum(1 for _ in reader), 2)
            self.assertTrue((path / "initial_state.npy").is_file())
            self.assertTrue((path / "final-agentview.png").is_file())


if __name__ == "__main__":
    unittest.main()
