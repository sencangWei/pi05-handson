import sys
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_release_focus import RepeatedFrames, install_sampling, sampling_indices


class ReleaseFocusTests(unittest.TestCase):
    def test_focus_has_context_and_never_crosses_episode_boundaries(self):
        indices, focused = sampling_indices([[-1, 1, 1, -1, -1], [-1, 1, -1]], multiplier=4, context=1)
        self.assertEqual(focused, [2, 3, 4, 6, 7])
        counts = Counter(indices)
        self.assertEqual({i: counts[i] for i in range(8)}, {0: 1, 1: 1, 2: 4, 3: 4, 4: 4, 5: 1, 6: 4, 7: 4})

    def test_uniform_control_preserves_original_order_and_all_frames(self):
        indices, _ = sampling_indices([[1, -1], [-1, 1, -1]], multiplier=1)
        self.assertEqual(indices, list(range(5)))

    def test_invalid_terminal_labels_are_rejected(self):
        for episodes in ([], [[-1, -1]], [[-1, 1]], [[1, float("nan"), -1]]):
            with self.assertRaises(ValueError):
                sampling_indices(episodes)

    def test_sampling_changes_only_access_not_future_action_chunk(self):
        dataset = [{"actions": [[i, t] for t in range(10)]} for i in range(3)]
        repeated = RepeatedFrames(dataset, [0, 1, 2, 2])
        self.assertIs(repeated[3], dataset[2])
        self.assertEqual(len(repeated), 4)
        with self.assertRaises(ValueError):
            RepeatedFrames(dataset, [3])

    def test_installation_preserves_original_create_signature(self):
        calls = []
        def create(config, horizon, model):
            calls.append((config, horizon, model))
            return ["first", "second"]
        loader = SimpleNamespace(create_torch_dataset=create)
        original = install_sampling(loader, [0, 1, 1], 2)
        config = SimpleNamespace(repo_id="course/libero_black_bowl")
        dataset = loader.create_torch_dataset(config, 10, "model")
        self.assertIs(original, create)
        self.assertEqual(dataset[2], "second")
        self.assertEqual(calls, [(config, 10, "model")])
        with self.assertRaises(ValueError):
            loader.create_torch_dataset(config, 50, "model")


if __name__ == "__main__":
    unittest.main()
