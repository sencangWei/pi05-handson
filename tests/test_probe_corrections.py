import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import probe_corrections as probe


class ProbeTests(unittest.TestCase):
    def test_five_of_six_gripper_samples_required(self):
        gt = np.array([0, 0, .1, 0, 0, 0, 1.])
        predictions = np.tile(gt, (6, 1))
        predictions[0, 6] = -1
        self.assertTrue(probe.check_predictions(gt, predictions, 2)["pass"])
        predictions[1, 6] = -1
        self.assertFalse(probe.check_predictions(gt, predictions, 2)["pass"])

    def test_movement_direction_and_magnitude(self):
        gt = np.array([0, -.1, 0, 0, 0, 0, -1.])
        for target, expected in [(-.1, True), (-.01, False), (.1, False)]:
            predictions = np.tile(gt, (6, 1))
            predictions[:, 1] = target
            self.assertEqual(probe.check_predictions(gt, predictions, 1)["pass"], expected)

    def test_gripper_only_has_no_xyz_restriction_and_hold_does(self):
        gt = np.array([0, 0, 0, 0, 0, 0, -1.])
        predictions = np.tile(gt, (6, 1))
        predictions[:, :3] = .2
        self.assertTrue(probe.check_predictions(gt, predictions, None)["pass"])
        self.assertFalse(probe.check_predictions(gt, predictions, -1)["pass"])
        predictions[:, :3] = .03
        self.assertTrue(probe.check_predictions(gt, predictions, -1)["pass"])

    def test_one_large_motion_cannot_hide_five_wrong_directions(self):
        gt = np.array([0, 0, .1, 0, 0, 0, 1.])
        predictions = np.tile(gt, (6, 1))
        predictions[:5, 2] = -.01
        predictions[5, 2] = 1.
        result = probe.check_predictions(gt, predictions, 2)
        self.assertGreater(result["mean"][2], .03)
        self.assertEqual(result["movement_correct"], 1)
        self.assertFalse(result["pass"])

    def test_opposing_large_actions_cannot_cancel_into_a_hold(self):
        gt = np.array([0, 0, 0, 0, 0, 0, -1.])
        predictions = np.tile(gt, (6, 1))
        predictions[::2, :3] = .5
        predictions[1::2, :3] = -.5
        result = probe.check_predictions(gt, predictions, -1)
        np.testing.assert_array_equal(result["mean"][:3], np.zeros(3))
        self.assertEqual(result["movement_correct"], 0)
        self.assertFalse(result["pass"])

    def test_fake_infer_preserves_preflipped_images_and_uses_first_action(self):
        image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        frame = {"state": np.arange(8), "action": np.array([0, 0, .1, 0, 0, 0, 1.]),
                 "image": image, "wrist_image": image + 1}
        calls = []

        def infer(observation):
            calls.append(observation)
            return {"actions": np.tile(frame["action"], (10, 1))}

        with patch.object(probe, "POINTS", [(15, "lift", 2)]):
            points = probe.probe_points(infer, {15: frame}, "task", lambda value: value.copy(), 6)
        self.assertTrue(points[0]["pass"])
        self.assertEqual(len(calls), 6)
        np.testing.assert_array_equal(calls[0]["observation/image"], image)
        np.testing.assert_array_equal(calls[0]["observation/wrist_image"], image + 1)
        self.assertEqual(calls[0]["prompt"], "task")

    def test_malformed_or_nonfinite_prediction_rejected(self):
        for predictions in [np.zeros((0, 7)), np.zeros((6, 8)), np.full((6, 7), np.nan)]:
            with self.assertRaises(ValueError):
                probe.check_predictions(np.ones(7), predictions, 2)

    def test_fake_infer_invalid_chunk_and_network_error_cannot_pass(self):
        frame = {"state": np.ones(8), "action": np.ones(7),
                 "image": np.zeros((2, 2, 3), dtype=np.uint8),
                 "wrist_image": np.zeros((2, 2, 3), dtype=np.uint8)}
        with patch.object(probe, "POINTS", [(15, "lift", 2)]):
            with self.assertRaises(ValueError):
                probe.probe_points(lambda _: {"actions": np.ones((1, 7))}, {15: frame}, "task", np.copy, 6)

            def disconnected(_):
                raise ConnectionError("server disconnected")

            with self.assertRaises(ConnectionError):
                probe.probe_points(disconnected, {15: frame}, "task", np.copy, 6)

    def test_metadata_checkpoint_and_hashes_must_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "assets").mkdir()
            (root / "assets/norm_stats.json").write_text("{}")
            (root / "params").mkdir()
            (root / "params/manifest.ocdbt").write_text("params")
            metadata = {"checkpoint": str(root), "task": "task", "action_horizon": 10,
                        "params_manifest_sha256": probe.digest(root / "params/manifest.ocdbt"),
                        "norm_stats_sha256": probe.digest(root / "assets/norm_stats.json")}
            probe.validate_model(metadata, root, "task")
            for key, value in [("checkpoint", str(root / "old")), ("params_manifest_sha256", "bad"),
                               ("norm_stats_sha256", "bad"), ("task", "other"), ("action_horizon", 5)]:
                with self.assertRaises(ValueError):
                    probe.validate_model(dict(metadata, **{key: value}), root, "task")

    def test_bad_source_saves_error_and_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "probe.json"
            (Path(temporary) / "metadata.json").write_text("{}")
            arguments = ["--checkpoint", temporary, "--output", str(output)]
            with patch.object(probe, "SOURCE", Path(temporary)), patch.object(probe, "SOURCE_SHA", "bad"):
                self.assertEqual(probe.main(arguments), 1)
            original = output.read_text()
            self.assertEqual(json.loads(original)["status"], "ERROR")
            with self.assertRaises(FileExistsError):
                probe.main(arguments)
            self.assertEqual(output.read_text(), original)


if __name__ == "__main__":
    unittest.main()
