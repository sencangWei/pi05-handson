import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_training_curves import parse_metrics
from live_preview import LivePreview


class VisualToolTests(unittest.TestCase):
    def test_parser_keeps_steps_and_signed_scientific_values(self):
        rows = parse_metrics("noise\nStep 2000: grad_norm=6.29e-2, loss=0.0217, param_norm=1803.2943\r\n")
        self.assertEqual(rows[0][0], 2000)
        self.assertAlmostEqual(rows[0][1]["grad_norm"], 0.0629)
        self.assertEqual(rows[0][1]["loss"], 0.0217)

    def test_repeated_step_is_not_duplicated(self):
        rows = parse_metrics("Step 10: loss=0.2\nStep 10: loss=0.1\n")
        self.assertEqual(rows, [(10, {"loss": 0.1})])

    def test_nonfinite_metrics_are_rejected(self):
        with self.assertRaises(ValueError):
            parse_metrics("Step 0: loss=1e309\n")

    def test_pause_single_step_and_quit_without_opening_window(self):
        view = LivePreview.__new__(LivePreview)
        view.paused = True
        view.single_step = False
        view.quit_requested = False
        view.handle = SimpleNamespace(sync=lambda: None, is_running=lambda: True)
        view.sync = lambda: view.handle.sync()
        view.on_key(ord("N"))
        self.assertTrue(view.before_step())
        self.assertTrue(view.paused)
        self.assertFalse(view.single_step)
        view.on_key(32)
        self.assertFalse(view.paused)
        view.on_key(ord("Q"))
        self.assertFalse(view.before_step())


if __name__ == "__main__":
    unittest.main()
