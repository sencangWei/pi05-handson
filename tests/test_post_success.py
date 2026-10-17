import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_libero import arguments, update_benchmark_progress


class PostSuccessTests(unittest.TestCase):
    def result(self, count):
        return {"first_benchmark_success_control_steps": None,
                "control_steps": count, "success": False}

    def test_no_success_signal_never_stops_early(self):
        result = self.result(100)
        self.assertFalse(update_benchmark_progress(result, False, 0))
        self.assertFalse(result["success"])

    def test_default_keeps_original_stop_at_first_success(self):
        result = self.result(103)
        self.assertTrue(update_benchmark_progress(result, True, 0))
        self.assertEqual(result["first_benchmark_success_control_steps"], 103)
        self.assertEqual(result["post_success_steps_executed"], 0)

    def test_followup_is_exactly_forty_further_steps_even_if_predicate_changes(self):
        result = self.result(103)
        self.assertFalse(update_benchmark_progress(result, True, 40))
        result["control_steps"] = 142
        self.assertFalse(update_benchmark_progress(result, False, 40))
        result["control_steps"] = 143
        self.assertTrue(update_benchmark_progress(result, False, 40))
        self.assertTrue(result["success"])
        self.assertFalse(result["final_benchmark_success"])
        self.assertEqual(result["first_benchmark_success_control_steps"], 103)

    def test_followup_cli_is_opt_in(self):
        with patch.object(sys, "argv", ["evaluate_libero.py", "--output", "unused"]):
            self.assertEqual(arguments().post_success_steps, 0)


if __name__ == "__main__":
    unittest.main()
