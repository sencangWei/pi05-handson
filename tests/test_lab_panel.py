"""Real Tk widgets with an explicitly fake backend; no simulation or model."""
from pathlib import Path
import sys
import tkinter as tk
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lab_panel import LabPanel


class FakeBackend:
    layout_count = 3

    def __init__(self):
        self.calls = []
        self.data = dict(status="界面检查，未连接仿真", ready=False, running=False, finished=False,
                         layout_id=None, control_steps=0, on_reached=False,
                         model_name="没有模型连接", output_dir="/tmp/ui-check")

    def state(self):
        return self.data

    def reset(self, layout, x, y):
        self.calls.append(("reset", layout, x, y))
        self.data.update(ready=True, finished=False, running=False, layout_id=layout, control_steps=0)

    def toggle_run(self):
        self.data["running"] = not self.data["running"]

    def single_step(self):
        self.data["control_steps"] += 1

    def save(self):
        self.data.update(finished=True, running=False)

    def tick(self):
        self.calls.append(("tick",))

    def close(self):
        self.calls.append(("close",))


class PanelTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Tk display unavailable: {error}")
        self.root.withdraw()
        self.backend = FakeBackend()
        self.panel = LabPanel(self.root, self.backend, test_mode=True)
        self.dialog = patch("lab_panel.messagebox.showerror")
        self.dialog.start()

    def tearDown(self):
        if hasattr(self, "panel"):
            self.panel.close()
            self.dialog.stop()

    def test_buttons_load_offsets_in_meters_and_preserve_pause(self):
        self.panel.layout_var.set(2)
        self.panel.x_var.set(5)
        self.panel.y_var.set(-3)
        self.panel.load_button.invoke()
        self.assertEqual(self.backend.calls[-1], ("reset", 2, 0.05, -0.03))
        self.panel.run_button.invoke()
        self.assertEqual(self.panel.run_button.cget("text"), "暂停")
        self.panel.run_button.invoke()
        self.panel.step_button.invoke()
        self.assertEqual(self.backend.data["control_steps"], 1)
        self.panel.save_button.invoke()
        self.assertEqual(str(self.panel.run_button.cget("state")), "disabled")
        self.assertEqual(str(self.panel.step_button.cget("state")), "disabled")

    def test_invalid_offset_never_reaches_backend(self):
        self.panel.x_var.set(11)
        self.panel.load_button.invoke()
        self.assertEqual(self.backend.calls, [])

    def test_original_position_resets_offsets(self):
        self.panel.x_var.set(5)
        self.panel.y_var.set(-3)
        self.panel._original()
        self.assertEqual(self.backend.calls[-1], ("reset", 0, 0, 0))

    def test_failed_close_keeps_panel_and_can_retry(self):
        with patch.object(self.backend, "close", side_effect=OSError("save failed")):
            self.panel.close()
        self.assertFalse(self.panel._closed)
        self.assertTrue(self.root.winfo_exists())
        self.panel.close()
        self.assertTrue(self.panel._closed)

    def test_poll_error_can_recover_after_new_operation(self):
        with patch.object(self.backend, "tick", side_effect=RuntimeError("service unavailable")):
            self.panel._poll()
        self.assertFalse(self.panel._tick_enabled)
        self.panel.load_button.invoke()
        self.assertTrue(self.panel._tick_enabled)


if __name__ == "__main__":
    unittest.main()
