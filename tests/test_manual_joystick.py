"""Real Tk component with fake callbacks; no simulator, model or hardware."""
import math
from pathlib import Path
import sys
import tkinter as tk
from tkinter import ttk
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from manual_joystick import TranslationJoystick, translation_vector


class VectorTests(unittest.TestCase):
    def test_screen_signs_circle_and_deadzone(self):
        self.assertEqual(translation_vector(55, 0), (1., 0.))
        self.assertEqual(translation_vector(0, -55), (0., 1.))
        self.assertEqual(translation_vector(0, 0), (0., 0.))
        self.assertEqual(translation_vector(1, 1), (0., 0.))
        x, y = translation_vector(1000, 1000)
        self.assertAlmostEqual(math.hypot(x, y), 1)
        self.assertGreater(x, 0)
        self.assertLess(y, 0)

    def test_invalid_vector_rejected(self):
        for arguments in [(float("nan"), 0), (0, float("inf")), (0, 0, 0)]:
            with self.assertRaises(ValueError):
                translation_vector(*arguments)


class JoystickTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Tk display unavailable: {error}")
        self.root.title("Joystick fake callback test")
        self.calls, self.errors = [], []
        self.joystick = TranslationJoystick(self.root, self.move, self.errors.append)
        self.joystick.pack()
        self.root.update()
        self.joystick.set_enabled(True)

    def tearDown(self):
        if hasattr(self, "root"):
            self.root.destroy()

    def move(self, x, y):
        self.calls.append((x, y))
        return True

    def press(self, x=130, y=75):
        self.joystick._press(SimpleNamespace(x=x, y=y))

    def pending(self):
        return self.root.tk.call("after", "info")

    def run_one_tick(self):
        self.root.after(65, self.root.quit)
        self.root.mainloop()

    def assert_stopped(self):
        self.assertFalse(self.joystick._held)
        self.assertIsNone(self.joystick._after_id)
        self.assertEqual(self.joystick._vector, (0., 0.))
        self.assertNotEqual(self.joystick.canvas.grab_current(), self.joystick.canvas)

    def test_drag_has_only_one_after_and_repeats_at_20hz(self):
        self.press()
        identifier = self.joystick._after_id
        for _ in range(10):
            self.joystick._motion(SimpleNamespace(x=130, y=75))
        self.assertEqual(self.joystick._after_id, identifier)
        self.assertEqual(len(self.pending()), 1)
        self.run_one_tick()
        self.assertEqual(self.calls, [(1., 0.)])
        self.assertEqual(len(self.pending()), 1)
        self.joystick.stop()

    def test_zero_and_disabled_do_not_schedule_or_call(self):
        self.press(75, 75)
        self.assertIsNone(self.joystick._after_id)
        self.assertEqual(self.calls, [])
        self.joystick._motion(SimpleNamespace(x=130, y=75))
        self.joystick._motion(SimpleNamespace(x=76, y=76))
        self.assertIsNone(self.joystick._after_id)
        self.joystick.set_enabled(False)
        self.press()
        self.assert_stopped()
        self.assertEqual(self.calls, [])

    def test_release_even_outside_widget_cancels_capture(self):
        self.press()
        self.assertEqual(self.joystick.canvas.grab_current(), self.joystick.canvas)
        self.joystick.canvas.event_generate("<ButtonRelease-1>", x=500, y=500)
        self.root.update()
        self.assert_stopped()
        self.assertEqual(len(self.pending()), 0)

    def test_disable_focusout_escape_and_unmap_cancel(self):
        for event in [None, "<FocusOut>", "<Escape>", "<Unmap>"]:
            self.joystick.set_enabled(True)
            self.press()
            if event is None:
                self.joystick.set_enabled(False)
            elif event == "<Unmap>":
                self.joystick.event_generate(event)
            else:
                self.joystick.canvas.event_generate(event)
            self.root.update()
            self.assert_stopped()
            self.assertEqual(len(self.pending()), 0)

    def test_destroy_cancels_pending_callback(self):
        self.press()
        self.joystick.destroy()
        self.assertIsNone(self.joystick._after_id)
        self.assertEqual(len(self.pending()), 0)
        self.run_one_tick()
        self.assertEqual(self.calls, [])

    def test_stop_preserves_native_combobox_popup_grab(self):
        combo = ttk.Combobox(self.root, values=("train", "holdout"), state="readonly")
        combo.pack()
        self.root.update()
        self.root.tk.call("ttk::combobox::Post", str(combo))
        self.root.update()
        popup = self.root.tk.call("grab", "current", str(self.root))
        self.assertIn("popdown", str(popup))
        self.joystick.set_enabled(False)
        self.assertEqual(self.root.tk.call("grab", "current", str(self.root)), popup)
        self.assertEqual(self.calls, [])
        self.root.tk.call("ttk::combobox::Unpost", str(combo))

    def test_false_and_exception_stop_without_tk_callback_error(self):
        for callback in [lambda x, y: False, lambda x, y: 1/0]:
            self.joystick.on_move = callback
            self.press()
            self.run_one_tick()
            self.assert_stopped()
            self.assertEqual(len(self.pending()), 0)
        self.assertEqual(len(self.errors), 1)
        self.assertIsInstance(self.errors[0], ZeroDivisionError)


if __name__ == "__main__":
    unittest.main()
