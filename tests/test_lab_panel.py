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
    backend_class = FakeBackend

    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Tk display unavailable: {error}")
        self.root.withdraw()
        self.backend = self.backend_class()
        self.panel = LabPanel(self.root, self.backend, test_mode=True)
        self.dialog = patch("lab_panel.messagebox.showerror")
        self.dialog.start()
        self.root.update_idletasks()

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


class FakeWorkbench(FakeBackend):
    def __init__(self):
        super().__init__()
        self.checkpoint_list = ["/course/checkpoints/focus/run-2/2000", "/course/checkpoints/base/run-1/10000"]
        self.workbench_reads = self.result_reads = 0
        self.jobs = {"model": {"running": False, "owned": False},
                     "training": {"running": False, "owned": False}, "port_busy": False}
        self.rows = [dict(result_path="/tmp/evaluation/result.json", directory="/tmp/ui-check",
                          video="/tmp/evaluation/video.mp4", model="focus/2000", layout=1,
                          on=True, steps=110, review="待复核", note="")]

    def checkpoints(self):
        return self.checkpoint_list

    def workbench_state(self):
        self.workbench_reads += 1
        return dict(jobs=self.jobs, connection="尚未检查", model_log="model tail", training_log="train tail")

    def dataset_info(self):
        return dict(task="black bowl", train_episodes=40, train_frames=4000,
                    holdout_episodes=10, holdout_frames=1000)

    def results(self):
        self.result_reads += 1
        return self.rows

    def start_model(self, checkpoint):
        self.calls.append(("start_model", checkpoint))

    def stop_model(self):
        self.calls.append(("stop_model",))

    def check_connection(self):
        self.calls.append(("check_connection",))

    def start_training(self, parent, steps, batch, learning_rate, uniform=False):
        self.calls.append(("start_training", parent, steps, batch, learning_rate, uniform))

    def stop_training(self):
        self.calls.append(("stop_training",))

    def review_result(self, result_path, passed, note):
        self.calls.append(("review", result_path, passed, note))
        self.rows[0].update(review={None: "待复核", True: "收尾成功", False: "收尾失败"}[passed], note=note)

    def play_result(self, result_path):
        self.calls.append(("play", result_path))

    def open_result_directory(self, result_path):
        self.calls.append(("open", result_path))

    def open_curves(self):
        self.calls.append(("curves",))

    def capture(self):
        self.calls.append(("capture",))

    def reset_camera(self):
        self.calls.append(("reset_camera",))


class WorkbenchTests(PanelTests):
    backend_class = FakeWorkbench

    def test_owned_model_can_be_replaced_without_manual_stop(self):
        self.backend.jobs.update(port_busy=True, model={"owned": True, "running": True})
        self.panel._refresh_workbench()
        self.assertEqual(str(self.panel.model_start_button.cget("state")), "normal")
        self.backend.jobs["model"]["owned"] = False
        self.panel._refresh_workbench()
        self.assertEqual(str(self.panel.model_start_button.cget("state")), "disabled")

    def test_three_tabs_defaults_and_no_automatic_job_start(self):
        names = [self.panel.notebook.tab(tab, "text") for tab in self.panel.notebook.tabs()]
        self.assertEqual(names, ["仿真操作", "模型与训练", "结果与录像"])
        self.assertEqual(self.backend.calls, [])
        self.assertEqual(self.panel._checkpoint(self.panel.model_var), self.backend.checkpoint_list[0])
        self.assertFalse(self.panel.model_var.get().startswith("/"))
        self.assertEqual(self.panel.batch_var.get(), "4")
        self.assertEqual(self.panel.steps_var.get(), "2000")
        self.assertIn("4000 帧", self.panel.dataset_var.get())
        self.root.update_idletasks()
        self.assertLess(self.root.winfo_reqheight(), 1000)

    def test_model_and_training_buttons_use_selected_absolute_paths(self):
        self.panel.model_start_button.invoke()
        self.assertEqual(self.backend.calls[-1], ("start_model", self.backend.checkpoint_list[0]))
        self.panel.parent_var.set(tuple(self.panel.checkpoint_paths)[1])
        self.panel.sampler_var.set("均匀")
        self.panel.training_start_button.invoke()
        self.assertEqual(self.backend.calls[-1],
                         ("start_training", self.backend.checkpoint_list[1], 2000, 4, 5e-5, True))

    def test_invalid_training_then_valid_retry_keeps_gui_usable(self):
        for value in ("0", "bad", "-1"):
            self.panel.steps_var.set(value)
            self.panel.training_start_button.invoke()
        self.assertEqual(self.backend.calls, [])
        self.panel.steps_var.set("2000")
        self.panel.lr_var.set("nan")
        self.panel.training_start_button.invoke()
        self.assertEqual(self.backend.calls, [])
        self.panel.lr_var.set("5e-5")
        self.panel.training_start_button.invoke()
        self.assertEqual(self.backend.calls[-1][0], "start_training")

    def test_external_jobs_cannot_be_stopped_and_training_gates_motion(self):
        self.backend.jobs.update(model={"running": True, "owned": False}, port_busy=True)
        self.panel._refresh_workbench()
        for button in (self.panel.model_stop_button, self.panel.training_stop_button,
                       self.panel.model_start_button, self.panel.training_start_button):
            self.assertEqual(str(button.cget("state")), "disabled")
            button.invoke()
        self.assertEqual(self.backend.calls, [])
        self.backend.jobs["model"]["owned"] = True
        self.panel._refresh_workbench()
        self.panel.model_stop_button.invoke()
        self.assertEqual(self.backend.calls[-1], ("stop_model",))
        self.panel.load_button.invoke()
        self.backend.data["training_running"] = True
        self.panel._refresh()
        self.assertEqual(str(self.panel.run_button.cget("state")), "disabled")
        self.assertEqual(str(self.panel.step_button.cget("state")), "disabled")
        self.assertEqual(str(self.panel.save_button.cget("state")), "normal")

    def test_poll_throttles_logs_and_preserves_result_selection(self):
        path = self.backend.rows[0]["result_path"]
        self.panel.results_tree.selection_set(path)
        before = self.backend.workbench_reads
        with patch("lab_panel.time.monotonic", return_value=100):
            self.panel._poll()
            self.panel._poll()
        self.assertEqual(self.backend.workbench_reads, before + 1)
        self.assertEqual(self.backend.result_reads, 1)
        self.assertEqual(self.panel.results_tree.selection(), (path,))
        self.assertEqual(str(self.panel.model_log.cget("state")), "disabled")

    def test_review_keeps_on_and_dispatches_selected_video(self):
        path = self.backend.rows[0]["result_path"]
        self.panel.results_tree.selection_set(path)
        self.panel.review_var.set("收尾失败")
        self.panel.note_var.set("还未松爪")
        self.panel._save_review()
        self.assertEqual(self.backend.calls[-1], ("review", path, False, "还未松爪"))
        self.assertTrue(self.backend.rows[0]["on"])
        self.panel._play_current()
        self.assertEqual(self.backend.calls[-1], ("play", path))
        self.panel._select_result()
        self.assertEqual(self.panel.note_var.get(), "还未松爪")

    def test_checkpoint_refresh_retains_existing_choice(self):
        self.panel.model_var.set(tuple(self.panel.checkpoint_paths)[1])
        selected = self.panel.model_var.get()
        self.backend.checkpoint_list.insert(0, "/course/checkpoints/focus/run-3/4000")
        self.panel._refresh_checkpoints()
        self.assertEqual(self.panel.model_var.get(), selected)

    def test_exact_current_video_and_expired_review(self):
        self.backend.data["current_result_path"] = "/tmp/current/result.json"
        self.panel._play_current()
        self.assertEqual(self.backend.calls[-1], ("play", "/tmp/current/result.json"))
        path = self.backend.rows[0]["result_path"]
        self.backend.rows[0]["review"] = "复核过期"
        self.panel.results_tree.selection_set(path)
        self.panel._select_result()
        self.assertEqual(self.panel.review_var.get(), "待复核")

    def test_camera_and_curve_buttons_only_dispatch_explicit_clicks(self):
        self.panel.capture_button.invoke()
        self.panel.camera_reset_button.invoke()
        self.panel.curves_button.invoke()
        self.panel.connection_button.invoke()
        self.assertEqual(self.backend.calls,
                         [("capture",), ("reset_camera",), ("curves",), ("check_connection",)])

    def test_empty_results_and_workbench_error_recover_without_stopping_ticks(self):
        self.backend.rows = []
        self.panel._refresh_results()
        self.assertEqual(self.panel.results_tree.get_children(), ())
        with patch.object(self.backend, "workbench_state", side_effect=OSError("read failed")):
            self.panel._poll()
        self.assertTrue(self.panel._tick_enabled)
        self.panel._workbench_next = 0
        self.panel._poll()
        self.assertTrue(self.panel._tick_enabled)

    def test_log_refresh_preserves_reading_position(self):
        log = self.panel.model_log
        content = "\n".join(f"line {i}" for i in range(300))
        self.panel._update_log(log, content)
        log.yview_moveto(0.2)
        first = log.yview()[0]
        self.panel._update_log(log, content + "\nnew tail")
        self.assertAlmostEqual(log.yview()[0], first, places=2)
        log.yview_moveto(1)
        self.panel._update_log(log, content + "\nnext tail")
        self.assertGreater(log.yview()[1], 0.99)


if __name__ == "__main__":
    unittest.main()
