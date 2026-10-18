"""Real Tk widgets with an explicitly fake backend; no simulation or model."""
import json
from pathlib import Path
import sys
import tempfile
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


class FakeCorrectionWorkbench(FakeWorkbench):
    def __init__(self):
        super().__init__()
        self.correction = dict(active=False, frames=0, directory="/tmp/correction", approved_train=0,
                               approved_holdout=0, status="尚未接管")
        self.bundle_paths = []
        self.camera_reads = 0

    def begin_correction(self, split="train"):
        self.calls.append(("begin_correction", split))
        self.data["finished"] = True
        self.correction.update(active=True, frames=0)

    def resume_correction(self, directory):
        self.calls.append(("resume_correction", directory))
        self.data.update(ready=True, finished=True, running=False)
        self.correction.update(active=True, frames=770)

    def correction_state(self):
        return self.correction

    def manual_jog(self, axis, sign, scale, repeats):
        self.calls.append(("manual_jog", axis, sign, scale, repeats))
        self.correction["frames"] += repeats

    def manual_gripper(self, opened, repeats=10):
        self.calls.append(("manual_gripper", opened, repeats))
        self.correction["frames"] += repeats

    def manual_hold(self, repeats=10):
        self.calls.append(("manual_hold", repeats))
        self.correction["frames"] += repeats

    def finish_correction(self, approved):
        self.calls.append(("finish_correction", approved))
        self.correction["active"] = False

    def camera_images(self):
        import numpy as np
        self.camera_reads += 1
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        image[:2, :2] = (255, 0, 0)
        return {"image": image, "wrist_image": image.copy()}

    def corrections_bundles(self):
        return self.bundle_paths

    def prepare_corrections(self, base_manifest=None):
        self.calls.append(("prepare_corrections", base_manifest) if base_manifest is not None else ("prepare_corrections",))

    def start_training(self, parent, steps, batch, learning_rate, uniform=False, correction_manifest=None):
        super().start_training(parent, steps, batch, learning_rate, uniform)
        if correction_manifest is not None:
            self.calls[-1] += (correction_manifest,)


class CorrectionTests(PanelTests):
    backend_class = FakeCorrectionWorkbench

    def _begin(self):
        self.panel.load_button.invoke()
        self.panel.correction_start_button.invoke()

    def test_resume_draft_without_loaded_scene_then_gates_second_resume(self):
        with patch("lab_panel.filedialog.askdirectory", return_value="/tmp/draft"):
            self.panel.correction_resume_button.invoke()
        self.assertEqual(self.backend.calls[-1], ("resume_correction", "/tmp/draft"))
        self.assertIn("770 帧", self.panel.correction_status_var.get())
        self.assertEqual(str(self.panel.correction_resume_button.cget("state")), "disabled")
        self.assertEqual(str(self.panel.hold_button.cget("state")), "normal")

    def test_fourth_page_is_idle_until_explicit_takeover(self):
        names = [self.panel.notebook.tab(tab, "text") for tab in self.panel.notebook.tabs()]
        self.assertEqual(names, ["仿真操作", "模型与训练", "结果与录像", "人工纠正"])
        self.assertEqual(self.backend.calls, [])
        self.assertEqual(self.backend.camera_reads, 0)
        self.assertEqual(self.panel.jog_scale_var.get(), "0.1")
        self.assertEqual(self.panel.jog_repeats_var.get(), "5")
        for button in (self.panel.correction_start_button, self.panel.hold_button,
                       self.panel.correction_approve_button, self.panel.correction_draft_button):
            self.assertEqual(str(button.cget("state")), "disabled")
        self.root.update_idletasks()
        self.assertLess(self.root.winfo_reqheight(), 1100)

    def test_pause_takeover_buttons_and_explicit_human_steps(self):
        self.panel.load_button.invoke()
        self.panel.run_button.invoke()
        self.assertEqual(str(self.panel.correction_start_button.cget("state")), "disabled")
        self.panel.run_button.invoke()
        self.panel.correction_split_var.set("holdout")
        self.panel.correction_start_button.invoke()
        self.assertEqual(self.backend.calls[-1], ("begin_correction", "holdout"))
        for button in (self.panel.run_button, self.panel.load_button, self.panel.save_button,
                       self.panel.model_start_button, self.panel.training_start_button):
            self.assertEqual(str(button.cget("state")), "disabled")
        self.assertEqual(str(self.panel.correction_split_combo.cget("state")), "disabled")
        for (axis, sign), button in self.panel.jog_buttons.items():
            button.invoke()
            self.assertEqual(self.backend.calls[-1], ("manual_jog", axis, sign, 0.1, 5))
        self.panel.gripper_open_button.invoke()
        self.panel.gripper_close_button.invoke()
        self.panel.hold_button.invoke()
        self.assertEqual(self.backend.calls[-3:], [("manual_gripper", True, 10),
                                                  ("manual_gripper", False, 10), ("manual_hold", 10)])
        self.assertIn("90 帧", self.panel.correction_status_var.get())
        self.panel.correction_draft_button.invoke()
        self.assertEqual(self.backend.calls[-1], ("finish_correction", False))
        self.assertFalse(any(call[0] == "start_training" for call in self.backend.calls))

    def test_manual_validation_and_failed_approval_preserve_takeover(self):
        self._begin()
        before = list(self.backend.calls)
        for scale, repeats in (("nan", "5"), ("0.31", "5"), ("0.1", "0"), ("0.1", "1.5")):
            self.panel.jog_scale_var.set(scale)
            self.panel.jog_repeats_var.set(repeats)
            self.panel.jog_buttons[(0, 1)].invoke()
        self.assertEqual(self.backend.calls, before)
        self.panel.hold_button.invoke()
        with patch.object(self.backend, "finish_correction", side_effect=ValueError("请先松爪撤手")):
            self.panel.correction_approve_button.invoke()
        self.assertTrue(self.backend.correction["active"])
        self.assertEqual(str(self.panel.hold_button.cget("state")), "normal")
        self.panel.correction_approve_button.invoke()
        self.assertEqual(self.backend.calls[-1], ("finish_correction", True))

    def test_finished_trial_can_be_taken_over_and_manual_controls_stay_enabled(self):
        self.panel.load_button.invoke()
        self.backend.data["finished"] = True
        self.backend.jobs.update(model={"owned": True, "running": True}, port_busy=True)
        self.panel._refresh()
        self.panel._refresh_workbench()
        self.assertEqual(str(self.panel.correction_start_button.cget("state")), "normal")
        self.panel.correction_start_button.invoke()
        self.assertEqual(self.backend.calls[-1], ("begin_correction", "train"))
        self.assertTrue(self.backend.data["finished"])
        self.panel._refresh_workbench()
        self.assertEqual(str(self.panel.model_stop_button.cget("state")), "disabled")
        for button in self.panel._manual_buttons:
            self.assertEqual(str(button.cget("state")), "normal")
        self.panel.hold_button.invoke()
        self.assertEqual(self.backend.calls[-1], ("manual_hold", 10))
        self.assertEqual(str(self.panel.correction_approve_button.cget("state")), "normal")
        self.assertEqual(str(self.panel.correction_draft_button.cget("state")), "normal")

    def test_previews_only_poll_on_selected_page_and_throttle_without_actions(self):
        with patch("lab_panel.time.monotonic", return_value=100):
            self.panel._poll()
            self.assertEqual(self.backend.camera_reads, 0)
            self.panel.notebook.select(self.panel.correction_page)
            self.panel._poll()
            self.panel._poll()
        self.assertEqual(self.backend.camera_reads, 1)
        for photo in self.panel._camera_photos.values():
            self.assertEqual((photo.width(), photo.height()), (256, 256))
        self.assertTrue(all(call == ("tick",) for call in self.backend.calls))
        with patch("lab_panel.time.monotonic", return_value=100.31):
            self.panel._poll()
        self.assertEqual(self.backend.camera_reads, 2)

    def test_tk_png_fallback_preserves_canonical_camera_orientation(self):
        with patch("PIL.ImageTk.PhotoImage", side_effect=RuntimeError("Tk bridge unavailable")):
            self.panel._refresh_cameras()
        photo = self.panel._camera_photos["image"]
        self.assertEqual(photo.get(0, 0), (255, 0, 0))
        self.assertEqual(photo.get(255, 255), (0, 0, 0))
        with patch.object(self.backend, "camera_images", return_value=None):
            self.panel._refresh_cameras()
        self.assertEqual(self.panel._camera_photos, {})

    def test_dataset_explicit_selection_uniform_training_and_original_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text("{}")
            self.backend.bundle_paths = [str(manifest)]
            self.panel.dataset_refresh_button.invoke()
            self.assertEqual(self.panel.training_dataset_var.get(), "原始示范")
            self.panel.training_dataset_var.set(str(manifest))
            self.panel._select_dataset()
            self.assertEqual(self.panel.sampler_var.get(), "均匀")
            self.assertEqual(str(self.panel.sampler_combo.cget("state")), "disabled")
            self.panel.training_start_button.invoke()
            self.assertEqual(self.backend.calls[-1], ("start_training", self.backend.checkpoint_list[0],
                                                     2000, 4, 5e-5, True, str(manifest)))
            before = len(self.backend.calls)
            manifest.unlink()
            self.panel.training_start_button.invoke()
            self.assertEqual(len(self.backend.calls), before)
            self.backend.bundle_paths = []
            self.panel.dataset_refresh_button.invoke()
            self.assertEqual(self.panel.training_dataset_var.get(), str(manifest))
            self.panel.training_start_button.invoke()
            self.assertEqual(len(self.backend.calls), before)
            self.panel.training_dataset_var.set("unknown")
            self.panel.training_start_button.invoke()
            self.assertEqual(len(self.backend.calls), before)
            self.panel.training_dataset_var.set("原始示范")
            self.panel._select_dataset()
            self.assertEqual(str(self.panel.sampler_combo.cget("state")), "readonly")
            self.panel.sampler_var.set("收尾重点")
            self.panel.training_start_button.invoke()
            self.assertEqual(self.backend.calls[-1], ("start_training", self.backend.checkpoint_list[0],
                                                     2000, 4, 5e-5, False))

    def test_expanded_dataset_hint_identifies_public_only_demonstration_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text('{"dataset_kind":"expanded_public_v1","original_train_episodes":37,'
                                '"holdout_source_episode_ids_excluded":[0,1,2,3,4,5,6,7,8,9],'
                                '"task":"black bowl","train_episodes":37,"train_frames":4400,'
                                '"holdout_episodes":10,"holdout_frames":1170}')
            self.backend.bundle_paths = [str(manifest)]
            self.panel.dataset_refresh_button.invoke()
            self.panel.training_dataset_var.set(str(manifest))
            self.panel._select_dataset()
            self.assertEqual(self.panel.correction_dataset_hint.get(),
                             "公开示范扩充：37条训练，10条留出；人工纠正暂未混入")
            self.assertEqual(self.panel.sampler_var.get(), "均匀")
            self.assertEqual(str(self.panel.sampler_combo.cget("state")), "disabled")
            self.assertFalse(self.backend.calls)
            expected = "black bowl\n\n训练：37 条 / 4400 帧；留出：10 条 / 1170 帧"
            self.assertEqual(self.panel.dataset_var.get(), expected)
            with patch.object(self.backend, "dataset_info", return_value={
                    "task": "black bowl", "train_episodes": 16, "train_frames": 1971,
                    "holdout_episodes": 4, "holdout_frames": 473}):
                self.panel._refresh_workbench()
                self.panel._refresh_workbench()
                self.assertEqual(self.panel.dataset_var.get(), expected)
                self.panel.training_dataset_var.set("原始示范")
                self.panel._select_dataset()
                self.assertEqual(self.panel.dataset_var.get(),
                                 "black bowl\n\n训练：16 条 / 1971 帧；留出：4 条 / 473 帧")
                self.panel._refresh_workbench()
                self.assertIn("训练：16 条 / 1971 帧", self.panel.dataset_var.get())

    def test_mixed_dataset_hint_and_counts_keep_human_corrections_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps({
                "dataset_kind": "expanded_public_corrections_v1", "original_train_episodes": 37,
                "correction_train_episodes": 2, "task": "black bowl", "train_episodes": 39,
                "train_frames": 6155, "holdout_episodes": 10, "holdout_frames": 1170}))
            self.backend.bundle_paths = [str(manifest)]
            self.panel.dataset_refresh_button.invoke()
            self.panel.training_dataset_var.set(str(manifest))
            self.panel._select_dataset()
            self.assertEqual(self.panel.correction_dataset_hint.get(),
                             "混合示范：37条公开＋2条人工纠正；10条公开留出，原归一化不变")
            self.assertEqual(self.panel.dataset_var.get(),
                             "black bowl\n\n训练：39 条 / 6155 帧；留出：10 条 / 1170 帧")
            self.panel._refresh_workbench()
            self.assertIn("训练：39 条 / 6155 帧", self.panel.dataset_var.get())
            self.assertEqual(self.panel.sampler_var.get(), "均匀")
            self.assertEqual(str(self.panel.sampler_combo.cget("state")), "disabled")
            self.assertFalse(self.backend.calls)

    def test_curated_hint_reports_selected_frames_and_keeps_source_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps({
                "dataset_kind": "expanded_public_corrections_v1", "original_train_episodes": 37,
                "correction_train_episodes": 2, "correction_train_frames": 346,
                "curation_selection": "evidence/selection.json", "task": "black bowl",
                "train_episodes": 39, "train_frames": 4746, "holdout_episodes": 10, "holdout_frames": 1170}))
            self.backend.bundle_paths = [str(manifest)]
            self.panel.dataset_refresh_button.invoke()
            self.panel.training_dataset_var.set(str(manifest))
            self.panel._select_dataset()
            self.assertEqual(self.panel.correction_dataset_hint.get(),
                             "精选纠正：37条公开＋2段连续片段/346帧；原始录制保留，新录制需另行筛选")
            self.assertIn("训练：39 条 / 4746 帧", self.panel.dataset_var.get())
            self.assertFalse(self.backend.calls)

    def test_curated_recovery_phases_are_visible_and_never_start_training(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps({
                "dataset_kind": "expanded_public_corrections_v1", "original_train_episodes": 37,
                "correction_train_episodes": 4, "correction_train_frames": 481,
                "curation_selection": "evidence/selection.json", "task": "black bowl",
                "selected_source_ranges": [{"kind": kind} for kind in
                                           ("placement_release", "lift_clearance", "regrasp_lift", "placement_release")],
                "train_episodes": 41, "train_frames": 4881, "holdout_episodes": 10, "holdout_frames": 1170}))
            self.backend.bundle_paths = [str(manifest)]
            self.panel.dataset_refresh_button.invoke()
            self.panel.training_dataset_var.set(str(manifest))
            self.panel._select_dataset()
            self.assertEqual(self.panel.correction_dataset_hint.get(),
                             "精选纠正：37条公开＋4段/481帧（放置2、恢复2）；原始录制保留")
            self.assertIn("训练：41 条 / 4881 帧", self.panel.dataset_var.get())
            self.assertFalse(self.backend.calls)

    def test_focus_alias_shows_real_sampling_mode_without_starting_training(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps({
                "dataset_kind": "expanded_public_corrections_v1", "original_train_episodes": 37,
                "correction_train_episodes": 5, "correction_train_frames": 1066,
                "training_sampling_profile": {"path": "evidence/focus.json", "sha256": "pinned"},
                "task": "black bowl", "train_episodes": 42, "train_frames": 5466,
                "holdout_episodes": 10, "holdout_frames": 1170}))
            self.backend.bundle_paths = [str(manifest)]
            self.panel.dataset_refresh_button.invoke()
            self.panel.training_dataset_var.set(str(manifest))
            self.panel._select_dataset()
            self.assertEqual(self.panel.sampler_var.get(), "恢复重点")
            self.assertIn("恢复重点：42条/5466帧", self.panel.correction_dataset_hint.get())
            self.assertFalse(self.backend.calls)

    def test_prepare_forwards_selected_expanded_or_mixed_dataset_only(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            self.backend.bundle_paths = [str(manifest)]
            for kind in ("expanded_public_v1", "expanded_public_corrections_v1", None):
                manifest.write_text(json.dumps({"dataset_kind": kind}))
                self.panel.dataset_refresh_button.invoke()
                self.panel.training_dataset_var.set(str(manifest))
                self.panel.prepare_corrections_button.invoke()
                expected = ("prepare_corrections", str(manifest)) if kind is not None else ("prepare_corrections",)
                self.assertEqual(self.backend.calls[-1], expected)
            before = len(self.backend.calls)
            self.panel.training_dataset_var.set("unknown")
            self.panel.prepare_corrections_button.invoke()
            self.assertEqual(len(self.backend.calls), before)
            self.panel.training_dataset_var.set(str(manifest))
            manifest.unlink()
            self.panel.prepare_corrections_button.invoke()
            self.assertEqual(len(self.backend.calls), before)

    def test_prepare_data_only_dispatches_preparation(self):
        self.panel.prepare_corrections_button.invoke()
        self.assertEqual(self.backend.calls, [("prepare_corrections",)])


if __name__ == "__main__":
    unittest.main()
