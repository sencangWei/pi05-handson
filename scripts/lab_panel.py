"""Chinese controls for the simulation lab; the native viewer displays the scene."""

import math
import base64
import io
import json
from pathlib import Path
import subprocess
import time
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk


class LabPanel:
    def __init__(self, root, lab, *, test_mode=False):
        self.root, self.lab = root, lab
        self._closed = False
        self._tick_enabled = True
        self._after_id = None
        self._workbench = hasattr(lab, "workbench_state")
        self._correction = self._workbench and hasattr(lab, "begin_correction")
        self._camera_next = 0.0
        self._camera_photos = {}
        self._workbench_next = 0.0
        # Some bundled Tk builds expose core X fonts only; use a CJK font Tk actually lists.
        families = {name.lower() for name in tkfont.families(root)}
        family = "Noto Sans CJK SC" if "noto sans cjk sc" in families else "song ti"
        tkfont.nametofont("TkDefaultFont").configure(family=family, size=-18, weight="normal")
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure(".", font="TkDefaultFont")
        style.configure("Treeview", rowheight=28)
        root.minsize(720 if self._workbench else 560, 720 if self._workbench else 510)
        root.title("LIBERO 仿真实验面板" + (" — 界面检查，未连接仿真" if test_mode else ""))
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        container = root
        if self._workbench:
            self.notebook = ttk.Notebook(root)
            self.notebook.grid(sticky="nsew")
            container = self.notebook
        frame = ttk.Frame(container, padding=14)
        if self._workbench:
            self.notebook.add(frame, text="仿真操作")
        else:
            frame.grid(sticky="nsew")
        frame.columnconfigure(1, weight=1)
        ttk.Label(frame, text="任务：拿起盘子旁的黑碗，放到盘子上。",
                  wraplength=470).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(frame, text="三维窗口：空格暂停/继续，N 单步，Q 结束。",
                  wraplength=470).grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 12))
        self.layout_var = tk.IntVar(root, value=0)
        self.x_var = tk.DoubleVar(root, value=0.0)
        self.y_var = tk.DoubleVar(root, value=0.0)
        ttk.Label(frame, text="初始状态编号").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(frame, from_=0, to=lab.layout_count - 1, textvariable=self.layout_var,
                    width=8).grid(row=2, column=1, sticky="w")
        for row, name, variable in ((3, "碗 X 偏移 (cm)", self.x_var),
                                    (4, "碗 Y 偏移 (cm)", self.y_var)):
            ttk.Label(frame, text=name).grid(row=row, column=0, sticky="w")
            ttk.Scale(frame, from_=-10, to=10, variable=variable).grid(
                row=row, column=1, sticky="ew", padx=8, pady=5)
            ttk.Spinbox(frame, from_=-10, to=10, increment=0.5, format="%.1f",
                        textvariable=variable, width=7).grid(row=row, column=2)
        ttk.Label(frame, text="编号对应同一场景的小范围摆放差异。\n\n明显移动黑碗请调 X/Y 偏移，再点加载；会保存上一轮。",
                  wraplength=470).grid(row=5, column=0, columnspan=3, sticky="w", pady=6)
        controls = ttk.Frame(frame)
        controls.grid(row=6, column=0, columnspan=3, sticky="w", pady=6)
        self.load_button = ttk.Button(controls, text="加载/重置布局", command=self._load)
        self.load_button.grid(row=0, column=0, padx=(0, 5))
        ttk.Button(controls, text="原位置", command=self._original).grid(row=0, column=1, padx=5)
        self.run_button = ttk.Button(controls, text="运行", command=lambda: self._perform(lab.toggle_run))
        self.run_button.grid(row=0, column=2, padx=5)
        self.step_button = ttk.Button(controls, text="单步", command=lambda: self._perform(lab.single_step))
        self.step_button.grid(row=0, column=3, padx=5)
        self.save_button = ttk.Button(controls, text="结束并保存",
                                      command=lambda: self._perform(lab.save))
        self.save_button.grid(row=1, column=0, pady=8, sticky="w")
        ttk.Button(controls, text="打开结果目录", command=lambda: self._perform(self._open_results)).grid(
            row=1, column=1, columnspan=2, padx=5)
        ttk.Button(controls, text="退出", command=self.close).grid(row=1, column=3, padx=5)
        if hasattr(lab, "toggle_demo"):
            self.demo_button = ttk.Button(controls, text="录制 Demo", command=lambda: self._perform(
                lambda: lab.toggle_demo(root.winfo_id())))
            self.demo_button.grid(row=2, column=0, columnspan=2, sticky="w", pady=5)
            self.delete_demo_button = ttk.Button(controls, text="删除上一个 Demo",
                                                  command=lambda: self._perform(lab.delete_last_demo))
            self.delete_demo_button.grid(row=2, column=2, columnspan=2, sticky="w", padx=5)
            self.demo_var = tk.StringVar(root)
            ttk.Label(controls, textvariable=self.demo_var, wraplength=650).grid(
                row=3, column=0, columnspan=4, sticky="w")
        self.info_var = tk.StringVar(root)
        ttk.Label(frame, textvariable=self.info_var, justify="left", wraplength=470).grid(
            row=7, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self.status_var = tk.StringVar(root)
        ttk.Label(frame, textvariable=self.status_var, justify="left", wraplength=470).grid(
            row=8, column=0, columnspan=3, sticky="w", pady=(8, 0))
        if self._workbench:
            camera = ttk.Frame(frame)
            camera.grid(row=9, column=0, columnspan=3, sticky="w", pady=8)
            for column, (name, label, action) in enumerate((("capture_button", "相机截图", lab.capture),
                         ("camera_reset_button", "重置相机", lab.reset_camera),
                         ("play_current_button", "播放当前录像", self._play_current))):
                button = ttk.Button(camera, text=label, command=lambda action=action: self._perform(action))
                button.grid(row=0, column=column, padx=(0, 8))
                setattr(self, name, button)
            ttk.Label(frame, text="退出后，后台模型/训练继续运行，可重开管理。",
                      wraplength=650).grid(row=10, column=0, columnspan=3, sticky="w")
            self._build_workbench()
            if self._correction:
                self._build_correction()
                self._perform(self._refresh_datasets)
            self._perform(self._refresh_checkpoints)
            self._perform(self._refresh_results)
            self._perform(self._refresh_workbench)
        self._refresh()
        self._after_id = root.after(30, self._poll)

    def _load(self):
        def reset():
            layout = self.layout_var.get()
            x, y = self.x_var.get(), self.y_var.get()
            if not 0 <= layout < self.lab.layout_count:
                raise ValueError(f"布局编号须在 0～{self.lab.layout_count - 1} 之间")
            if not all(math.isfinite(v) and -10 <= v <= 10 for v in (x, y)):
                raise ValueError("X/Y 偏移须在 -10～10 cm 之间")
            self.lab.reset(layout, x / 100.0, y / 100.0)
        self._perform(reset)

    def _original(self):
        self.x_var.set(0.0)
        self.y_var.set(0.0)
        self._load()

    def _error(self, error):
        self.status_var.set(f"操作失败：{error}")
        messagebox.showerror("仿真实验面板", str(error), parent=self.root)

    def _perform(self, operation):
        try:
            operation()
            self._tick_enabled = True
        except Exception as error:
            self._error(error)
        self._refresh()
        self._workbench_next = 0.0

    def _refresh(self):
        state = self.lab.state()
        correcting = self._refresh_correction(state) if self._correction else False
        self.load_button.configure(state="disabled" if state.get("training_running", False) or correcting else "normal")
        available = state["ready"] and not state["finished"] and not state.get("training_running", False) and not correcting
        self.run_button.configure(state="normal" if available else "disabled",
                                  text="暂停" if state["running"] else "运行")
        self.step_button.configure(state="normal" if available else "disabled")
        self.save_button.configure(state="normal" if state["ready"] and not correcting else "disabled")
        if hasattr(self, "demo_button"):
            recording = state.get("demo_recording", False)
            self.demo_button.configure(text="停止并保存 Demo" if recording else "录制 Demo",
                                       state="normal" if recording or state["ready"] else "disabled")
            self.delete_demo_button.configure(state="normal" if state.get("demo_available") and not recording else "disabled")
            self.demo_var.set(state.get("demo_status", ""))
        layout = state["layout_id"] if state["layout_id"] is not None else "未加载"
        reached = "已达到" if state["on_reached"] else "未达到"
        model = state["model_name"]
        if state.get("model_path"):
            model += f"\n{state['model_path']}"
        self.info_var.set(f"当前布局：{layout}    控制步数：{state['control_steps']}\n\n"
                          f"官方 On 判据：{reached}；松爪撤手请观察\n\n模型：{model}")
        performance = state.get("performance_text", "")
        self.status_var.set(f"状态：{state['status']}\n{performance}\n\n结果目录：{state['output_dir']}")

    def _poll(self):
        if self._closed:
            return
        try:
            if self._tick_enabled:
                self.lab.tick()
            self._refresh()
        except Exception as error:
            self._tick_enabled = False
            self._error(error)
        if self._workbench and time.monotonic() >= self._workbench_next:
            self._workbench_next = time.monotonic() + 1.0
            try:
                self._refresh_workbench()
            except Exception as error:
                self._error(error)
        if (self._correction and self.notebook.select() == str(self.correction_page)
                and time.monotonic() >= self._camera_next):
            self._camera_next = time.monotonic() + 0.3
            try:
                self._refresh_cameras()
            except Exception as error:
                self._error(error)
        state = self.lab.state()
        active = state.get("running", False) or state.get("waiting", False)
        self._after_id = self.root.after(10 if active else 30, self._poll)

    def _build_workbench(self):
        page = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(page, text="模型与训练")
        page.columnconfigure(1, weight=1)
        self.model_var, self.parent_var = tk.StringVar(), tk.StringVar()
        self.steps_var = tk.StringVar(value="2000")
        self.batch_var = tk.StringVar(value="4")
        self.lr_var = tk.StringVar(value="5e-5")
        self.sampler_var = tk.StringVar(value="收尾重点")
        self.connection_var, self.dataset_var = tk.StringVar(), tk.StringVar()
        self.checkpoint_paths = {}
        ttk.Label(page, text="推理 checkpoint").grid(row=0, column=0, sticky="w")
        self.model_combo = ttk.Combobox(page, textvariable=self.model_var, state="readonly")
        self.model_combo.grid(row=0, column=1, columnspan=3, sticky="ew", padx=6)
        actions = ttk.Frame(page)
        actions.grid(row=1, column=0, columnspan=4, sticky="w", pady=6)
        for column, (name, text, action) in enumerate((
                ("model_start_button", "启动所选模型", self._start_model),
                ("model_stop_button", "停止本面板模型", self.lab.stop_model),
                ("connection_button", "检查连接", self.lab.check_connection),
                ("checkpoint_refresh_button", "刷新模型", self._refresh_checkpoints))):
            button = ttk.Button(actions, text=text, command=lambda action=action: self._perform(action))
            if name == "model_stop_button":
                button.configure(state="disabled")
            button.grid(row=0, column=column, padx=(0, 6))
            setattr(self, name, button)
        ttk.Label(page, textvariable=self.connection_var, wraplength=660).grid(
            row=2, column=0, columnspan=4, sticky="w")
        ttk.Label(page, text="继续训练起点").grid(row=3, column=0, sticky="w", pady=(8, 0))
        self.parent_combo = ttk.Combobox(page, textvariable=self.parent_var, state="readonly")
        self.parent_combo.grid(row=3, column=1, columnspan=3, sticky="ew", padx=6)
        options = ttk.Frame(page)
        options.grid(row=4, column=0, columnspan=4, sticky="ew", pady=6)
        for column, (label, variable, choices) in enumerate((
                ("Steps", self.steps_var, None), ("Batch", self.batch_var, ("1", "2", "4")),
                ("学习率", self.lr_var, None), ("采样", self.sampler_var, ("收尾重点", "均匀")))):
            ttk.Label(options, text=label).grid(row=0, column=column * 2, padx=(0, 4))
            widget = (ttk.Combobox(options, values=choices, state="readonly", textvariable=variable,
                                  width=8) if choices else ttk.Entry(options, textvariable=variable, width=8))
            widget.grid(row=0, column=column * 2 + 1, padx=(0, 8))
            if variable is self.sampler_var:
                self.sampler_combo = widget
        ttk.Label(page, text="点击才开始训练，并保存关闭三维场景、停止本面板模型。\n\n外部服务占用时，请先在原终端退出。",
                  wraplength=660).grid(row=5, column=0, columnspan=4, sticky="w")
        train_actions = ttk.Frame(page)
        train_actions.grid(row=6, column=0, columnspan=4, sticky="w", pady=6)
        for column, (name, text, action) in enumerate((
                ("training_start_button", "开始训练", self._start_training),
                ("training_stop_button", "停止本面板训练", self.lab.stop_training),
                ("curves_button", "训练曲线（快照）", self.lab.open_curves))):
            button = ttk.Button(train_actions, text=text, command=lambda action=action: self._perform(action))
            if name == "training_stop_button":
                button.configure(state="disabled")
            button.grid(row=0, column=column, padx=(0, 8))
            setattr(self, name, button)
        ttk.Label(page, textvariable=self.dataset_var, wraplength=660).grid(
            row=7, column=0, columnspan=4, sticky="w")
        if self._correction:
            datasets = ttk.Frame(page)
            datasets.grid(row=12, column=0, columnspan=4, sticky="ew", pady=6)
            datasets.columnconfigure(1, weight=1)
            ttk.Label(datasets, text="训练数据集").grid(row=0, column=0, sticky="w")
            self.training_dataset_var = tk.StringVar(value="原始示范")
            self.dataset_paths = {"原始示范": None}
            self.dataset_combo = ttk.Combobox(datasets, textvariable=self.training_dataset_var, state="readonly")
            self.dataset_combo.grid(row=0, column=1, sticky="ew", padx=6)
            self.dataset_combo.bind("<<ComboboxSelected>>", self._select_dataset)
            self.dataset_refresh_button = ttk.Button(datasets, text="刷新数据集",
                                                     command=lambda: self._perform(self._refresh_datasets))
            self.dataset_refresh_button.grid(row=0, column=2)
            self.correction_dataset_hint = tk.StringVar()
            ttk.Label(datasets, textvariable=self.correction_dataset_hint, wraplength=660).grid(
                row=1, column=0, columnspan=3, sticky="w")
        for row, name, label in ((8, "model_log", "模型日志"), (10, "training_log", "训练日志")):
            ttk.Label(page, text=label).grid(row=row, column=0, columnspan=4, sticky="w", pady=(5, 0))
            log_frame = ttk.Frame(page)
            log_frame.grid(row=row + 1, column=0, columnspan=4, sticky="nsew")
            log_frame.columnconfigure(0, weight=1)
            log_frame.rowconfigure(0, weight=1)
            page.rowconfigure(row + 1, weight=1)
            log = tk.Text(log_frame, height=5, width=50, state="disabled", wrap="word", font="TkDefaultFont")
            log.grid(row=0, column=0, sticky="nsew")
            scroll = ttk.Scrollbar(log_frame, command=log.yview)
            scroll.grid(row=0, column=1, sticky="ns")
            log.configure(yscrollcommand=scroll.set)
            setattr(self, name, log)
        results_page = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(results_page, text="结果与录像")
        results_page.columnconfigure(0, weight=1)
        results_page.rowconfigure(1, weight=1)
        ttk.Label(results_page, text="On 是官方放置判据；完整收尾请播放录像后人工复核。",
                  wraplength=660).grid(row=0, column=0, sticky="w", pady=(0, 8))
        columns = ("model", "layout", "on", "steps", "review")
        self.results_tree = ttk.Treeview(results_page, columns=columns, show="headings", selectmode="browse")
        for name, title, width in zip(columns, ("模型", "布局", "On", "步数", "收尾复核"), (220, 60, 60, 70, 110)):
            self.results_tree.heading(name, text=title)
            self.results_tree.column(name, width=width, minwidth=45)
        self.results_tree.grid(row=1, column=0, sticky="nsew")
        self.results_tree.bind("<<TreeviewSelect>>", self._select_result)
        result_scroll = ttk.Scrollbar(results_page, command=self.results_tree.yview)
        result_scroll.grid(row=1, column=1, sticky="ns")
        self.results_tree.configure(yscrollcommand=result_scroll.set)
        buttons = ttk.Frame(results_page)
        buttons.grid(row=2, column=0, sticky="w", pady=8)
        for column, (text, action) in enumerate((("刷新结果", self._refresh_results),
                                              ("播放录像", lambda: self.lab.play_result(self._selected_result())),
                                              ("打开所在目录", lambda: self.lab.open_result_directory(self._selected_result())))):
            ttk.Button(buttons, text=text, command=lambda action=action: self._perform(action)).grid(
                row=0, column=column, padx=(0, 8))
        review = ttk.Frame(results_page)
        review.grid(row=3, column=0, sticky="ew")
        review.columnconfigure(1, weight=1)
        self.review_var, self.note_var = tk.StringVar(value="待复核"), tk.StringVar()
        ttk.Combobox(review, values=("待复核", "收尾成功", "收尾失败"), state="readonly",
                     textvariable=self.review_var, width=10).grid(row=0, column=0, padx=(0, 8))
        ttk.Entry(review, textvariable=self.note_var).grid(row=0, column=1, sticky="ew")
        ttk.Button(review, text="保存复核", command=lambda: self._perform(self._save_review)).grid(
            row=0, column=2, padx=(8, 0))
        self._results = {}

    def _build_correction(self):
        from manual_joystick import TranslationJoystick
        page = self.correction_page = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(page, text="人工纠正")
        page.columnconfigure(0, weight=1)
        ttk.Label(page, text="先暂停模型，再开始人工接管；按钮和摇杆执行的动作都会录制。\n"
                  "放稳后：松爪 10 步 → 撤手至少 10cm → 保持 10 步，最后 10 步均须满足 On。\n"
                  "摇杆跟随三维窗口的视角；下面两路模型相机的方向可能不同。草稿不进入训练。",
                  wraplength=660, justify="left").grid(row=0, column=0, sticky="w")
        start = ttk.Frame(page)
        start.grid(row=1, column=0, sticky="w", pady=8)
        self.correction_split_var = tk.StringVar(value="train")
        ttk.Label(start, text="本段用途").grid(row=0, column=0)
        self.correction_split_combo = ttk.Combobox(start, values=("train", "holdout"), width=9,
                                                  textvariable=self.correction_split_var, state="readonly")
        self.correction_split_combo.grid(row=0, column=1, padx=6)
        self.correction_start_button = ttk.Button(start, text="开始人工接管",
                                                  command=lambda: self._perform(self._begin_correction))
        self.correction_start_button.grid(row=0, column=2, padx=6)
        self.correction_resume_button = ttk.Button(start, text="继续草稿", command=lambda: self._perform(self._resume_correction))
        self.correction_resume_button.grid(row=0, column=3, padx=6)
        ttk.Label(page, text="train 用于训练，holdout 仅评测；相同布局与偏移不能分属两组。",
                  wraplength=660).grid(row=2, column=0, sticky="w")
        cameras = ttk.Frame(page)
        cameras.grid(row=3, column=0, sticky="w", pady=8)
        self.camera_labels = {}
        for column, (key, title) in enumerate((("image", "场景相机"), ("wrist_image", "腕部相机"))):
            box = ttk.Frame(cameras, width=266, height=285)
            box.grid(row=0, column=column, padx=(0, 10))
            box.grid_propagate(False)
            ttk.Label(box, text=title).grid(row=0, column=0, sticky="w")
            label = ttk.Label(box, text="加载场景后显示", anchor="center")
            label.grid(row=1, column=0, sticky="nsew")
            box.columnconfigure(0, weight=1)
            box.rowconfigure(1, weight=1)
            self.camera_labels[key] = label
        controls = ttk.Frame(page)
        controls.grid(row=4, column=0, sticky="w", pady=4)
        self.translation_joystick = TranslationJoystick(controls, self._joystick_move, self._error)
        self.translation_joystick.grid(row=0, column=0, rowspan=4, sticky="n", padx=(0, 12))
        options = ttk.Frame(controls)
        options.grid(row=0, column=1, sticky="w", pady=4)
        self.motion_mode_var = tk.StringVar(value="普通移动")
        self.normal_motion_button = ttk.Radiobutton(options, text="普通移动", value="普通移动",
            variable=self.motion_mode_var, command=lambda: self._manual_motion_preset(.6))
        self.normal_motion_button.grid(row=0, column=0, columnspan=2, sticky="w")
        self.precision_motion_button = ttk.Radiobutton(options, text="精细移动", value="精细移动",
            variable=self.motion_mode_var, command=lambda: self._manual_motion_preset(.1))
        self.precision_motion_button.grid(row=0, column=2, columnspan=2, sticky="w")
        self.jog_scale_var = tk.StringVar(value="0.6")
        self.jog_repeats_var = tk.StringVar(value="5")
        ttk.Label(options, text="平移幅度 (0.05～0.8)").grid(row=1, column=0)
        ttk.Spinbox(options, from_=0.05, to=0.8, increment=0.05, width=6,
                    textvariable=self.jog_scale_var).grid(row=1, column=1, padx=6)
        ttk.Label(options, text="按钮连续步数 (1～20)").grid(row=1, column=2)
        ttk.Spinbox(options, from_=1, to=20, width=5,
                    textvariable=self.jog_repeats_var).grid(row=1, column=3, padx=6)
        self.motion_hint_var = tk.StringVar(value="普通 0.6 用于空中搬运；精细 0.1 用于抓放。旋转固定 0.1。")
        ttk.Label(options, textvariable=self.motion_hint_var, wraplength=410).grid(
            row=2, column=0, columnspan=4, sticky="w", pady=(4, 0))
        legend = ttk.Frame(controls)
        legend.grid(row=1, column=1, sticky="w")
        for column, (text, color) in enumerate((("世界 X 红", "#ce2020"),
                                              ("Y 绿", "#188231"), ("Z 蓝", "#235ac4"))):
            ttk.Label(legend, text=text, foreground=color).grid(row=0, column=column, padx=(0, 8))
        self.top_view_button = ttk.Button(legend, text="俯视对准",
                                          command=lambda: self._perform(self.lab.top_camera))
        self.top_view_button.grid(row=0, column=3)
        jog = ttk.Frame(controls)
        jog.grid(row=2, column=1, sticky="w", pady=4)
        self.jog_buttons = {}
        self._manual_buttons = []
        for axis, label in enumerate(("X", "Y", "Z", "Rx", "Ry", "Rz")):
            for row, sign in enumerate((1, -1)):
                text = f"{label} {'+' if sign > 0 else '−'}"
                if axis == 2:
                    text = "Z ↑ 抬高" if sign > 0 else "Z ↓ 降低"
                button = ttk.Button(jog, text=text, width=7,
                                    command=lambda axis=axis, sign=sign: self._perform(
                                        lambda: self._manual_jog(axis, sign)))
                button.grid(row=row, column=axis, padx=(0, 6), pady=3)
                self.jog_buttons[(axis, sign)] = button
                self._manual_buttons.append(button)
        actions = ttk.Frame(controls)
        actions.grid(row=3, column=1, sticky="w", pady=6)
        for column, (name, label, action) in enumerate((
                ("gripper_open_button", "松爪（10步）", lambda: self.lab.manual_gripper(True, repeats=10)),
                ("gripper_close_button", "闭爪（10步）", lambda: self.lab.manual_gripper(False, repeats=10)),
                ("hold_button", "保持（10步）", lambda: self.lab.manual_hold(repeats=10)))):
            button = ttk.Button(actions, text=label, command=lambda action=action: self._perform(action))
            button.grid(row=0, column=column, padx=(0, 8))
            setattr(self, name, button)
            self._manual_buttons.append(button)
        finish = ttk.Frame(page)
        finish.grid(row=7, column=0, sticky="w", pady=6)
        self.correction_approve_button = ttk.Button(finish, text="保存为合格",
                                                    command=lambda: self._perform(
                                                        lambda: self.lab.finish_correction(approved=True)))
        self.correction_approve_button.grid(row=0, column=0, padx=(0, 8))
        self.correction_draft_button = ttk.Button(finish, text="结束并存草稿",
                                                  command=lambda: self._perform(
                                                      lambda: self.lab.finish_correction(approved=False)))
        self.correction_draft_button.grid(row=0, column=1, padx=(0, 8))
        self.prepare_corrections_button = ttk.Button(finish, text="准备混合训练数据",
                                                     command=lambda: self._perform(self._prepare_corrections))
        self.prepare_corrections_button.grid(row=0, column=2)
        self.correction_status_var = tk.StringVar()
        ttk.Label(page, textvariable=self.correction_status_var, justify="left", wraplength=660).grid(
            row=8, column=0, sticky="w", pady=6)

    def _begin_correction(self):
        split = self.correction_split_var.get()
        if split not in ("train", "holdout"):
            raise ValueError("本段用途须选择 train 或 holdout")
        self.lab.begin_correction(split=split)

    def _resume_correction(self):
        project = getattr(self.lab, "project", Path.home() / ".pi05-handson")
        directory = filedialog.askdirectory(parent=self.root, title="选择含最终状态的纠正草稿目录",
                                           initialdir=str(project / "corrections"), mustexist=True)
        if directory:
            self.lab.resume_correction(directory)

    def _manual_jog(self, axis, sign):
        scale, repeats = float(self.jog_scale_var.get()), int(self.jog_repeats_var.get())
        if not math.isfinite(scale) or not 0.05 <= scale <= 0.8 or not 1 <= repeats <= 20:
            raise ValueError("平移幅度须在 0.05～0.8，连续步数须为 1～20 的整数")
        self.lab.manual_jog(axis, sign, scale=.1 if axis >= 3 else scale, repeats=repeats)

    def _manual_motion_preset(self, scale):
        self.translation_joystick.stop()
        self.jog_scale_var.set(str(scale))

    def _joystick_move(self, x, y):
        try:
            self.lab.manual_translate(x, y, scale=float(self.jog_scale_var.get()))
        except Exception as error:
            self._error(error)
            return False
        self._refresh()
        return True

    def _refresh_correction(self, scene):
        state = self.lab.correction_state()
        active = bool(state["active"])
        training = scene.get("training_running", False)
        available = scene["ready"] and not scene["running"] and not training
        self.translation_joystick.set_enabled(
            active and available and self.notebook.select() == str(self.correction_page))
        self.top_view_button.configure(state="normal" if scene["ready"] and not training else "disabled")
        self.correction_start_button.configure(state="normal" if available and not active else "disabled")
        self.correction_resume_button.configure(state="disabled" if active or training or scene["running"] else "normal")
        self.correction_split_combo.configure(state="disabled" if active else "readonly")
        for button in self._manual_buttons:
            button.configure(state="normal" if active and available else "disabled")
        self.correction_approve_button.configure(
            state="normal" if active and available and state.get("frames", 0) > 0 else "disabled")
        self.correction_draft_button.configure(state="normal" if active and not training else "disabled")
        self.prepare_corrections_button.configure(state="disabled" if active or training else "normal")
        if active:
            self.model_start_button.configure(state="disabled")
            self.model_stop_button.configure(state="disabled")
            self.training_start_button.configure(state="disabled")
        self.correction_status_var.set(f"{state.get('status', '')}；本段 {state.get('frames', 0)} 帧\n\n"
                                       f"已合格：训练 {state.get('approved_train', 0)} 段；"
                                       f"留出 {state.get('approved_holdout', 0)} 段\n\n"
                                       f"纠正目录：{state.get('directory', '')}")
        return active

    def _refresh_cameras(self):
        images = self.lab.camera_images()
        if images is None:
            self._camera_photos.clear()
            for label in self.camera_labels.values():
                label.configure(image="", text="加载场景后显示")
            return
        from PIL import Image
        for key, label in self.camera_labels.items():
            picture = Image.fromarray(images[key]).resize((256, 256))
            try:
                from PIL import ImageTk
                photo = ImageTk.PhotoImage(picture, master=self.root)
            except (ImportError, RuntimeError, tk.TclError):
                png = io.BytesIO()
                picture.save(png, format="PNG")
                photo = tk.PhotoImage(master=self.root, data=base64.b64encode(png.getvalue()))
            self._camera_photos[key] = photo
            label.configure(image=photo, text="")

    def _refresh_datasets(self):
        paths = self.lab.corrections_bundles()
        self.dataset_paths = {"原始示范": None}
        self.dataset_paths.update({str(Path(path)): str(path) for path in paths if Path(path).is_absolute()})
        self.dataset_combo.configure(values=tuple(self.dataset_paths))
        self._select_dataset()

    def _select_dataset(self, _event=None):
        manifest_path = self.dataset_paths.get(self.training_dataset_var.get())
        correction = manifest_path is not None
        if correction:
            self.sampler_var.set("均匀")
        self.sampler_combo.configure(state="disabled" if correction else "readonly")
        hint = "纠正数据：原示范与人工纠正均匀混合，原归一化不变" if correction else ""
        if manifest_path is not None:
            try:
                manifest = json.loads(Path(manifest_path).read_text())
                if manifest.get("training_sampling_profile"):
                    self.sampler_var.set("恢复重点")
                if manifest.get("dataset_kind") == "expanded_public_v1":
                    hint = (f"公开示范扩充：{manifest['original_train_episodes']}条训练，"
                            f"{len(manifest['holdout_source_episode_ids_excluded'])}条留出；人工纠正暂未混入")
                elif manifest.get("dataset_kind") == "expanded_public_corrections_v1":
                    hint = (f"混合示范：{manifest['original_train_episodes']}条公开＋"
                            f"{manifest['correction_train_episodes']}条人工纠正；"
                            f"{manifest['holdout_episodes']}条公开留出，原归一化不变")
                    if manifest.get("curation_selection"):
                        hint = (f"精选纠正：{manifest['original_train_episodes']}条公开＋"
                                f"{manifest['correction_train_episodes']}段连续片段/{manifest['correction_train_frames']}帧；"
                                "原始录制保留，新录制需另行筛选")
                        recoveries = sum(isinstance(clip, dict) and clip.get("kind") in
                                         ("lift_clearance", "regrasp_lift")
                                         for clip in manifest.get("selected_source_ranges", []))
                        if recoveries:
                            placements = manifest['correction_train_episodes'] - recoveries
                            hint = (f"精选纠正：{manifest['original_train_episodes']}条公开＋"
                                    f"{manifest['correction_train_episodes']}段/{manifest['correction_train_frames']}帧"
                                    f"（放置{placements}、恢复{recoveries}）；原始录制保留")
                if manifest.get("training_sampling_profile"):
                    hint = (f"恢复重点：{manifest['train_episodes']}条/{manifest['train_frames']}帧；"
                            "按固定阶段比例抽样，原始示范完整保留；先检查关键动作，再做仿真")
            except (OSError, ValueError, KeyError, TypeError):
                hint = "训练数据清单无效，请刷新；启动前将重新校验"
        self.correction_dataset_hint.set(hint)
        self._refresh_dataset_info()

    def _refresh_dataset_info(self):
        data = self.lab.dataset_info()
        if self._correction:
            manifest_path = self.dataset_paths.get(self.training_dataset_var.get())
            if manifest_path is not None:
                try:
                    manifest = json.loads(Path(manifest_path).read_text())
                    if manifest.get("dataset_kind") in ("expanded_public_v1", "expanded_public_corrections_v1"):
                        data = {key: manifest[key] for key in
                                ("task", "train_episodes", "train_frames", "holdout_episodes", "holdout_frames")}
                except (OSError, ValueError, KeyError, TypeError):
                    self.dataset_var.set("所选训练数据清单无效，请刷新数据集")
                    return
        self.dataset_var.set(f"{data.get('task', '')}\n\n训练：{data.get('train_episodes', 0)} 条 / "
                             f"{data.get('train_frames', 0)} 帧；留出：{data.get('holdout_episodes', 0)} 条 / "
                             f"{data.get('holdout_frames', 0)} 帧")

    def _prepare_corrections(self):
        selected = self.training_dataset_var.get()
        if selected not in self.dataset_paths:
            raise ValueError("请刷新并选择有效的训练数据集")
        path = self.dataset_paths[selected]
        manifest = json.loads(Path(path).read_text()) if path is not None else {}
        if manifest.get("dataset_kind") in ("expanded_public_v1", "expanded_public_corrections_v1"):
            self.lab.prepare_corrections(base_manifest=path)
        elif manifest.get("dataset_kind") is None:
            self.lab.prepare_corrections()
        else:
            raise ValueError("所选训练数据版本不受支持，请刷新")
        self._refresh_datasets()

    def _refresh_checkpoints(self):
        paths = self.lab.checkpoints()
        self.checkpoint_paths = {str(Path(path).relative_to(Path(path).parents[2])): path for path in paths}
        labels = tuple(self.checkpoint_paths)
        for combo, variable in ((self.model_combo, self.model_var), (self.parent_combo, self.parent_var)):
            combo.configure(values=labels)
            if variable.get() not in labels:
                variable.set(labels[0] if labels else "")

    def _checkpoint(self, variable):
        if variable.get() not in self.checkpoint_paths:
            raise ValueError("请先选择一个 checkpoint")
        return self.checkpoint_paths[variable.get()]

    def _start_model(self):
        self.lab.start_model(self._checkpoint(self.model_var))
        self._workbench_next = 0.0

    def _start_training(self):
        steps, batch, lr = int(self.steps_var.get()), int(self.batch_var.get()), float(self.lr_var.get())
        if steps <= 0 or batch not in (1, 2, 4) or not math.isfinite(lr) or lr <= 0:
            raise ValueError("Steps 须为正整数，Batch 须为 1/2/4，学习率须为正数")
        options = {"uniform": self.sampler_var.get() == "均匀"}
        if self._correction:
            selected = self.training_dataset_var.get()
            if selected not in self.dataset_paths:
                raise ValueError("请刷新并选择有效的训练数据集")
            manifest = self.dataset_paths[selected]
            if manifest is not None:
                if not Path(manifest).is_absolute() or not Path(manifest).is_file():
                    raise ValueError("所选纠正数据清单已失效，请刷新数据集")
                options["correction_manifest"] = manifest
                options["uniform"] = True
        self.lab.start_training(self._checkpoint(self.parent_var), steps, batch, lr, **options)
        self._workbench_next = 0.0

    @staticmethod
    def _update_log(widget, content):
        if widget.get("1.0", "end-1c") == content:
            return
        first, last = widget.yview()
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        widget.configure(state="disabled")
        widget.see("end") if last >= 0.99 else widget.yview_moveto(first)

    def _refresh_workbench(self):
        state = self.lab.workbench_state()
        model, training = (state["jobs"].get(name, {}) for name in ("model", "training"))
        self.model_stop_button.configure(state="normal" if model.get("running") and model.get("owned") else "disabled")
        self.training_stop_button.configure(state="normal" if training.get("running") and training.get("owned") else "disabled")
        self.model_start_button.configure(state="disabled" if (state["jobs"].get("port_busy") and not model.get("owned")) or training.get("running") else "normal")
        self.training_start_button.configure(state="disabled" if training.get("running") or (
            state["jobs"].get("port_busy") and not model.get("owned")) else "normal")
        if self._correction and self.lab.correction_state()["active"]:
            self.model_start_button.configure(state="disabled")
            self.model_stop_button.configure(state="disabled")
            self.training_start_button.configure(state="disabled")
        def job_label(job):
            return ((f"本面板运行中 PID {job.get('pid', '?')}" if job.get("owned") else "外部服务运行中")
                    if job.get("running") else "已停止")
        self.connection_var.set(f"{state.get('connection', '')}\n\n模型：{job_label(model)}；训练：{job_label(training)}")
        for name in ("model_log", "training_log"):
            self._update_log(getattr(self, name), state.get(name, ""))
        self._refresh_dataset_info()

    def _selected_result(self):
        selection = self.results_tree.selection()
        if not selection:
            raise ValueError("请先选择一条结果")
        return selection[0]

    def _refresh_results(self):
        selected = self.results_tree.selection()
        self._results = {row["result_path"]: row for row in self.lab.results()}
        self.results_tree.delete(*self.results_tree.get_children())
        for path, row in self._results.items():
            self.results_tree.insert("", "end", iid=path, values=(row["model"], row["layout"],
                "达到" if row["on"] else "未达到", row["steps"], row["review"]))
        if selected and selected[0] in self._results:
            self.results_tree.selection_set(selected[0])

    def _select_result(self, _event=None):
        selection = self.results_tree.selection()
        if selection:
            row = self._results.get(selection[0], {})
            review = row.get("review", "待复核")
            self.review_var.set(review if review in ("收尾成功", "收尾失败") else "待复核")
            self.note_var.set(row.get("note", ""))

    def _save_review(self):
        self.lab.review_result(self._selected_result(),
                               {"待复核": None, "收尾成功": True, "收尾失败": False}[self.review_var.get()],
                               self.note_var.get())
        self._refresh_results()

    def _play_current(self):
        state = self.lab.state()
        if state.get("current_result_path"):
            self.lab.play_result(state["current_result_path"])
            return
        directory = str(state["output_dir"])
        result = next((row for row in self.lab.results() if row["directory"] == directory), None)
        if result is None:
            raise ValueError("当前轮次尚无已保存录像，请先结束并保存")
        self.lab.play_result(result["result_path"])

    def _open_results(self):
        path = self.lab.state()["output_dir"]
        if not path:
            raise ValueError("尚无结果目录")
        subprocess.run(["xdg-open", str(path)], check=True, timeout=5, capture_output=True)

    def close(self):
        if self._closed:
            return
        try:
            self.lab.close()
        except Exception as error:
            self._error(error)
            return
        self._closed = True
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
        self.root.destroy()
