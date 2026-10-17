"""Chinese controls for the simulation lab; the native viewer displays the scene."""

import math
import subprocess
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox, ttk


class LabPanel:
    def __init__(self, root, lab, *, test_mode=False):
        self.root, self.lab = root, lab
        self._closed = False
        self._tick_enabled = True
        self._after_id = None
        # Some bundled Tk builds expose core X fonts only; use a CJK font Tk actually lists.
        families = {name.lower() for name in tkfont.families(root)}
        family = "Noto Sans CJK SC" if "noto sans cjk sc" in families else "song ti"
        tkfont.nametofont("TkDefaultFont").configure(family=family, size=-18, weight="normal")
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure(".", font="TkDefaultFont")
        root.minsize(560, 510)
        root.title("LIBERO 仿真实验面板" + (" — 界面检查，未连接仿真" if test_mode else ""))
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.columnconfigure(0, weight=1)
        frame = ttk.Frame(root, padding=14)
        frame.grid(sticky="nsew")
        frame.columnconfigure(1, weight=1)
        ttk.Label(frame, text="任务：拿起盘子旁的黑碗，放到盘子上。",
                  wraplength=470).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(frame, text="三维窗口：空格暂停/继续，N 单步，Q 结束。",
                  wraplength=470).grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 12))
        self.layout_var = tk.IntVar(root, value=0)
        self.x_var = tk.DoubleVar(root, value=0.0)
        self.y_var = tk.DoubleVar(root, value=0.0)
        ttk.Label(frame, text="布局编号").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(frame, from_=0, to=lab.layout_count - 1, textvariable=self.layout_var,
                    width=8).grid(row=2, column=1, sticky="w")
        for row, name, variable in ((3, "碗 X 偏移 (cm)", self.x_var),
                                    (4, "碗 Y 偏移 (cm)", self.y_var)):
            ttk.Label(frame, text=name).grid(row=row, column=0, sticky="w")
            ttk.Scale(frame, from_=-10, to=10, variable=variable).grid(
                row=row, column=1, sticky="ew", padx=8, pady=5)
            ttk.Spinbox(frame, from_=-10, to=10, increment=0.5, format="%.1f",
                        textvariable=variable, width=7).grid(row=row, column=2)
        ttk.Label(frame, text="偏移基于原位置；加载生效，重置会保存上一轮。",
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
        self.info_var = tk.StringVar(root)
        ttk.Label(frame, textvariable=self.info_var, justify="left", wraplength=470).grid(
            row=7, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self.status_var = tk.StringVar(root)
        ttk.Label(frame, textvariable=self.status_var, justify="left", wraplength=470).grid(
            row=8, column=0, columnspan=3, sticky="w", pady=(8, 0))
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

    def _refresh(self):
        state = self.lab.state()
        available = state["ready"] and not state["finished"]
        self.run_button.configure(state="normal" if available else "disabled",
                                  text="暂停" if state["running"] else "运行")
        self.step_button.configure(state="normal" if available else "disabled")
        self.save_button.configure(state="normal" if state["ready"] else "disabled")
        layout = state["layout_id"] if state["layout_id"] is not None else "未加载"
        reached = "已达到" if state["on_reached"] else "未达到"
        model = state["model_name"]
        if state.get("model_path"):
            model += f"\n{state['model_path']}"
        self.info_var.set(f"当前布局：{layout}    控制步数：{state['control_steps']}\n\n"
                          f"官方 On 判据：{reached}；松爪撤手请观察\n\n模型：{model}")
        self.status_var.set(f"状态：{state['status']}\n\n结果目录：{state['output_dir']}")

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
        self._after_id = self.root.after(30, self._poll)

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
