"""Smoke-check the responsive GUI, then reset and leave it paused for the user."""
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "openpi/third_party/libero"))
os.environ.update(MUJOCO_GL="glx", PYOPENGL_PLATFORM="glx", TOKENIZERS_PARALLELISM="false")
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".local/libero"))
from simulation_lab import LiberoTask, PolicyConnection
from correction_workbench import CorrectionWorkbench
from lab_panel import LabPanel
import tkinter as tk

decision = json.loads((ROOT / "evidence/negative-recovery-20261019/deployment-decision.json").read_text())
config = json.loads((ROOT / ".local/responsive-loop-20261019/restore-config.json").read_text())
checkpoint = Path(config["checkpoint"])
with (ROOT / ".local/lab-panel.lock").open("a+") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    probe = PolicyConnection(8755)
    try:
        metadata = probe.metadata_only()
        assert metadata["checkpoint"] == str(checkpoint), metadata
    finally:
        probe.close()
    output = ROOT / "evaluations" / f"interactive_lab_responsive_{time.time_ns()}"
    lab = CorrectionWorkbench(output, LiberoTask(), port=8755)
    window = tk.Tk()
    try:
        panel = LabPanel(window, lab)
        window.title("LIBERO 仿真实验面板 — 流畅交互版（实验模型）")
        window.geometry("820x1260+10+60")
        panel.layout_var.set(config["layout_id"])
        panel.x_var.set(config["dx_m"] * 100)
        panel.y_var.set(config["dy_m"] * 100)
        label = next(k for k, value in panel.checkpoint_paths.items() if value == str(checkpoint))
        panel.model_var.set(label)
        parent_label = next(k for k, value in panel.checkpoint_paths.items() if value == str(ROOT / decision["recommended_parent_checkpoint"]))
        panel.parent_var.set(parent_label)
        lab.reset(config["layout_id"], config["dx_m"], config["dy_m"])
        assert lab.view.paused and lab.record.result["control_steps"] == 0
        lab.model_name = metadata["model_stage"]
        lab.check_connection()
        panel._refresh()
        def finish_ready():
            assert lab.view.paused and lab.record.result["control_steps"] == 0
            assert lab.record.result["control_loop"] == "responsive_v1"
            ready = {"status": "READY_PAUSED", "checkpoint": str(checkpoint), "metadata": metadata,
                     "gui_pid": os.getpid(), "output": str(output), "layout_id": config["layout_id"],
                     "dx_m": config["dx_m"], "dy_m": config["dy_m"], "state": lab.state(),
                     "model_selection_label": label, "policy_actions_executed": 0, "control_loop": "responsive_v1", "startup_smoke_actions": 20}
            (ROOT / ".local/responsive-loop-20261019/gui-ready.json").write_text(json.dumps(ready, indent=2) + "\n")
            lab.event("ready_for_user", checkpoint=str(checkpoint), policy_actions_executed=0)
            print("READY_PAUSED", json.dumps(ready), flush=True)

        smoke_started = time.monotonic()
        def watch_smoke():
            if time.monotonic() - smoke_started > 90:
                raise RuntimeError("Responsive GUI smoke timed out")
            if not smoke_stage[0] and lab.record.result["control_steps"] >= 20:
                lab.save("responsive_gui_20step_smoke")
                smoke = dict(lab.record.result)
                (ROOT / ".local/responsive-loop-20261019/ui-smoke.json").write_text(json.dumps(smoke, indent=2) + "\n")
                lab.reset(config["layout_id"], config["dx_m"], config["dy_m"])
                smoke_stage[0] = True
            if smoke_stage[0] and lab.pending is None:
                panel._refresh()
                finish_ready()
                return
            window.after(10, watch_smoke)
        smoke_stage = [False]
        lab.toggle_run()
        window.after(10, watch_smoke)
        window.mainloop()
    finally:
        lab.close()
