"""Open the selected evaluated model, paused at the negative-position test scene."""
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
checkpoint = ROOT / decision["gui_checkpoint"]
with (ROOT / ".local/lab-panel.lock").open("a+") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    probe = PolicyConnection(8755)
    try:
        metadata = probe.metadata_only()
        assert metadata["checkpoint"] == str(checkpoint), metadata
    finally:
        probe.close()
    output = ROOT / "evaluations" / f"interactive_lab_retrain_review_{time.time_ns()}"
    lab = CorrectionWorkbench(output, LiberoTask(), port=8755)
    window = tk.Tk()
    try:
        panel = LabPanel(window, lab)
        window.title("LIBERO 仿真实验面板 — " + decision["gui_label"])
        window.geometry("820x1260+10+60")
        panel.layout_var.set(24)
        panel.x_var.set(-4.00749063670412)
        panel.y_var.set(6.928838951310862)
        label = next(k for k, value in panel.checkpoint_paths.items() if value == str(checkpoint))
        panel.model_var.set(label)
        parent_label = next(k for k, value in panel.checkpoint_paths.items() if value == str(ROOT / decision["recommended_parent_checkpoint"]))
        panel.parent_var.set(parent_label)
        lab.reset(24, -0.040074906367041196, 0.06928838951310862)
        assert lab.view.paused and lab.record.result["control_steps"] == 0
        lab.model_name = metadata["model_stage"]
        lab.check_connection()
        panel._refresh()
        ready = {"status": "READY_PAUSED", "checkpoint": str(checkpoint), "metadata": metadata,
                 "gui_pid": os.getpid(), "output": str(output), "layout_id": 24,
                 "dx_m": -0.040074906367041196, "dy_m": 0.06928838951310862, "state": lab.state(),
                 "model_selection_label": label, "policy_actions_executed": 0}
        (ROOT / ".local/negative-recovery-20261019/gui-ready.json").write_text(json.dumps(ready, indent=2) + "\n")
        lab.event("ready_for_user", checkpoint=str(checkpoint), policy_actions_executed=0)
        print("READY_PAUSED", json.dumps(ready), flush=True)
        window.mainloop()
    finally:
        lab.close()
