"""Exercise actual Tk/manual recording in scratch, then leave user scene paused."""
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import tkinter as tk

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.update(MUJOCO_GL="glx", PYOPENGL_PLATFORM="glx")
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".local/libero"))
import numpy as np  # noqa: E402
from correction_data import load_frames, write_json  # noqa: E402
from correction_workbench import CorrectionWorkbench  # noqa: E402
from lab_panel import LabPanel  # noqa: E402
from simulation_lab import LiberoTask, PolicyConnection  # noqa: E402

EVIDENCE = Path(__file__).resolve().parent
LOCAL = ROOT / ".local/manual-tempo-controls-20261019"


def smoke(task):
    scratch = LOCAL / "smoke-root"
    scratch.mkdir(exist_ok=False)
    lab = CorrectionWorkbench(scratch / "evaluations/session", task, root=scratch, port=8755)
    window = tk.Tk()
    errors = []
    panel = LabPanel(window, lab)
    window.title("LIBERO — 采集校准检查（不进入训练）")
    lab.reset(24)

    def check():
        try:
            panel.notebook.select(panel.correction_page)
            panel._begin_correction()
            lab.correction.metadata.update(human_actions_only=False, diagnostic_only=True,
                                           recorded_by="automated GUI smoke, never approved for training")
            write_json(lab.correction.directory / "metadata.json", lab.correction.metadata)
            panel.jog_repeats_var.set("1")
            panel._refresh()
            panel.normal_motion_button.invoke()
            panel.jog_buttons[(0, 1)].invoke()
            assert lab.correction.count == 1
            panel.precision_motion_button.invoke()
            assert lab.correction.count == 1
            panel.jog_buttons[(0, 1)].invoke()
            panel.normal_motion_button.invoke()
            panel.jog_buttons[(3, 1)].invoke()
            assert lab.correction.count == 3
            directory = lab.finish_correction(False)
            metadata = json.loads((directory / "metadata.json").read_text())
            frames = load_frames(directory, metadata)
            expected = np.array([[.6, 0, 0, 0, 0, 0, -1], [.1, 0, 0, 0, 0, 0, -1],
                                 [0, 0, 0, .1, 0, 0, -1]])
            np.testing.assert_allclose(np.stack([f["action"] for f in frames]), expected)
            assert metadata["status"] == "draft" and metadata["diagnostic_only"]
            report = {"status": "PASS", "physics": "real MuJoCo, actual Tk button dispatch and CorrectionRecorder",
                      "policy_calls": 0, "actual_control_steps": 3, "recording_directory": str(directory),
                      "raw_actions": [f["action"].tolist() for f in frames],
                      "preset_switch_adds_steps": False, "metadata": metadata,
                      "not_training_data": True}
            write_json(EVIDENCE / "gui-recording-smoke.json", report)
            print("GUI_RECORDING_SMOKE_PASS", flush=True)
        except BaseException as error:
            errors.append(error)
        finally:
            panel.close()

    window.after(400, check)
    try:
        window.mainloop()
    finally:
        lab.close()
    if errors:
        raise errors[0]


def main():
    config = json.loads((LOCAL / "restore-config.json").read_text())
    with (ROOT / ".local/lab-panel.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        probe = PolicyConnection(8755)
        try:
            metadata = probe.metadata_only()
            assert metadata["checkpoint"] == config["checkpoint"]
        finally:
            probe.close()
        task = LiberoTask()
        smoke(task)
        output = ROOT / "evaluations" / f"interactive_lab_manual_tempo_{time.time_ns()}"
        lab = CorrectionWorkbench(output, task, root=ROOT, port=8755)
        window = tk.Tk()
        window.tk.call("tk", "appname", f"pi05-manual-tempo-{os.getpid()}")
        panel = LabPanel(window, lab)
        window.title("LIBERO 仿真实验面板 — 普通/精细采集（实验模型）")
        window.geometry("820x1260+10+60")
        panel.layout_var.set(config["layout_id"])
        panel.x_var.set(config["dx_m"]*100)
        panel.y_var.set(config["dy_m"]*100)
        label = next(k for k, v in panel.checkpoint_paths.items() if v == config["checkpoint"])
        panel.model_var.set(label)
        decision = json.loads((ROOT / "evidence/negative-recovery-20261019/deployment-decision.json").read_text())
        parent = str(ROOT / decision["recommended_parent_checkpoint"])
        panel.parent_var.set(next(k for k, v in panel.checkpoint_paths.items() if v == parent))
        lab.reset(config["layout_id"], config["dx_m"], config["dy_m"])
        lab.model_name = metadata["model_stage"]
        panel.notebook.select(panel.correction_page)
        panel._refresh()
        window.createcommand("lab_snapshot", lambda: json.dumps({"state": lab.state(),
            "correction": lab.correction_state(), "scale": panel.jog_scale_var.get(),
            "mode": panel.motion_mode_var.get()}))

        def ready():
            assert lab.view.paused and lab.record.result["control_steps"] == 0 and not lab.correction_active
            write_json(EVIDENCE / "gui-ready.json", {"status": "READY_PAUSED", "gui_pid": os.getpid(),
                "output": str(output), "restore_config": config, "checkpoint": config["checkpoint"],
                "state": lab.state(), "normal_translation": .6, "precision_translation": .1,
                "ui_rotation": .1, "policy_calls": 0, "recording_active": False})
            print("READY_PAUSED_FOR_RECORDING", os.getpid(), str(output), flush=True)
        window.after(700, ready)
        try:
            window.mainloop()
        finally:
            lab.close()


if __name__ == "__main__":
    main()
