"""One-shot local supervisor; only observes/launches this recorded training pair."""
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from lab_jobs import LabJobs
STATUS = ROOT / "evidence/retrain-20261019/run-progress.json"


def save(value):
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(STATUS)


def wait_training(jobs, experiment):
    expected = str(ROOT / "scripts/train_tracking_teacher.py")
    record = jobs._jobs["training"]
    assert expected in record["cmdline"] and experiment in record["cmdline"]
    deadline = time.monotonic() + 10800
    while jobs._owned_process("training") is not None:
        if time.monotonic() > deadline:
            raise TimeoutError("Recorded training exceeded three-hour observation budget; left untouched")
        time.sleep(5)
    checkpoint = jobs.validate_checkpoint(ROOT / "checkpoints/pi05_handson_lora" / experiment / "1999")
    assert (checkpoint / "train_state").is_dir()
    log = Path(record["log"]).read_text(errors="replace")
    assert "Traceback (most recent call last)" not in log, record["log"]
    assert "Finished asynchronous save" in log and str(checkpoint) in log
    return {"experiment": experiment, "checkpoint": str(checkpoint), "log": record["log"],
            "params_manifest_sha256": __import__("hashlib").sha256((checkpoint / "params/manifest.ocdbt").read_bytes()).hexdigest()}


def main():
    progress = {"status": "RUNNING", "phase": "training_control", "completed_training": []}
    save(progress)
    jobs = LabJobs(ROOT)
    try:
        progress["completed_training"].append(wait_training(jobs, "retrain_teacher_1x_20261019"))
        assert not jobs.snapshot()["model"]["running"]
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        command = [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "scripts/train_tracking_teacher.py"),
                   "--manifest", str(ROOT / "evidence/tracking-teacher-retrain_rate2_mix_v2-manifest.json"),
                   "--parent-checkpoint", str(ROOT / "checkpoints/pi05_handson_lora/ui_correct_1792311653585294069/1999"),
                   "--experiment", "retrain_teacher_2x_20261019", "--steps", "2000", "--batch-size", "4", "--learning-rate", "1e-5"]
        state = jobs._launch("training", command)
        (ROOT / "evidence/retrain-20261019/training-candidate-process.json").write_text(
            json.dumps({"command": command, "state": state, "status": "RUNNING"}, indent=2) + "\n")
        progress["phase"] = "training_candidate"
        save(progress)
        progress["completed_training"].append(wait_training(jobs, "retrain_teacher_2x_20261019"))
        progress["phase"] = "evaluation"
        save(progress)
        with (ROOT / "logs/retrain-20261019-evaluation-driver.log").open("xb") as output:
            driver = subprocess.Popen([str(ROOT / ".venv-sim/bin/python"), "-u", str(ROOT / ".local/retrain-20261019/run_evaluation.py")],
                                      cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = driver.wait(timeout=14400)
                if code:
                    raise subprocess.CalledProcessError(code, driver.args)
            except BaseException:
                # Only the process group created above belongs to this evaluation.
                try:
                    os.killpg(driver.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    driver.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(driver.pid, signal.SIGKILL)
                    driver.wait(timeout=5)
                raise
        progress["status"] = "COMPLETE"
        progress["phase"] = "training_and_matrix_complete_analysis_pending"
        save(progress)
    except BaseException as error:
        progress["status"] = "FAILED"
        progress["exception"] = repr(error)
        save(progress)
        raise


if __name__ == "__main__":
    main()
