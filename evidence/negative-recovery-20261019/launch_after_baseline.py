"""Start the approved local training only after valid paired-control inputs finish."""
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))
from correction_data import digest,write_json
from lab_jobs import LabJobs
from summarize_negative_recovery import summarize
progress=ROOT/"evidence/negative-recovery-20261019/evaluation-control-holding-repair-progress.json"
record=ROOT/".local/negative-recovery-20261019/training-process.json"
deadline=time.monotonic()+1800
write_json(record,{"status":"WAITING_FOR_CONTROL_EVALUATION"})
while True:
    state=json.loads(progress.read_text())
    if state["status"]!="RUNNING":break
    if time.monotonic()>deadline:raise TimeoutError("Control evaluation did not finish within30minutes")
    time.sleep(1)
assert state["status"]=="COMPLETE",state
summary=summarize(ROOT/"evidence/negative-recovery-20261019/experiment-plan.json",ROOT/"evaluations/negative-recovery-20261019")
assert summary["arms"]["control"]["completed_trials"]==20,summary["arms"]["control"]
assert all(r["status"]=="COMPLETE" for r in summary["runs"] if r["arm"]=="control")
write_json(ROOT/"evidence/negative-recovery-20261019/control-summary-before-training.json",summary)
manifest=ROOT/"evidence/placement-teacher-negative_quadrants_mix_v1-manifest.json"
dry=json.loads((ROOT/"evidence/negative-recovery-20261019/training-dry-run.json").read_text())
assert digest(manifest)=="ce68f5a773c907575864e06f5b82ad231c576de73653b4382ee88c19de590829"
assert dry["status"]=="PASS"
jobs=LabJobs();before=jobs.snapshot()
assert not before["model"]["running"] and not before["training"]["running"]
assert not (ROOT/"checkpoints/pi05_handson_lora/negative_quadrants_20261019").exists()
command=[str(ROOT/".venv/bin/python"),"-u",str(ROOT/"scripts/train_placement_teacher.py"),"--manifest",str(manifest),"--parent-checkpoint",str(ROOT/"checkpoints/pi05_handson_lora/retrain_teacher_1x_20261019/1999"),"--experiment","negative_quadrants_20261019","--steps","2000","--batch-size","4","--learning-rate","1e-5"]
job=jobs._launch("training",command)
write_json(record,{"status":"TRAINING_PROCESS_STARTED_INPUT_VALIDATION_FIRST","job":job,"command":command,"control_trials_validated":20,"before":before})
print(json.dumps(job),flush=True)
