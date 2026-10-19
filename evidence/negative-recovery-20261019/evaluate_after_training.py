"""Run the fixed candidate matrix once the approved training process and final save finish."""
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))
from correction_data import digest,write_json
job=json.loads((ROOT/".local/negative-recovery-20261019/training-process.json").read_text())["job"]
status=ROOT/".local/negative-recovery-20261019/candidate-evaluation-process.json"
write_json(status,{"status":"WAITING_FOR_FINAL_TRAINING_SAVE","training_job":job})
try: process=psutil.Process(job["pid"])
except psutil.NoSuchProcess: process=None
deadline=time.monotonic()+7200
while process is not None:
    try:
        if not process.is_running() or process.status()==psutil.STATUS_ZOMBIE:break
    except psutil.NoSuchProcess:break
    if time.monotonic()>deadline:raise TimeoutError("Training exceeded two-hour observation bound")
    time.sleep(2)
checkpoint=ROOT/"checkpoints/pi05_handson_lora/negative_quadrants_20261019/1999"
for relative in ["params/manifest.ocdbt","train_state/manifest.ocdbt","_CHECKPOINT_METADATA","assets/course/libero_black_bowl/norm_stats.json"]:
    assert (checkpoint/relative).is_file(),str(checkpoint/relative)
assert digest(checkpoint/"assets/course/libero_black_bowl/norm_stats.json")==digest(ROOT/"assets/pi05_handson_lora/course/libero_black_bowl/norm_stats.json")
log=Path(job["log"]).read_text(errors="replace")
assert "Traceback (most recent call last)" not in log
assert "1999" in log and "Done waiting for Save Finalize" in log
assert not (ROOT/"evaluations/negative-recovery-20261019/candidate").exists()
command=[str(ROOT/".venv/bin/python"),"-u",str(ROOT/"evidence/negative-recovery-20261019/run_evaluation.py"),"--arm","candidate"]
output=ROOT/"logs/negative-recovery-candidate-orchestration.log"
write_json(status,{"status":"RUNNING_CANDIDATE_MATRIX","checkpoint":str(checkpoint),"params_manifest_sha256":digest(checkpoint/"params/manifest.ocdbt"),"command":command,"log":str(output)})
with output.open("xb") as stream:
    result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
write_json(status,{"status":"COMPLETE" if result.returncode==0 else "FAILED","returncode":result.returncode,"command":command,"log":str(output)})
if result.returncode:raise SystemExit(result.returncode)
print("CANDIDATE_MATRIX_COMPLETE",flush=True)
