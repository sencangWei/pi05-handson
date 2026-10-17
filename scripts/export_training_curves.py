"""Convert existing text logs into TensorBoard events; does not train a model."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time


ROOT = Path(__file__).resolve().parents[1]
PHASES = [
    ("01_updates_0_to_200_batch1", "06-training-my-first-200.log"),
    ("02_updates_200_to_2000_batch1", "11-resume-to2000.log"),
    ("03_updates_2000_to_10000_batch4", "17-training-to10000-batch4.log"),
]


def parse_metrics(text):
    rows = {}
    for match in re.finditer(r"Step\s+(\d+):\s*([^\r\n]+)", text):
        values = {}
        for metric in re.finditer(r"(loss|grad_norm|param_norm)=([-+\d.eE]+)", match[2]):
            value = float(metric[2])
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite {metric[1]} at step {match[1]}")
            values[metric[1]] = value
        if "loss" in values:
            rows[int(match[1])] = values
    return sorted(rows.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".local/tensorboard")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Use a fresh TensorBoard output directory: {args.output}")
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    manifest = {"purpose": "historical text-log visualization; no training launched",
                "wall_time": "export time, not original training wall-clock time", "phases": []}
    for phase, filename in PHASES:
        path = ROOT / "logs" / filename
        rows = parse_metrics(path.read_text(errors="replace"))
        if not rows:
            raise ValueError(f"No training metrics in {path}")
        writer = EventFileWriter(str(args.output / phase))
        try:
            for step, values in rows:
                summary = Summary(value=[Summary.Value(tag=f"train/{k}", simple_value=v)
                                         for k, v in values.items()])
                writer.add_event(Event(wall_time=time.time(), step=step, summary=summary))
            writer.flush()
        finally:
            writer.close()
        manifest["phases"].append({"phase": phase, "source": str(path.relative_to(ROOT)),
                                   "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                   "logged_steps": len(rows), "first_step": rows[0][0], "last_step": rows[-1][0]})
    (args.output / "export-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
