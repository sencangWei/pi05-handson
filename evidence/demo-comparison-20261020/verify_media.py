"""Validate saved trajectories, output videos, source identity, and freeze behavior."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import imageio.v2 as iio
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_negative_recovery import analyze  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assets = ROOT / "exports/demo_comparison_20261020"
    sources = json.loads((HERE / "sources.json").read_text())
    render = json.loads((HERE / "render.json").read_text())
    first_state = HERE / sources[0]["archived_scene"] / "initial_state.npy"
    last_state = HERE / sources[-1]["archived_scene"] / "initial_state.npy"
    assert first_state.read_bytes() == last_state.read_bytes()
    checks = []
    for source in sources:
        directory = HERE / source["archived_scene"]
        for name, value in source["source_artifacts"].items():
            assert sha(directory / name) == value["sha256"]
        assert json.loads((directory / "scene.json").read_text()) == source["scene"]
        assert json.loads((directory / "result.json").read_text()) == source["result"]
        with (directory / "actions.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        metrics = analyze(rows, 800)
        assert metrics == source["recomputed_metrics"]
        if source is not sources[0]:
            assert metrics["followup"]["strict_success"] is True
        assert sha(assets / source["published_original"]) == source["screen_video_sha256"]
        checks.append({"demo": source["demo"], "controls": metrics["control_steps"],
                       "strict_first_release_followup": metrics["followup"]["strict_success"]})
    for output in render["comparisons"]:
        path = assets / output["file"]
        assert sha(path) == output["sha256"]
        for source in output["inputs"]:
            assert sha(ROOT / source["path"]) == source["sha256"]
        with (HERE / f"{path.stem}-decode.log").open("w") as log:
            subprocess.run(["ffmpeg", "-v", "error", "-threads", "2", "-i", str(path),
                            "-map", "0:v:0", "-threads", "2", "-f", "null", "-"],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        assert not (HERE / f"{path.stem}-decode.log").read_text()
    reader = iio.get_reader(assets / "negative10-gui-before-after.mp4")
    early = reader.get_data(10 * 15).astype(float)
    later = reader.get_data(25 * 15).astype(float)
    frozen_a = reader.get_data(60 * 15).astype(float)
    frozen_b = reader.get_data(90 * 15).astype(float)
    reader.close()
    left_motion = float(np.mean(np.abs(early[128:768, :1278] - later[128:768, :1278])))
    right_motion = float(np.mean(np.abs(early[128:768, 1282:] - later[128:768, 1282:])))
    right_freeze_error = float(np.mean(np.abs(frozen_a[128:768, 1282:] - frozen_b[128:768, 1282:])))
    assert left_motion > 1 and right_motion > 1 and right_freeze_error < 2
    inventory = []
    for path in sorted(assets.glob("*.mp4")):
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_frames:format=duration",
            "-of", "json", str(path)], text=True))
        assert probe["streams"][0]["codec_name"] == "h264"
        inventory.append({"name": path.name, "bytes": path.stat().st_size,
                          "sha256": sha(path), "probe": probe})
    assert len(inventory) == 6
    result = {"status": "PASS", "assets": inventory, "source_checks": checks,
              "comparison_outputs_fully_decoded": 2, "initial_state_files_equal": True,
              "gui_alignment": "Run-event aligned approximately; both1x; not matched policy RNG",
              "camera_alignment": "One control step per frame,10FPS; not wall-clock",
              "body_pixel_mean_absolute_difference": {
                  "left_motion_10to25s": left_motion, "right_motion_10to25s": right_motion,
                  "right_frozen_60to90s": right_freeze_error}}
    (HERE / "media-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "assets": len(inventory), "outputs_decoded": 2,
                      "new_demos_pass": 3, "freeze_error": right_freeze_error}))


if __name__ == "__main__":
    main()
