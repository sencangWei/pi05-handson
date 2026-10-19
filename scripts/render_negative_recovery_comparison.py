"""Render the preregistered user008/seed0 policy comparison from complete real recordings."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from correction_data import digest, write_json
from summarize_teacher_retrain import analyze

ROOT = Path(__file__).resolve().parents[1]
PRIMARY_CASE, PRIMARY_SEED = "user008", 0
FPS = 20
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def read_source(root, plan, arm):
    directory = root/"evaluations/negative-recovery-20261019"/arm/"seed-0"/PRIMARY_CASE
    result_path = directory/"result.json"
    if not result_path.exists():
        return None
    result = json.loads(result_path.read_text())
    if "video_frames" not in result:
        return None
    case = plan["evaluation_cases"][0]
    if case["case"] != PRIMARY_CASE or plan["policy_seeds"][0] != PRIMARY_SEED:
        raise ValueError("Primary case must remain the preregistered first case and seed")
    metadata = result["model_metadata"]
    if (metadata["model_stage"] != f"{plan['arms'][arm]}/{plan['checkpoint_step']}"
            or metadata["diagnostic_rng_reset_per_connection"] != PRIMARY_SEED
            or result["initial_state_sha256"] != json.loads((root/case["source_scene"]/"scene.json").read_text())["initial_state_sha256"]
            or result["source_scene_sha256"] != case["scene_sha256"]
            or Path(result["saved_scene_replay"]).resolve() != (root/case["source_scene"]).resolve()
            or digest(directory/"initial_state.npy") != case["initial_state_file_sha256"]
            or result["max_steps"] != 800 or result["replan_steps"] != 1
            or result["post_release_steps_requested"] != 50 or result["video_fps"] != 10):
        raise ValueError(f"{arm}: model, scene, seed or fixed evaluation contract differs")
    rows = list(csv.DictReader((directory/"actions.csv").open()))
    metrics = analyze(rows, result["bowl_edit"]["after_xyz"][2], budget=800)
    n = len(rows)
    first = metrics["first_stable_release_control_step"]
    follow = metrics["followup"]
    if (result["control_steps"] != n or result["video_frames"] != n+1
            or result["first_ten_step_release_retreat_control_steps"] != first
            or result["post_release_steps_executed"] != follow["observed_steps"]
            or result["post_release_followup_complete"] != follow["quantity_complete"]):
        raise ValueError(f"{arm}: independently recomputed events disagree with result")
    passed = bool(follow["quantity_complete"] and follow["strict_50_step_gate_maintained"]
                  and follow["maximum_bowl_drift_3d_m"] <= .005)
    video = directory/result["video"]
    if not video.exists():
        return None
    capture = cv2.VideoCapture(str(video))
    images = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame.shape != (256, 256, 3):
            raise ValueError(f"{arm}: expected one actual 256x256 global camera")
        images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    if len(images) != n+1 or input_fps != 10:
        raise ValueError(f"{arm}: incomplete video decode or input FPS mismatch")
    return {"directory": directory, "result": result, "rows": rows, "metrics": metrics,
            "strict_pass": passed, "images": images,
            "source_sha256": {name: digest(directory/name) for name in
                              ("scene.json", "initial_state.npy", "actions.csv", "result.json", result["video"])}}


def panel_text(draw, source, arm, step, finished, x, font, small):
    n = source["metrics"]["control_steps"]
    local = min(step, n)
    first = source["metrics"]["first_stable_release_control_step"]
    status = ("PASS" if source["strict_pass"] else "FAIL") if local >= n else "PENDING"
    current_on = source["rows"][local-1]["done"] if local else "N/A (initial)"
    color = "#7ee3a0" if status == "PASS" else "#ff9696" if status == "FAIL" else "#ffe09a"
    draw.text((x+12, 44), f"{arm.upper()}  |  step {local}/{n}", font=font, fill="white")
    draw.text((x+12, 76), f"Current On: {current_on} | release10+50: {status}", font=small, fill=color)
    marker = "FINISHED (frozen)" if finished else "ACTIVE" if local < n else "FINISHED"
    draw.text((x+12, 620), marker, font=font, fill="#ffe09a" if finished else "white")
    event = (str(first) if first is not None and local >= first else
             "not reached (censored at 800)" if local >= n else "pending")
    draw.text((x+12, 650), f"First strict release10: {event}", font=small, fill="white")
    drift = source["metrics"]["followup"]["maximum_bowl_drift_3d_m"]
    drift_text = f"{drift*1000:.2f}mm" if drift is not None and local >= n else "pending" if local < n else "N/A"
    draw.text((x+12, 678), f"Followup drift: {drift_text} | limit 5mm", font=small, fill="white")


def compose(sources, step):
    image = Image.new("RGB", (1024, 768), "#15191f")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT, 22)
    small = ImageFont.truetype(FONT, 18)
    draw.text((12, 8), "Fixed PRIMARY: user008 / policy seed 0 | TRAINING position (not heldout)", font=font, fill="white")
    for side, (arm, source) in enumerate(sources.items()):
        index = min(step, len(source["images"])-1)
        view = Image.fromarray(source["images"][index]).resize((512, 512), Image.Resampling.NEAREST)
        image.paste(view, (side*512, 104))
        panel_text(draw, source, arm, step, step > len(source["images"])-1, side*512, font, small)
    draw.line((512, 44, 512, 710), fill="#666666", width=2)
    draw.text((12, 716), "20 FPS = native physics 20 Hz; same control steps aligned", font=small, fill="white")
    draw.text((12, 742), "inference pauses omitted; not wall-time speed. One global camera per policy.", font=small, fill="#ffe09a")
    return np.asarray(image)


def render(root=ROOT, check_only=False):
    import imageio.v2 as imageio

    root = Path(root).resolve()
    evidence = root/"evidence/negative-recovery-20261019"
    plan_path = evidence/"experiment-plan.json"
    plan = json.loads(plan_path.read_text())
    sources = {arm: read_source(root, plan, arm) for arm in ("control", "candidate")}
    missing = [arm for arm, source in sources.items() if source is None]
    if missing:
        return {"status": "WAITING", "missing_complete_recordings": missing,
                "primary": {"case": PRIMARY_CASE, "policy_seed": PRIMARY_SEED}}
    if sources["control"]["result"]["initial_state_sha256"] != sources["candidate"]["result"]["initial_state_sha256"]:
        raise ValueError("Paired policy runs must start from the exact same integration state")
    if check_only:
        return {"status": "SOURCES_VALIDATED", "frames": {arm: len(s["images"]) for arm, s in sources.items()}}
    output = evidence/"user008-seed0-comparison.mp4"
    poster = output.with_suffix(".png")
    manifest_path = output.with_suffix(".json")
    count = max(len(source["images"]) for source in sources.values())
    sample_indices = sorted({0, min(100, count-1), min(300, count-1), count-1})
    expected = {}
    with imageio.get_writer(output, fps=FPS, codec="libx264", macro_block_size=16,
                            ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p"]) as writer:
        for step in range(count):
            frame = compose(sources, step)
            writer.append_data(frame)
            if step in sample_indices:
                expected[step] = frame
    poster_step = min(300, count-1)
    Image.fromarray(expected[poster_step]).save(poster)
    capture = cv2.VideoCapture(str(output))
    fps, size = capture.get(cv2.CAP_PROP_FPS), [int(capture.get(k)) for k in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)]
    decoded, comparisons = 0, []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        if decoded in expected:
            actual = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            # Compare camera-only crops to the exact source-frame composition, allowing lossy H264 encoding.
            for side, (arm, source) in enumerate(sources.items()):
                x = side*512
                mae = float(np.abs(actual[104:616, x+4:x+508].astype(float)
                                   - expected[decoded][104:616, x+4:x+508].astype(float)).mean())
                if mae > 8:
                    raise ValueError("Output does not match corresponding actual source frame")
                comparisons.append({"output_frame": decoded, "arm": arm,
                                    "source_frame": min(decoded, len(source["images"])-1),
                                    "camera_rgb_mae": mae})
        decoded += 1
    capture.release()
    if decoded != count or fps != FPS or size != [1024, 768]:
        raise ValueError("Rendered video decode/frame-count/FPS/size verification failed")
    manifest = {"schema_version": 1, "status": "VERIFIED", "primary_case": PRIMARY_CASE, "policy_seed": PRIMARY_SEED,
                "selection": "Fixed plan first case and seed before candidate results; failures never replaced by another case.",
                "evaluation_role": "training_position", "not_heldout_benchmark": True,
                "plan_sha256": digest(plan_path), "source_code_sha256": digest(Path(__file__)),
                "video": output.name, "video_sha256": digest(output), "poster": poster.name, "poster_frame": poster_step,
                "poster_sha256": digest(poster), "output_fps": FPS, "output_frames": count, "output_size": size,
                "timing": "Native20Hz control-step replay; original10FPS sources replayed at20FPS without dropping or inventing source steps. Inference pauses omitted; not wall-time speed.",
                "alignment": "Output frame i uses source frame min(i,N); initial frame0, one frame per action. Short side visibly frozen after its final frame.",
                "strict_pass_definition": "First10 consecutive On+open command+actual gap>=.06m+EEF-bowl>=.10m; then50 actual policy steps maintaining every term and bowl drift<=.005m.",
                "sources": {arm: {"directory": str(s["directory"].relative_to(root)),
                                  "sha256": s["source_sha256"], "input_fps": 10, "input_frames": len(s["images"]),
                                  "model_metadata": s["result"]["model_metadata"], "strict_pass": s["strict_pass"],
                                  "metrics": s["metrics"]} for arm, s in sources.items()},
                "decode_verification": {"frames": decoded, "fps": fps, "sample_source_frame_comparisons": comparisons}}
    write_json(manifest_path, manifest)
    return {"status": "VERIFIED", "video": str(output), "manifest": str(manifest_path),
            "frames": count, "strict_pass": {arm: s["strict_pass"] for arm, s in sources.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = render(args.root, args.check_only)
    print(json.dumps(result))
    return 2 if result["status"] == "WAITING" else 0


if __name__ == "__main__":
    raise SystemExit(main())
