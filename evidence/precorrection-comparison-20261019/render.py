"""Render all seed0 paired recordings, aligned by control step with frozen tails."""
import hashlib
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path(__file__).resolve().parent
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)


def read_frames(directory, run):
    capture = cv2.VideoCapture(str(directory / "video.mp4"))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        assert frame.shape == (256, 256, 3)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    assert capture.get(cv2.CAP_PROP_FPS) == 10
    capture.release()
    assert len(frames) == run["metrics"]["control_steps"]+1
    assert hashlib.sha256((directory / "video.mp4").read_bytes()).hexdigest() == run["video_sha256"]
    return frames


def render_case(summary, case):
    runs = [next(r for r in summary["runs"] if r["arm"] == arm and r["case"] == case and r["policy_seed"] == 0)
            for arm in ["control", "candidate"]]
    assert all(run["status"] == "COMPLETE" for run in runs)
    assert runs[0]["initial_state_sha256"] == runs[1]["initial_state_sha256"]
    sources = [read_frames(Path(run["directory"]), run) for run in runs]
    count = max(map(len, sources))
    output = EVIDENCE / f"{case}-seed0-comparison.mp4"
    if output.exists():
        raise FileExistsError(output)
    samples = {}
    with imageio.get_writer(output, fps=20, codec="libx264", macro_block_size=16,
                            ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p"]) as writer:
        for step in range(count):
            image = Image.new("RGB", (1024, 672), "#15191f")
            draw = ImageDraw.Draw(image)
            draw.text((12, 8), f"Case: {case} | policy seed0 | development comparison", font=FONT, fill="white")
            for side, (run, frames) in enumerate(zip(runs, sources)):
                local = min(step, len(frames)-1)
                label = "Before manual corrections" if side == 0 else "Current experimental model"
                draw.text((side*512+12, 36), label, font=FONT, fill="white")
                marker = "FINAL FRAME HOLD" if step > local else "FINISHED" if local == len(frames)-1 else "ACTIVE"
                draw.text((side*512+12, 62), f"step {local}/{len(frames)-1} | {marker}", font=FONT, fill="#ffe09a")
                image.paste(Image.fromarray(frames[local]).resize((512, 512)), (side*512, 96))
                result = "PASS" if run["metrics"]["followup"]["strict_success"] is True else "FAIL within budget"
                draw.text((side*512+12, 614), f"Release10 + followup50: {result if local == len(frames)-1 else 'pending'}", font=FONT, fill="white")
            draw.text((12, 644), "20FPS = physics20Hz; aligned control steps; inference waits omitted (not wall time)", font=FONT, fill="#ffe09a")
            array = np.asarray(image)
            writer.append_data(array)
            if step in {0, min(100, count-1), min(300, count-1), count-1}:
                samples[step] = array
    Image.fromarray(samples[min(300, count-1)]).save(output.with_suffix(".png"))
    capture = cv2.VideoCapture(str(output))
    assert capture.get(cv2.CAP_PROP_FPS) == 20
    index, checks = 0, []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index in samples:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for side in [0, 1]:
                crop = (slice(98, 606), slice(side*512+2, side*512+510))
                error = float(np.abs(rgb[crop].astype(float)-samples[index][crop].astype(float)).mean())
                assert error < 8, error
                checks.append({"output_frame": index, "side": side,
                               "source_frame": min(index, len(sources[side])-1), "rgb_mae": error})
        index += 1
    capture.release()
    assert index == count
    manifest = {"status": "VERIFIED", "case": case, "policy_seed": 0,
                "selection": "Every preregistered case is rendered; primary negative_slow was fixed before results.",
                "frames": count, "fps": 20, "source_videos_sha256": [r["video_sha256"] for r in runs],
                "video_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "alignment": "framei = source min(i,N), initial frame0; visibly frozen after each shorter run.",
                "timing": "20FPS native physics20Hz, inference waits omitted; not wall-clock speed.",
                "decoded_source_comparisons": checks}
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"case": case, "frames": count, "verified": True}), flush=True)


if __name__ == "__main__":
    data = json.loads((EVIDENCE / "summary.json").read_text())
    assert data["status"] == "COMPLETE"
    for spec in data["plan"]["evaluation_cases"]:
        render_case(data, spec["case"])
