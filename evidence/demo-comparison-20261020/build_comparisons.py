"""Render same-state, side-by-side comparisons from immutable saved recordings."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import imageio.v2 as iio
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def panels(name, width, lines):
    half = width // 2
    header = Image.new("RGB", (width, 128), "#142334")
    draw = ImageDraw.Draw(header)
    title_font = ImageFont.truetype(FONT, 26 if width == 1280 else 36)
    small = ImageFont.truetype(FONT, 17 if width == 1280 else 26)
    for x, title, model, color in (
        (20, "调整前：800步未完成放置", "ui_correct_1792434450272918886/999", "#ffb7ae"),
        (half + 20, "调整后：408步放稳并撤手", "ui_correct_1792480483802126335/1999", "#a6efd4"),
    ):
        draw.text((x, 8), title, font=title_font, fill=color)
        draw.text((x, 56), model, font=small, fill="white")
    footer = Image.new("RGB", (width, 256), "#142334")
    draw = ImageDraw.Draw(footer)
    font = ImageFont.truetype(FONT, 22 if width == 1280 else 30)
    for j, line in enumerate(lines):
        assert draw.textbbox((0, 0), line, font=font)[2] <= width - 40, line
        draw.text((20, 8 + j * 47), line, font=font, fill="#edf4fb")
    header_path, footer_path = HERE / f"{name}-header.png", HERE / f"{name}-footer.png"
    header.save(header_path)
    footer.save(footer_path)
    return header_path, footer_path


def render(name, paths, trims, durations, fps, width, footer, destination):
    total = math.ceil(max(durations) * fps) / fps
    header, bottom = panels(name, width, footer)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-n"]
    for path in paths:
        command += ["-i", str(path)]
    for path in (header, bottom):
        command += ["-loop", "1", "-framerate", str(fps), "-i", str(path)]
    filters = []
    for i, trim in enumerate(trims):
        filters.append(
            f"[{i}:v]trim=start={trim:.9f},setpts=PTS-STARTPTS,fps={fps},"
            f"scale={width // 2}:640:force_original_aspect_ratio=decrease:force_divisible_by=2,"
            f"pad={width // 2}:640:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,"
            f"tpad=stop_mode=clone:stop_duration={total:.9f},trim=duration={total:.9f}[v{i}]"
        )
    filters.append("[v0][v1]hstack=inputs=2,drawbox=x=iw/2-2:y=0:w=4:h=ih:color=white:t=fill[body]")
    filters.append("[2:v][body][3:v]vstack=inputs=3[layout]")
    filters.append(
        f"[layout]drawtext=fontfile={FONT}:textfile={HERE / 'ended.txt'}:"
        f"fontsize={22 if width == 1280 else 30}:fontcolor=white:"
        f"x={width // 2 + 20}:y=94:enable='gte(t,{durations[1]:.9f})'[out]"
    )
    command += ["-filter_complex_threads", "2", "-filter_complex", ";".join(filters),
                "-map", "[out]", "-an", "-t", f"{total:.9f}", "-r", str(fps),
                "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-threads", "4",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)]
    with (HERE / f"{name}-encoder.log").open("xb") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    probe = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=width,height,avg_frame_rate,nb_frames:format=duration", "-of", "json", str(destination)], text=True))
    assert probe["streams"][0]["nb_frames"] == str(round(total * fps))
    return {"file": destination.name, "sha256": sha(destination), "bytes": destination.stat().st_size,
            "inputs": [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in paths],
            "start_trim_s": trims, "source_playback_duration_s": durations,
            "fps": fps, "duration_s": total, "width": width, "height": 1024,
            "right_freeze_after_s": durations[1], "command": command, "probe": probe}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "exports/demo_comparison_20261020")
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    records = json.loads((HERE / "sources.json").read_text())
    before, after = records[0], records[-1]
    assert (HERE / before["archived_scene"] / "initial_state.npy").read_bytes() == (
        HERE / after["archived_scene"] / "initial_state.npy").read_bytes()
    (HERE / "ended.txt").write_text("已成功结束｜保持末帧")
    common = "处理：4条805帧人工纠错 + 旧示范回放；π0.5 LoRA继续微调2000步"
    sampling = "抽样：公开50% / 旧人工20% / 新纠错30%｜batch4｜lr1e−5｜动作窗口10，每步重规划"
    gui_paths = [destination / r["published_original"] for r in (before, after)]
    for path, rec in zip(gui_paths, (before, after), strict=True):
        assert sha(path) == rec["screen_video_sha256"]
    trims = [r["run_offset_s"] for r in (before, after)]
    durations = [float(r["probe"]["format"]["duration"]) - trim
                 for r, trim in zip((before, after), trims, strict=True)]
    gui = render("gui", gui_paths, trims, durations, 15, 2560, [
        "同一初始物理状态｜布局0｜碗 X = −10 cm，Y = −10 cm",
        common, sampling,
        "录屏原速1×；按运行按钮日志近似对齐；保留中途过程；结束的一侧保持末帧",
        "用户可旋转观察视角；两次推理随机流未配对；这是开发案例，不是泛化成功率",
    ], destination / "negative10-gui-before-after.mp4")
    camera_paths = [HERE / r["archived_scene"] / "video.mp4" for r in (before, after)]
    durations = [r["result"]["video_frames"] / r["result"]["video_fps"] for r in (before, after)]
    camera = render("camera", camera_paths, [0, 0], durations, 10, 1280, [
        "相同初始状态、固定相机｜布局0｜X/Y均−10 cm",
        "按控制步对齐：一帧一步，10 FPS回放；不是实际墙钟时间",
        "4条805帧人工纠错 + 旧数据回放；π0.5 LoRA微调2000步",
        "抽样50%/20%/30%｜batch4｜lr1e−5｜窗口10，每步重规划",
        "两次推理随机流未配对；右侧结束后保持末帧",
    ], destination / "negative10-camera-before-after.mp4")
    header = Image.open(HERE / "camera-header.png")
    footer = Image.open(HERE / "camera-footer.png")
    poster = Image.new("RGB", (1280, 1024), "black")
    poster.paste(header, (0, 0))
    for i, path in enumerate(camera_paths):
        reader = iio.get_reader(path)
        picture = Image.fromarray(reader.get_data(reader.count_frames() - 1)).resize((640, 640))
        reader.close()
        poster.paste(picture, (i * 640, 128))
    poster.paste(footer, (0, 768))
    poster.save(HERE / "comparison-poster.jpg", quality=92)
    (HERE / "render.json").write_text(json.dumps({"comparisons": [gui, camera]}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"rendered": [gui["file"], camera["file"]], "output": str(destination)}), flush=True)


if __name__ == "__main__":
    main()
