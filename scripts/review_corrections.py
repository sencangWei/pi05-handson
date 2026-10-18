"""Create local review videos and action summaries; never approve or trim demonstrations."""
import argparse
import html
import json
from pathlib import Path

import numpy as np

from correction_data import approval_checks, digest, load_frames, write_json


def summarize(frames):
    segments = []
    for index, frame in enumerate(frames):
        grip = float(frame["action"][6])
        command = "close" if grip > 0 else "open" if grip < 0 else "neutral"
        if not segments or segments[-1]["command"] != command:
            segments.append({"command": command, "start": index, "end": index})
        else:
            segments[-1]["end"] = index
    return {"frames": len(frames), "gripper_segments": segments,
            "close_frames": sum(float(f["action"][6]) > 0 for f in frames),
            "open_frames": sum(float(f["action"][6]) < 0 for f in frames),
            "on_frames": sum(bool(f["done"]) for f in frames),
            "approval_gates": approval_checks(frames),
            "curation_status": "pending_visual_review", "automatically_selected_for_training": False}


def render_pair(frame, index):
    from PIL import Image, ImageDraw
    picture = Image.new("RGB", (512, 280), "white")
    for column, key in enumerate(("image", "wrist_image")):
        picture.paste(Image.fromarray(frame[key]), (column * 256, 24))
    grip = float(frame["action"][6])
    caption = f"frame {index} | {'CLOSE' if grip > 0 else 'OPEN' if grip < 0 else 'NEUTRAL'} | post_On={bool(frame['done'])}"
    ImageDraw.Draw(picture).text((8, 6), caption, fill="black")
    return picture


def prepare_review(root, output):
    import imageio.v2 as imageio
    from PIL import Image
    root, output = Path(root).resolve(), Path(output).resolve()
    output.relative_to(root / ".local")  # Review images remain local rather than entering code backups.
    output.mkdir(parents=True, exist_ok=False)
    report = {"purpose": "Human review of complete correction trajectories, not automatic expert labeling",
              "training_data_modified": False, "episodes": []}
    cards = []
    for path in sorted((root / "corrections").glob("*/metadata.json")):
        metadata = json.loads(path.read_text())
        if metadata.get("status") != "approved":
            continue
        before = digest(path)
        frames = load_frames(path.parent, metadata)
        summary = summarize(frames)
        summary.update(source_directory=str(path.parent.relative_to(root)),
                       metadata_sha256=before, split=metadata["split"],
                       physical_control_dt=metadata["control_dt"], video_fps=10,
                       video_timing="one pre-action frame per control action at nominal 10 FPS; pauses omitted")
        name = path.parent.name
        video = output / f"{name}.mp4"
        # Display annotations do not change the source RGB or training labels.
        with imageio.get_writer(video, fps=10, macro_block_size=1) as writer:
            for index, frame in enumerate(frames):
                writer.append_data(np.asarray(render_pair(frame, index)))
        indices = set(np.linspace(0, len(frames) - 1, 12, dtype=int).tolist())
        for segment in summary["gripper_segments"]:
            indices.update((segment["start"], segment["end"]))
        indices = sorted(indices)
        sheet = Image.new("RGB", (1536, 280 * ((len(indices) + 2) // 3)), "#dddddd")
        for cell, index in enumerate(indices):
            sheet.paste(render_pair(frames[index], index), ((cell % 3) * 512, (cell // 3) * 280))
        sheet_name = f"{name}.png"
        sheet.save(output / sheet_name)
        summary.update(video=video.name, contact_sheet=sheet_name, video_frames=len(frames))
        if digest(path) != before:
            raise ValueError("Source metadata changed during review generation")
        report["episodes"].append(summary)
        cards.append(f"<section><h2>{html.escape(name)}：{len(frames)} 帧</h2>"
                     f"<p>闭爪 {summary['close_frames']} 帧；开爪 {summary['open_frames']} 帧。"
                     "终点检查通过；整段动作质量待复核。长闭爪本身不代表错误标签。</p>"
                     f'<video controls preload="metadata" width="768" src="{video.name}"></video>'
                     f'<p><a href="{sheet_name}">打开关键帧大图</a></p>'
                     f'<img style="max-width:100%" src="{sheet_name}" alt="动作关键帧">'
                     f'<details><summary>开合阶段（从第0帧计数）</summary><pre>{html.escape(json.dumps(summary["gripper_segments"], indent=2))}</pre></details></section>')
        print(f"Reviewed artifacts ready: {name}, {len(frames)} frames", flush=True)
    write_json(output / "review.json", report)
    (output / "index.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>人工纠正复核</title>'
        '<style>body{font:18px sans-serif;max-width:1100px;margin:32px auto;line-height:1.65}section{border-top:1px solid #bbb;margin-top:32px}</style>'
        '<h1>人工纠正：完整轨迹复核</h1><p>这些录像来自已保存的人工动作，不是模型评测结果。'
        '播放使用名义10帧/秒，不等于操作时的真实耗时；长视频可用播放器调速。</p>'
        '<p>逐段确认：是否主动脱离卡住状态、重新对准、稳定放置、松爪和撤手。'
        '记录正确连续片段的起止帧；错误尝试保留原件。本工具不自动裁剪、批准或加入训练。</p>'
        + ''.join(cards) + '</html>', encoding="utf-8")
    return output / "index.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare_review(args.project_root, args.output))


if __name__ == "__main__":
    main()
