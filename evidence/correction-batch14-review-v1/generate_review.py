"""Reproduce the local 14-record review; no raw data edits or training."""
import html
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

SESSION = "interactive_lab_20261018_172125_1792315285667491937"
OUT = ROOT / ".local/correction-batch14-review-v1"


def main():
    from correction_data import digest, load_frames, write_json
    from review_corrections import render_pair, summarize

    records, excluded, cards = [], [], []
    for path in sorted((ROOT / "corrections").glob("human_*/metadata.json")):
        metadata = json.loads(path.read_text())
        if SESSION not in metadata.get("source_scene", ""):
            continue
        if metadata["status"] != "approved":
            excluded.append({"source_directory": str(path.parent.relative_to(ROOT)),
                             "status": metadata["status"], "frames": metadata["frame_count"],
                             "metadata_sha256": digest(path), "reason": "User excludes drafts"})
            continue
        name = path.parent.name
        before_hash = digest(path)
        frames = load_frames(path.parent, metadata)
        summary = summarize(frames)
        summary.update(source_directory=str(path.parent.relative_to(ROOT)),
                       metadata_sha256=before_hash, scenario_key=metadata["scenario_key"],
                       scenario=metadata["scenario"], split=metadata["split"],
                       physical_control_dt=metadata["control_dt"], video_fps=10,
                       video_timing="10 FPS action-index replay, not wall-clock recording")
        video = OUT / f"{name}.mp4"
        if video.exists():
            raise FileExistsError(video)
        with imageio.get_writer(video, fps=10, macro_block_size=1) as writer:
            for index, frame in enumerate(frames):
                writer.append_data(np.asarray(render_pair(frame, index)))
        indices = set(range(0, len(frames), 30)) | {len(frames)-1, len(frames)-10}
        for segment in summary["gripper_segments"]:
            indices.update((segment["start"], segment["end"]))
        indices = sorted(indices)
        pages = []
        for page, start in enumerate(range(0, len(indices), 12)):
            chosen = indices[start:start+12]
            sheet = Image.new("RGB", (1536, 280*((len(chosen)+2)//3)), "#dddddd")
            for cell, index in enumerate(chosen):
                sheet.paste(render_pair(frames[index], index), ((cell%3)*512, (cell//3)*280))
            filename = f"{name}-p{page:02d}.png"
            sheet.save(OUT / filename)
            pages.append({"image": filename, "indices": chosen})
        summary.update(video=video.name, contact_pages=pages)
        assert digest(path) == before_hash
        records.append(summary)
        cards.append(f'<section><h2>{html.escape(name)} — {len(frames)} 帧</h2>'
                     f'<video controls preload="metadata" width="768" src="{video.name}"></video>'
                     + ''.join(f'<p><a href="{p["image"]}">关键帧 {p["indices"][0]}–{p["indices"][-1]}</a></p>'
                               f'<img width="100%" src="{p["image"]}">' for p in pages)
                     + '</section>')
        print(f"READY {name} frames={len(frames)} pages={len(pages)}", flush=True)
    assert len(records) == 14 and sum(r["frames"] for r in records) == 8122
    assert len(excluded) == 3
    report = {"status": "pending_visual_review", "session": SESSION,
              "records": records, "excluded_drafts": excluded,
              "raw_data_changed": False, "training_started": False}
    write_json(OUT / "inventory.json", report)
    (OUT / "index.html").write_text(
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>14段人工示范复核</title>'
        '<style>body{font:18px sans-serif;max-width:1100px;margin:30px auto;line-height:1.6}'
        'section{border-top:1px solid #888;margin-top:30px}</style>'
        '<h1>新采集14段人工示范</h1><p>仅展示合格原件，3段草稿按用户要求排除。'
        '左全景，右腕部；每动作1帧，以10FPS回放，不是真实操作耗时。'
        '这里只读校验和生成回放，不会启动训练。</p>' + ''.join(cards) + '</html>', encoding="utf-8")


if __name__ == "__main__":
    main()
