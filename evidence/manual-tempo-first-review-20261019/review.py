"""Review the first new-speed human recording; never edit or select training data."""
import html
import json
from pathlib import Path
import sys

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from correction_data import approval_checks, digest, load_frames, write_json  # noqa: E402
from review_corrections import render_pair, summarize  # noqa: E402

SOURCE = ROOT / "corrections/human_1792404027525762765"
OUT = Path(__file__).resolve().parent


def main():
    video = OUT / "full-review.mp4"
    if video.exists():
        raise FileExistsError(video)
    metadata_path = SOURCE / "metadata.json"
    metadata_hash = digest(metadata_path)
    metadata = json.loads(metadata_path.read_text())
    assert metadata["status"] == "approved" and metadata["human_actions_only"]
    frames = load_frames(SOURCE, metadata)
    assert len(frames) == 250
    gates = approval_checks(frames)
    actions = np.stack([frame["action"] for frame in frames])
    states = np.stack([frame["state"] for frame in frames])
    movement = np.linalg.norm(np.diff(states[:, :3], axis=0), axis=1)*1000
    norms, counts = np.unique(np.round(np.linalg.norm(actions[:, :3], axis=1), 3), return_counts=True)
    summary = summarize(frames)
    summary.update(source=str(SOURCE.relative_to(ROOT)), metadata_sha256=metadata_hash,
                   physical_control_dt=metadata["control_dt"], playback_fps=20,
                   frame_count=250, movement_labels={str(float(n)): int(c) for n, c in zip(norms, counts)},
                   carry_interval_zero_based_half_open=[185, 205],
                   carry_fresh_pre_move_median_mm=float(np.median(movement[185:204])),
                   last_ten_eef_bowl_distance_m=[float(np.linalg.norm(f["post_state"][:3]-f["post_bowl_pos"]))
                                               for f in frames[-10:]],
                   review_status="KEEP_ALIGNMENT_CORRECTION_CANDIDATE; not automatically imported",
                   visual_review="42 sampled pre-action dual-camera frames, including every command-run boundary; no full physics replay.",
                   note="An initial descent is followed by lifting/repositioning before the first close. Retain as alignment correction; do not describe it as a failed closed-gripper regrasp. Next collect a single-pass approach.",
                   limits="Movement uses adjacent fresh pre-state. On/final gates use recorded post-state and are not contact-force or fifty-step acceptance. Playback omits operator pauses. One demonstration does not establish learned recovery.")
    samples = {}
    with imageio.get_writer(video, fps=20, macro_block_size=1, codec="libx264") as writer:
        for index, frame in enumerate(frames):
            picture = np.asarray(render_pair(frame, index))
            writer.append_data(picture)
            if index in [0, 70, 115, 165, 185, 220, 249]:
                samples[index] = picture
    capture = cv2.VideoCapture(str(video))
    decoded, checks = 0, []
    assert capture.get(cv2.CAP_PROP_FPS) == 20
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        assert bgr.shape == (280, 512, 3)
        if decoded in samples:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            error = float(np.abs(rgb[24:].astype(float)-samples[decoded][24:].astype(float)).mean())
            assert error < 8
            checks.append({"frame": decoded, "camera_rgb_mae": error})
        decoded += 1
    capture.release()
    assert decoded == 250
    assert metadata_hash == digest(metadata_path)
    # Re-read and checksum all250 original NPZs after rendering.
    assert len(load_frames(SOURCE, metadata)) == 250
    summary.update(approval_gates=gates, video_sha256=digest(video), decoded_frames=decoded,
                   source_frame_comparisons=checks, source_unchanged=True,
                   review_source_sha256=digest(Path(__file__)))
    write_json(OUT / "summary.json", summary)
    write_json(OUT / "source-metadata.json", metadata)
    Image.fromarray(samples[185]).save(OUT / "carry-poster.png")
    (OUT / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>新速度示范第一条复核</title>'
        '<style>body{max-width:1100px;margin:30px auto;font:18px sans-serif;line-height:1.6}video,img{max-width:100%}</style>'
        '<h1>第一条普通/精细移动示范：250控制步</h1>'
        '<p>原件已保存为合格；本页仅复核，没有改原动作、裁剪或启动训练。普通0.6共150帧，精细0.1共70帧，零平移30帧。</p>'
        '<p>持碗搬运第186–205控制步，新鲜观测相邻移动中位约7.87mm/步；末10步放置、开爪及撤手门通过。</p>'
        '<p>前段有一次下降后抬起再平移对准，建议保留作重新对准的纠错候选；下一条录一次对准后下降的普通完整路径。没有自动将整条选入训练。</p>'
        '<p>20FPS对应物理20Hz，每帧是执行该动作之前的双RGB；角标post_On属于动作之后。省略操作等待，不是墙钟耗时。末帧是第250个动作之前，不额外伪造最终画面。</p>'
        '<video controls preload="metadata" poster="carry-poster.png" width="1024" src="full-review.mp4"></video>'
        '<p><a href="summary.json">指标与SHA核验</a></p>'
        + ''.join(f'<p><a href="sheet-{i}.jpg">关键帧第{i+1}页</a></p><img src="sheet-{i}.jpg">' for i in range(3))
        + '<pre>' + html.escape(json.dumps(summary["gripper_segments"], ensure_ascii=False, indent=2)) + '</pre>')
    print(json.dumps({"source": str(SOURCE), "frames": 250, "source_unchanged": True,
                      "approval_gates": gates, "carry_mm_per_step": summary["carry_fresh_pre_move_median_mm"]}), flush=True)


if __name__ == "__main__":
    main()
