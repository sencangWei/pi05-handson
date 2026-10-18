"""Raw human corrections, with explicit approval and observation gates."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

MAX_FRAMES = 5000
APPROVAL = "manual_user_with_observation_gates"
FRAME_KEYS = {"state", "action", "image", "wrist_image", "post_state", "post_bowl_pos", "done"}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_metadata(metadata):
    for key in ("task", "scenario_key", "source_scene"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"Missing correction metadata: {key}")
    if metadata.get("split") not in ("train", "holdout"):
        raise ValueError("Explicit train or holdout split is required")
    dt = metadata.get("control_dt")
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("control_dt must be finite and positive")


def validate_frame(frame):
    if set(frame) != FRAME_KEYS:
        raise ValueError("Unexpected raw frame fields")
    for key, shape in (("state", (8,)), ("action", (7,)), ("post_state", (8,)), ("post_bowl_pos", (3,))):
        array = np.asarray(frame[key])
        if array.shape != shape or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
            raise ValueError(f"Invalid finite {key} shape {shape}")
    if np.any(np.abs(frame["action"]) > 1):
        raise ValueError("Human actions must be inside [-1, 1]")
    for key in ("image", "wrist_image"):
        array = np.asarray(frame[key])
        if array.shape != (256, 256, 3) or array.dtype != np.uint8:
            raise ValueError(f"{key} must be uint8 RGB 256x256")
    done = np.asarray(frame["done"])
    if done.shape != () or done.dtype != np.bool_:
        raise ValueError("done must be a boolean scalar")


def approval_checks(frames):
    tail = frames[-10:]
    checks = {
        "ten_frames": len(tail) == 10,
        "task_done_last_ten": len(tail) == 10 and all(bool(f["done"]) for f in tail),
        "actual_fingers_open_last_ten": len(tail) == 10 and all(float(np.abs(f["post_state"][6:8]).sum()) >= .06 for f in tail),
        "open_command_last_ten": len(tail) == 10 and all(float(f["action"][6]) < 0 for f in tail),
        "eef_clear_of_bowl": bool(tail) and float(np.linalg.norm(tail[-1]["post_state"][:3] - tail[-1]["post_bowl_pos"])) >= .10,
    }
    if not all(checks.values()):
        raise ValueError("Approval observation gates failed: " + ", ".join(k for k, v in checks.items() if not v))
    return checks


def load_frames(directory, metadata):
    validate_metadata(metadata)
    records = metadata.get("files", [])
    count = metadata.get("frame_count")
    if type(count) is not int or count < 0 or count > MAX_FRAMES or len(records) != count:
        raise ValueError("Raw correction frame count differs")
    expected = [f"frame-{i:06d}.npz" for i in range(count)]
    if sorted(p.name for p in directory.glob("frame-*.npz")) != expected:
        raise ValueError("Raw correction frame inventory differs")
    frames = []
    for name, item in zip(expected, records, strict=True):
        path = directory / name
        if item.get("path") != name or path.is_symlink() or digest(path) != item.get("sha256"):
            raise ValueError(f"Raw correction checksum differs: {name}")
        with np.load(path, allow_pickle=False) as source:
            frame = {key: source[key] for key in source.files}
        validate_frame(frame)
        frames.append(frame)
    return frames


class CorrectionRecorder:
    def __init__(self, directory, metadata):
        validate_metadata(metadata)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.metadata = dict(metadata, status="pending", frame_count=0, files=[],
                             approval_type=None, human_actions_only=True,
                             pauses_excluded=True, max_frames=MAX_FRAMES)
        self.count = 0
        self.active = True
        write_json(self.directory / "metadata.json", self.metadata)

    def append(self, state8, action7, images, post_state8, post_bowl_pos3, done):
        if not self.active:
            raise ValueError("Correction recording is closed")
        if self.count >= MAX_FRAMES:
            raise ValueError(f"Correction episode is bounded to {MAX_FRAMES} actions")
        if set(images) != {"image", "wrist_image"}:
            raise ValueError("Both RGB cameras are required")
        frame = {"state": np.asarray(state8, dtype=np.float32),
                 "action": np.asarray(action7, dtype=np.float32),
                 "post_state": np.asarray(post_state8, dtype=np.float32),
                 "post_bowl_pos": np.asarray(post_bowl_pos3, dtype=np.float32),
                 "done": np.asarray(done), **images}
        validate_frame(frame)
        path = self.directory / f"frame-{self.count:06d}.npz"
        with path.open("xb") as handle:
            np.savez_compressed(handle, **frame)
        self.count += 1
        self.metadata["frame_count"] = self.count
        self.metadata["files"].append({"path": path.name, "sha256": digest(path)})
        write_json(self.directory / "metadata.json", self.metadata)

    def finish(self, approved):
        if not self.active or type(approved) is not bool:
            raise ValueError("Expected an active recording and explicit boolean approval")
        checks = None
        if approved:
            frames = load_frames(self.directory, self.metadata)
            checks = approval_checks(frames)
        self.metadata.update(status="approved" if approved else "draft",
                             approval_type=APPROVAL if approved else None,
                             observation_gates=checks, official_success_metric=False)
        write_json(self.directory / "metadata.json", self.metadata)
        self.active = False
        return self.directory
