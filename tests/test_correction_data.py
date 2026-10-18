import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from correction_data import CorrectionRecorder, approval_checks, digest, load_frames, write_json
from prepare_corrections import file_record, prepare

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = pq = None

TASK = "pick up the black bowl next to the plate and place it on the plate"


def record(root, name="example", split="train", scenario="seed-1", approved=True):
    recorder = CorrectionRecorder(root / "corrections" / name, {
        "task": TASK, "split": split, "scenario_key": scenario,
        "source_scene": "evaluation/scene", "control_dt": .05,
    })
    before = np.zeros(8, np.float32)
    after = before.copy()
    after[6:] = [.04, -.04]
    image = np.zeros((256, 256, 3), np.uint8)
    for _ in range(10):
        recorder.append(before, [0, 0, 0, 0, 0, 0, -1],
                        {"image": image, "wrist_image": image}, after, [.2, 0, 0], True)
    if approved is not None:
        recorder.finish(approved)
    return recorder


def public_fixture(root):
    from PIL import Image
    from prepare_public_data import feature_stats
    dataset = root / "data/lerobot/course/libero_black_bowl"
    (dataset / "data/chunk-000").mkdir(parents=True)
    (dataset / "meta").mkdir()
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    schema = pa.schema([(key, image_type) for key in ("image", "wrist_image")] +
                       [("state", pa.list_(pa.float32(), 8)), ("actions", pa.list_(pa.float32(), 7)),
                        ("timestamp", pa.float32())] + [(key, pa.int64()) for key in
                        ("frame_index", "episode_index", "index", "task_index")],
                       metadata={b"huggingface": b'{"fixture":true}'})
    stream = io.BytesIO()
    Image.new("RGB", (256, 256)).save(stream, format="PNG")
    image = {"bytes": stream.getvalue(), "path": None}
    features = {key: {"dtype": "image" if key in ("image", "wrist_image") else "float32"} for key in schema.names}
    episodes, stats = [], []
    for idx in range(16):
        row = dict(image=image, wrist_image=image, state=[0.] * 8, actions=[0.] * 6 + [-1.],
                   timestamp=0., frame_index=0, episode_index=idx, index=idx, task_index=0)
        table = pa.Table.from_pylist([row], schema=schema)
        pq.write_table(table, dataset / f"data/chunk-000/episode_{idx:06d}.parquet")
        episodes.append({"episode_index": idx, "tasks": [TASK], "length": 1})
        stats.append({"episode_index": idx, "stats": feature_stats(table, features)})
    write_json(dataset / "meta/info.json", {"fps": 10, "total_episodes": 16, "total_frames": 16,
               "codebase_version": "v2.1", "features": features, "chunks_size": 1000,
               "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"})
    for filename, rows in (("episodes.jsonl", episodes), ("episodes_stats.jsonl", stats),
                           ("tasks.jsonl", [{"task_index": 0, "task": TASK}])):
        (dataset / "meta" / filename).write_text("".join(json.dumps(row) + "\n" for row in rows))
    (root / "evidence").mkdir()
    write_json(root / "evidence/data_manifest.json", {
        "ready": True, "destination": str(dataset.relative_to(root)), "train_episodes": 16,
        "train_frames": 16, "train_source_episode_ids": list(range(16)),
        "holdout_source_episode_ids": [16, 17, 18, 19], "task": TASK,
        "files": [file_record(root, p) for p in sorted(dataset.rglob("*")) if p.is_file()],
    })
    norm = root / "assets/pi05_handson_lora/course/libero_black_bowl/norm_stats.json"
    norm.parent.mkdir(parents=True)
    norm.write_text('{"fixed":true}')
    return dataset


class RecorderTests(unittest.TestCase):
    def test_approval_rejects_then_can_continue_and_draft_zero_is_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = CorrectionRecorder(root / "zero", {"task": TASK, "split": "train", "scenario_key": "s", "source_scene": "s", "control_dt": .05})
            with self.assertRaisesRegex(ValueError, "ten_frames"):
                recorder.finish(True)
            self.assertTrue(recorder.active)
            recorder.finish(False)
            self.assertEqual(json.loads((recorder.directory / "metadata.json").read_text())["frame_count"], 0)
            approved = record(root)
            self.assertFalse(approved.active)
            metadata = json.loads((approved.directory / "metadata.json").read_text())
            self.assertEqual(metadata["status"], "approved")
            self.assertFalse(metadata["official_success_metric"])

    def test_actual_fingers_and_retreat_required(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = record(Path(directory), approved=None)
            path = recorder.directory / "frame-000009.npz"
            with np.load(path, allow_pickle=False) as source:
                frame = dict(source)
            frame["post_state"][6:] = 0
            with path.open("wb") as stream:
                np.savez_compressed(stream, **frame)
            recorder.metadata["files"][-1]["sha256"] = digest(path)
            with self.assertRaisesRegex(ValueError, "actual_fingers"):
                recorder.finish(True)
            self.assertTrue(recorder.active)

    def test_each_observation_gate_can_reject_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = record(Path(directory), approved=None)
            valid = load_frames(recorder.directory, recorder.metadata)
            for gate in ("task_done", "fingers", "command", "retreat"):
                frames = [{key: value.copy() for key, value in frame.items()} for frame in valid]
                if gate == "task_done":
                    frames[-1]["done"] = np.asarray(False)
                elif gate == "fingers":
                    frames[-1]["post_state"][6:] = 0
                elif gate == "command":
                    frames[-1]["action"][6] = 1
                else:
                    frames[-1]["post_bowl_pos"] = frames[-1]["post_state"][:3].copy()
                with self.subTest(gate=gate), self.assertRaises(ValueError):
                    approval_checks(frames)

    def test_bad_shapes_and_nonboolean_done(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = record(Path(directory), approved=None)
            image = np.zeros((256, 256, 3), np.uint8)
            with self.assertRaises(ValueError):
                recorder.append(np.zeros(8), [2] * 7, {"image": image, "wrist_image": image}, np.zeros(8), np.zeros(3), True)
            with self.assertRaises(ValueError):
                recorder.append(np.zeros(8), [0] * 7, {"image": image, "wrist_image": image}, np.zeros(8), np.zeros(3), 1)
            self.assertEqual(recorder.count, 10)


@unittest.skipIf(pa is None, "CPU Arrow environment required")
class ExportTests(unittest.TestCase):
    def test_arrow_roundtrip_originals_unchanged_holdout_and_draft_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = public_fixture(root)
            before = {p.name: digest(p) for p in original.rglob("*.parquet")}
            record(root)
            record(root, "holdout", "holdout", "seed-2")
            record(root, "draft", approved=False)
            record(root, "pending", approved=None)
            manifest_path = prepare(root, "test")
            manifest = json.loads(manifest_path.read_text())
            output = root / manifest["dataset_dir"]
            self.assertEqual(manifest["correction_train_episodes"], 1)
            self.assertEqual(manifest["correction_holdout_episodes_excluded"], 1)
            self.assertEqual({p.name: digest(p) for p in original.rglob("*.parquet")}, before)
            for name, checksum in before.items():
                self.assertEqual(digest(output / "data/chunk-000" / name), checksum)
            table = pq.read_table(output / "data/chunk-000/episode_000016.parquet")
            self.assertEqual(table.schema, pq.read_schema(original / "data/chunk-000/episode_000000.parquet"))
            self.assertEqual(table["index"].to_pylist(), list(range(16, 26)))
            self.assertEqual(table["episode_index"].to_pylist(), [16] * 10)
            self.assertEqual(table["actions"][0].as_py()[6], -1)
            self.assertTrue(table["image"][0].as_py()["bytes"].startswith(b"\x89PNG"))
            with self.assertRaises(FileExistsError):
                prepare(root, "test")

    def test_split_overlap_and_tamper_refuse_publication(self):
        for tamper in (False, True):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                public_fixture(root)
                recorder = record(root)
                if tamper:
                    with (recorder.directory / "frame-000000.npz").open("ab") as stream:
                        stream.write(b"tampered")
                else:
                    record(root, "holdout", "holdout", "seed-1")
                with self.assertRaises(ValueError):
                    prepare(root, "invalid")
                self.assertFalse((root / "evidence/corrections-invalid-manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
