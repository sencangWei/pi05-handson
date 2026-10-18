import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image
    from prepare_expanded_public import REVISION, SOURCE, TASK, main, plan, prepare
    from prepare_corrections import file_record
    from prepare_public_data import digest, write_json, write_jsonl
except ImportError:
    pa = None


def fixture(root):
    raw = root / "data/raw_public"
    (raw / "meta").mkdir(parents=True)
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    schema = pa.schema([(key, image_type) for key in ("image", "wrist_image")] +
                       [("state", pa.list_(pa.float32(), 8)), ("actions", pa.list_(pa.float32(), 7)),
                        ("timestamp", pa.float32())] + [(key, pa.int64()) for key in
                        ("frame_index", "episode_index", "index", "task_index")],
                       metadata={b"huggingface": b'{"fixture":true}'})
    features = {key: {"dtype": "int64", "shape": [1]} for key in schema.names}
    features.update({key: {"dtype": "image", "shape": [256, 256, 3]} for key in ("image", "wrist_image")})
    features.update(state={"dtype": "float32", "shape": [8]}, actions={"dtype": "float32", "shape": [7]},
                    timestamp={"dtype": "float32", "shape": [1]})
    info = {"codebase_version": "v2.0", "fps": 10, "total_episodes": 47, "total_frames": 94,
            "total_tasks": 1, "features": features, "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"}
    write_json(raw / "meta/info.json", info)
    write_jsonl(raw / "meta/tasks.jsonl", [{"task_index": 36, "task": TASK}])
    write_jsonl(raw / "meta/episodes.jsonl", [{"episode_index": idx, "tasks": [TASK], "length": 2} for idx in range(47)])
    stream = io.BytesIO()
    Image.new("RGB", (256, 256), (12, 34, 56)).save(stream, format="PNG")
    image = {"bytes": stream.getvalue(), "path": None}
    for idx in range(47):
        rows = [{"image": image, "wrist_image": image, "state": [idx / 100.] * 8,
                 "actions": [.1] * 6 + [-1.], "timestamp": frame / 10, "frame_index": frame,
                 "episode_index": idx, "index": 500 + 2 * idx + frame, "task_index": 36} for frame in range(2)]
        destination = raw / f"data/chunk-000/episode_{idx:06d}.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), destination)
    # Only original 20 source episodes were pinned by the first experiment.
    pinned_paths = list((raw / "meta").iterdir()) + [raw / f"data/chunk-000/episode_{idx:06d}.parquet" for idx in list(range(16)) + list(range(43, 47))]
    write_json(root / "evidence/data_manifest.json", {"ready": True, "source_repo": SOURCE,
               "source_revision": REVISION, "task": TASK, "train_source_episode_ids": list(range(16)),
               "holdout_source_episode_ids": list(range(43, 47)), "train_episodes": 16, "train_frames": 32,
               "files": [file_record(root, path) for path in pinned_paths]})
    norm = root / "assets/pi05_handson_lora/course/libero_black_bowl/norm_stats.json"
    norm.parent.mkdir(parents=True)
    norm.write_text('{"fixed":true}')
    return raw


@unittest.skipIf(pa is None, "Training CPU Arrow environment required")
class ExpandedPublicTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw = fixture(self.root)

    def test_deterministic_split_preserves_old_split_and_plan_never_downloads(self):
        first = plan(self.root)
        second = plan(self.root)
        self.assertEqual(first, second)
        train, holdout = set(first["train_source_episode_ids"]), set(first["holdout_source_episode_ids_excluded"])
        self.assertEqual((len(train), len(holdout)), (37, 10))
        self.assertFalse(train & holdout)
        self.assertEqual(train | holdout, set(range(47)))
        self.assertTrue(set(range(16)) <= train)
        self.assertTrue(set(range(43, 47)) <= holdout)
        (self.raw / "data/chunk-000/episode_000016.parquet").unlink()
        with patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("network forbidden")):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                main(["--project-root", str(self.root), "--plan"])
        self.assertEqual(json.loads(output.getvalue())["missing_source_episode_ids"], [16])
        self.assertFalse((self.root / "data/lerobot").exists())

    def test_export_preserves_payloads_and_old_artifacts_reindexes_both_splits(self):
        before = {str(path.relative_to(self.root)): digest(path) for path in self.root.rglob("*") if path.is_file()}
        # A malformed human correction must remain outside this public-only dataset.
        (self.root / "corrections/broken").mkdir(parents=True)
        (self.root / "corrections/broken/metadata.json").write_text("not JSON")
        with contextlib.redirect_stdout(io.StringIO()):
            manifest_path = prepare(self.root)
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["dataset_kind"], "expanded_public_v1")
        self.assertEqual((manifest["original_train_episodes"], manifest["original_train_frames"]), (37, 74))
        self.assertEqual(manifest["correction_train_episodes"], 0)
        self.assertFalse(manifest["normalization"]["recomputed"])
        self.assertEqual(len(manifest["inputs"]), 51)
        for key, value in before.items():
            self.assertEqual(digest(self.root / key), value)
        for directory_key, source_key in (("dataset_dir", "train_source_episode_ids"), ("holdout_dataset_dir", "holdout_source_episode_ids_excluded")):
            directory = self.root / manifest[directory_key]
            ids = manifest[source_key]
            info = json.loads((directory / "meta/info.json").read_text())
            self.assertEqual((info["total_episodes"], info["total_frames"]), (len(ids), len(ids) * 2))
            self.assertEqual(info["codebase_version"], "v2.1")
            self.assertEqual(json.loads((directory / "meta/tasks.jsonl").read_text())["task_index"], 0)
            for new_id, source_id in enumerate(ids):
                exported = pq.read_table(directory / f"data/chunk-000/episode_{new_id:06d}.parquet")
                source = pq.read_table(self.raw / f"data/chunk-000/episode_{source_id:06d}.parquet")
                for feature in ("image", "wrist_image", "state", "actions", "timestamp"):
                    self.assertTrue(exported[feature].equals(source[feature]))
                self.assertEqual(exported["index"].to_pylist(), [new_id * 2, new_id * 2 + 1])
                self.assertEqual(exported["episode_index"].to_pylist(), [new_id] * 2)
                self.assertEqual(exported["task_index"].to_pylist(), [0] * 2)
        with self.assertRaises(FileExistsError):
            prepare(self.root)

    def test_original_metadata_or_raw_tamper_refused(self):
        for relative in ("meta/info.json", "data/chunk-000/episode_000000.parquet"):
            path = self.raw / relative
            previous = path.read_bytes()
            path.write_bytes(previous + b" ")
            with self.subTest(relative=relative), self.assertRaisesRegex(ValueError, "differs from manifest"):
                plan(self.root)
            path.write_bytes(previous)

    def test_new_source_invalid_timestamp_task_or_payload_refuses_publication(self):
        path = self.raw / "data/chunk-000/episode_000016.parquet"
        original = pq.read_table(path)
        for feature, values in (("timestamp", [0., .2]), ("task_index", [999, 999]),
                                ("state", [[float("nan")] * 8] * 2),
                                ("image", [{"bytes": b"bad", "path": None}] * 2)):
            position = original.schema.get_field_index(feature)
            field = original.schema.field(position)
            modified = original.set_column(position, field, pa.array(values, type=field.type))
            pq.write_table(modified, path)
            with self.subTest(feature=feature), contextlib.redirect_stdout(io.StringIO()), self.assertRaises((ValueError, OSError)):
                prepare(self.root)
            self.assertFalse((self.root / "evidence/corrections-public37_v1-manifest.json").exists())
            self.assertFalse((self.root / "data/lerobot/course/libero_corrections_public37_v1").exists())
        pq.write_table(original, path)

    def test_input_changes_during_export_refused_before_publication(self):
        import prepare_expanded_public as exporter
        source = self.raw / "data/chunk-000/episode_000016.parquet"
        statistics = exporter.feature_stats
        changed = False

        def mutate_after_read(table, features):
            nonlocal changed
            result = statistics(table, features)
            if not changed and abs(table["state"][0].as_py()[0] - .16) < .001:
                changed = True
                source.write_bytes(source.read_bytes() + b" ")
            return result

        with patch.object(exporter, "feature_stats", side_effect=mutate_after_read), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "differs from manifest"):
                prepare(self.root)
        self.assertFalse((self.root / "evidence/corrections-public37_v1-manifest.json").exists())
        self.assertFalse((self.root / "data/lerobot/course/libero_corrections_public37_v1").exists())

    def test_existing_holdout_is_never_replaced(self):
        destination = self.root / "data/lerobot/course/libero_corrections_public37_v1_holdout"
        destination.mkdir(parents=True)
        marker = destination / "user-work.txt"
        marker.write_text("preserve me")
        with self.assertRaises(FileExistsError):
            prepare(self.root)
        self.assertEqual(marker.read_text(), "preserve me")
        self.assertFalse((self.root / "data/lerobot/course/libero_corrections_public37_v1").exists())

    def test_source_episode_overlap_and_duplicate_metadata_rejected(self):
        manifest_path = self.root / "evidence/data_manifest.json"
        original = json.loads(manifest_path.read_text())
        modified = dict(original, holdout_source_episode_ids=[0, 44, 45, 46])
        write_json(manifest_path, modified)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            plan(self.root)
        write_json(manifest_path, original)
        episodes_path = self.raw / "meta/episodes.jsonl"
        episodes = [json.loads(line) for line in episodes_path.read_text().splitlines()]
        episodes[-1]["episode_index"] = 0
        write_jsonl(episodes_path, episodes)
        for item in original["files"]:
            if item["path"] == str(episodes_path.relative_to(self.root)):
                item.update(file_record(self.root, episodes_path))
        write_json(manifest_path, original)
        with self.assertRaisesRegex(ValueError, "metadata counts"):
            plan(self.root)


if __name__ == "__main__":
    unittest.main()
