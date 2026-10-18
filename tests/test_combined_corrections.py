import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from correction_data import CorrectionRecorder, digest, write_json
from prepare_corrections import SELECTION_RULE, prepare, verify_records
from test_correction_data import pa, pq, record
from test_expanded_public import fixture, prepare as prepare_public


def selection_fixture(root, name="human_selected"):
    import numpy as np
    from test_correction_data import TASK
    recorder = CorrectionRecorder(root / "corrections" / name, {
        "task": TASK, "split": "train", "scenario_key": name, "source_scene": "fake fixture", "control_dt": .05})
    for idx in range(30):
        state = np.zeros(8, np.float32)
        state[0] = idx / 100
        post = state.copy()
        post[2], post[6], post[7] = .3, .04, -.04
        image = np.full((256, 256, 3), idx, np.uint8)
        recorder.append(state, [idx / 100, 0, 0, 0, 0, 0, -1], {"image": image, "wrist_image": image},
                        post, [.2, 0, 0], idx >= 15)
    recorder.finish(True)
    clip = {"source_directory": str(recorder.directory.relative_to(root)),
            "metadata_sha256": digest(recorder.directory / "metadata.json"), "source_frame_count": 30,
            "start": 5, "stop": 25, "reason": "Fixture useful continuous correction"}
    path = root / "evidence/correction-selection-fixture.json"
    write_json(path, {"schema_version": 1, "task": TASK, "rule": SELECTION_RULE, "clips": [clip]})
    return path, clip


@unittest.skipIf(pa is None, "CPU Arrow environment required")
class CombinedExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        fixture(self.root)
        with contextlib.redirect_stdout(io.StringIO()):
            self.base_path = prepare_public(self.root)
        self.base = json.loads(self.base_path.read_text())

    def test_human_payloads_appended_public_and_holdout_unchanged(self):
        record(self.root, "first")
        record(self.root, "second", scenario="seed-2")
        record(self.root, "held", split="holdout", scenario="seed-3")
        record(self.root, "draft", approved=False)
        before = {str(p.relative_to(self.root)): digest(p) for p in self.root.rglob("*") if p.is_file()}
        result = prepare(self.root, "combined", self.base_path)
        manifest = json.loads(result.read_text())
        self.assertEqual(manifest["dataset_kind"], "expanded_public_corrections_v1")
        self.assertEqual((manifest["train_episodes"], manifest["train_frames"]), (39, 94))
        self.assertEqual((manifest["correction_train_episodes"], manifest["correction_train_frames"]), (2, 20))
        self.assertEqual(manifest["correction_holdout_episodes_excluded"], 1)
        self.assertEqual(manifest["holdout_source_episode_ids_excluded"], self.base["holdout_source_episode_ids_excluded"])
        self.assertEqual(manifest["normalization"], self.base["normalization"])
        verify_records(self.root, manifest["inputs"] + manifest["files"])
        for relative, checksum in before.items():
            self.assertEqual(digest(self.root / relative), checksum)
        output = self.root / manifest["dataset_dir"]
        original = self.root / self.base["dataset_dir"]
        for idx in range(37):
            relative = f"data/chunk-000/episode_{idx:06d}.parquet"
            self.assertEqual(digest(output / relative), digest(original / relative))
        for idx, offset in ((37, 74), (38, 84)):
            table = pq.read_table(output / f"data/chunk-000/episode_{idx:06d}.parquet")
            self.assertEqual(table["index"].to_pylist(), list(range(offset, offset + 10)))
            self.assertEqual(table["episode_index"].to_pylist(), [idx] * 10)
            self.assertEqual(table["actions"][0].as_py(), [0.] * 6 + [-1.])
        with self.assertRaises(FileExistsError):
            prepare(self.root, "combined", self.base_path)

    def test_tampered_base_or_human_refuses_publication(self):
        recorder = record(self.root)
        for path in (self.root / self.base["dataset_dir"] / "data/chunk-000/episode_000001.parquet",
                     recorder.directory / "frame-000000.npz"):
            before = path.read_bytes()
            path.write_bytes(before + b"tampered")
            with self.assertRaisesRegex(ValueError, "differs"):
                prepare(self.root, "invalid", self.base_path)
            self.assertFalse((self.root / "evidence/corrections-invalid-manifest.json").exists())
            self.assertFalse((self.root / "data/lerobot/course/libero_corrections_invalid").exists())
            path.write_bytes(before)

    def test_human_scenario_overlap_or_nonpublic_base_rejected(self):
        record(self.root)
        record(self.root, "held", split="holdout", scenario="seed-1")
        with self.assertRaisesRegex(ValueError, "scenario_key"):
            prepare(self.root, "overlap", self.base_path)
        write_json(self.base_path, dict(self.base, dataset_kind="expanded_public_corrections_v1"))
        with self.assertRaisesRegex(ValueError, "public-only"):
            prepare(self.root, "recursive", self.base_path)


    def test_selected_slice_preserves_raw_payloads_and_keeps_unselected_scenario_evidence(self):
        import numpy as np
        from PIL import Image
        selection, clip = selection_fixture(self.root)
        record(self.root, "human_unselected", scenario="unselected")
        record(self.root, "human_held", split="holdout", scenario="held")
        before = {str(p.relative_to(self.root)): digest(p) for p in (self.root / "corrections").rglob("*") if p.is_file()}
        result = prepare(self.root, "selected", self.base_path, selection=selection)
        manifest = json.loads(result.read_text())
        self.assertEqual((manifest["train_episodes"], manifest["train_frames"]), (38, 94))
        self.assertEqual((manifest["correction_train_episodes"], manifest["correction_train_frames"]), (1, 20))
        self.assertEqual(manifest["correction_holdout_episodes_excluded"], 1)
        self.assertEqual(manifest["selected_source_ranges"], [clip])
        self.assertEqual(manifest["curation_selection"], str(selection.relative_to(self.root)))
        self.assertEqual(manifest["scenarios"], {"train": ["human_selected", "unselected"], "holdout": ["held"]})
        self.assertIn(str(selection.relative_to(self.root)), [item["path"] for item in manifest["inputs"]])
        verify_records(self.root, manifest["inputs"] + manifest["files"])
        table = pq.read_table(self.root / manifest["dataset_dir"] / "data/chunk-000/episode_000037.parquet")
        self.assertEqual(table["frame_index"].to_pylist(), list(range(20)))
        self.assertEqual(table["index"].to_pylist(), list(range(74, 94)))
        np.testing.assert_array_equal(np.array(table["state"].to_pylist(), dtype=np.float32)[:, 0], np.arange(5, 25, dtype=np.float32) / 100)
        for idx, row in enumerate(table.to_pylist(), 5):
            self.assertEqual(row["state"][0], row["actions"][0])
            with Image.open(io.BytesIO(row["image"]["bytes"])) as image:
                self.assertTrue((np.asarray(image) == idx).all())
        for relative, checksum in before.items():
            self.assertEqual(digest(self.root / relative), checksum)

    def test_selection_rejects_bounds_duplicate_checksum_split_and_unsafe_source(self):
        selection, clip = selection_fixture(self.root)
        original = json.loads(selection.read_text())
        record(self.root, "human_held", split="holdout", scenario="held")
        held = self.root / "corrections/human_held/metadata.json"
        mutations = [
            {"clips": [clip, clip]},
            *({"clips": [{**clip, **change}]} for change in [
                {"start": -1}, {"stop": 31}, {"start": True}, {"stop": 14},
                {"source_frame_count": 31}, {"metadata_sha256": "0" * 64},
                {"source_directory": "/tmp/human_external"}, {"source_directory": "corrections/../corrections/human_selected"},
                {"source_directory": "corrections/human_missing"}, {"reason": ""},
                {"source_directory": "corrections/human_held", "metadata_sha256": digest(held), "source_frame_count": 10,
                 "start": 0, "stop": 10}]),
            {"schema_version": True}, {"task": "other task"}, {"rule": "stitch frames"}, {"clips": []},
        ]
        for change in mutations:
            write_json(selection, {**original, **change})
            with self.subTest(change=change), self.assertRaises(ValueError):
                prepare(self.root, "invalidselection", self.base_path, selection=selection)
            self.assertFalse((self.root / "evidence/corrections-invalidselection-manifest.json").exists())
        write_json(selection, original)
        with self.assertRaisesRegex(ValueError, "requires"):
            prepare(self.root, "legacyselection", selection=selection)
        with self.assertRaises(ValueError):
            prepare(self.root, "externalselection", self.base_path, selection="/tmp/selection.json")

    def test_selection_endpoint_and_unselected_scenario_overlap_are_checked(self):
        selection, clip = selection_fixture(self.root)
        value = json.loads(selection.read_text())
        write_json(selection, {**value, "clips": [{**clip, "start": 0, "stop": 10}]})
        with self.assertRaisesRegex(ValueError, "task_done"):
            prepare(self.root, "badendpoint", self.base_path, selection=selection)
        write_json(selection, value)
        record(self.root, "human_unselected", scenario="overlap")
        record(self.root, "human_held", split="holdout", scenario="overlap")
        with self.assertRaisesRegex(ValueError, "scenario_key"):
            prepare(self.root, "overlap", self.base_path, selection=selection)


if __name__ == "__main__":
    unittest.main()
