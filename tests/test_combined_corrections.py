import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from correction_data import digest, write_json
from prepare_corrections import prepare, verify_records
from test_correction_data import pa, pq, record
from test_expanded_public import fixture, prepare as prepare_public


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


if __name__ == "__main__":
    unittest.main()
