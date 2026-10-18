import contextlib
import dataclasses
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_corrections as train
from test_correction_data import pa, prepare, public_fixture, record


def training_fixture(root):
    public_fixture(root)
    record(root)
    manifest = prepare(root, "test")
    parent = root / "checkpoints/pi05_handson_lora/parent/9"
    (parent / "params").mkdir(parents=True)
    (parent / "params/manifest.ocdbt").write_text("fake fixture; never loaded")
    norm = parent / "assets/course/libero_black_bowl/norm_stats.json"
    norm.parent.mkdir(parents=True)
    norm.write_text('{"fixed":true}')
    return manifest, parent


@unittest.skipIf(pa is None, "CPU Arrow environment required")
class TrainingTests(unittest.TestCase):
    def test_preflight_rejects_tamper_parent_norm_and_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, parent = training_fixture(root)
            manifest, _, _ = train.validate_inputs(manifest_path, parent, "new", root=root)
            norm = parent / "assets/course/libero_black_bowl/norm_stats.json"
            norm.write_text("different")
            with self.assertRaisesRegex(ValueError, "Normalization differs"):
                train.validate_inputs(manifest_path, parent, "new", root=root)
            norm.write_text('{"fixed":true}')
            (root / "evidence/new-launch.json").write_text("existing")
            with self.assertRaises(FileExistsError):
                train.validate_inputs(manifest_path, parent, "new", root=root)
            (root / "evidence/new-launch.json").unlink()
            table = root / manifest["dataset_dir"] / "data/chunk-000/episode_000016.parquet"
            with table.open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "Artifact differs"):
                train.validate_inputs(manifest_path, parent, "new", root=root)

    def test_invalid_manifest_fails_before_config_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "invalid.json"
            path.write_text("{}")
            with patch.object(train, "ROOT", root), patch.object(train, "make_experiment") as configure:
                with self.assertRaises(ValueError):
                    train.main(["--manifest", str(path), "--parent-checkpoint", str(root), "--experiment", "new", "--dry-run"])
                configure.assert_not_called()

    def test_dry_run_does_not_import_trainer_or_write_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, parent = training_fixture(root)
            with (patch.object(train, "ROOT", root),
                  patch.object(train, "make_experiment", return_value=object()) as configure,
                  patch.object(train, "build_record", return_value={"dry_run": True}),
                  patch.object(train.importlib.util, "spec_from_file_location") as load_trainer,
                  patch.dict(train.os.environ, {}, clear=False),
                  contextlib.redirect_stdout(io.StringIO()) as output):
                train.main(["--manifest", str(manifest_path), "--parent-checkpoint", str(parent),
                            "--experiment", "new", "--dry-run"])
                configure.assert_called_once()
                load_trainer.assert_not_called()
            self.assertIn("DRY_RUN_PASS", output.getvalue())
            self.assertFalse((root / "evidence/new-launch.json").exists())
            self.assertFalse((root / "checkpoints/pi05_handson_lora/new").exists())

    def test_asset_id_and_all_parent_weights_explicit_without_sampling_override(self):
        @dataclasses.dataclass(frozen=True)
        class Assets:
            asset_id: str
        @dataclasses.dataclass(frozen=True)
        class Data:
            repo_id: str
            assets: object = None
        @dataclasses.dataclass(frozen=True)
        class Schedule:
            peak_lr: float = 1.
            decay_lr: float = 1.
        @dataclasses.dataclass(frozen=True)
        class Config:
            data: Data
            lr_schedule: Schedule
            batch_size: int = 1
            weight_loader: object = None
            save_interval: int = 1
            keep_period: int = 0
        base = Config(Data("original"), Schedule())
        modules = {
            "course_config": SimpleNamespace(make_config=lambda steps, experiment: base),
            "openpi.training.config": SimpleNamespace(AssetsConfig=Assets),
            "openpi.training.weight_loaders": SimpleNamespace(CheckpointWeightLoader=lambda path: ("all_weights", path)),
        }
        with patch.dict(sys.modules, modules):
            cfg = train.make_experiment("course/libero_corrections_test", Path("/parent/9"), "new", 2000, 4, 5e-5)
        self.assertEqual(cfg.data.repo_id, "course/libero_corrections_test")
        self.assertEqual(cfg.data.assets.asset_id, "course/libero_black_bowl")
        self.assertEqual(cfg.weight_loader, ("all_weights", "/parent/9/params"))
        self.assertEqual(cfg.save_interval, 1000)
        self.assertEqual(cfg.lr_schedule.peak_lr, 5e-5)
        self.assertEqual(base.data.repo_id, "original")


@unittest.skipIf(pa is None, "CPU Arrow environment required")
class ExpandedTrainingTests(unittest.TestCase):
    def setUp(self):
        from test_expanded_public import fixture
        from prepare_expanded_public import prepare as expand
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        fixture(self.root)
        self.original_hash = train.digest(self.root / "evidence/data_manifest.json")
        with contextlib.redirect_stdout(io.StringIO()):
            self.path = expand(self.root)
        self.manifest = json.loads(self.path.read_text())
        self.parent = self.root / "checkpoints/pi05_handson_lora/parent/9"
        (self.parent / "params").mkdir(parents=True)
        (self.parent / "params/manifest.ocdbt").write_text("fixture; never loaded")
        norm = self.parent / "assets/course/libero_black_bowl/norm_stats.json"
        norm.parent.mkdir(parents=True)
        norm.write_text('{"fixed":true}')

    def validate(self):
        return train.validate_inputs(self.path, self.parent, "expanded", root=self.root)

    def test_expanded_public_accepts_separate_holdout_and_preserves_original(self):
        manifest, parent, _ = self.validate()
        self.assertEqual((manifest["train_episodes"], manifest["holdout_episodes"]), (37, 10))
        self.assertEqual(parent, self.parent)
        self.assertEqual(train.digest(self.root / "evidence/data_manifest.json"), self.original_hash)
        self.assertFalse((self.root / "evidence/expanded-launch.json").exists())

    def test_expanded_rejects_overlap_human_corrections_and_changed_source(self):
        mutations = [
            {"holdout_source_episode_ids_excluded": self.manifest["train_source_episode_ids"][:10]},
            {"correction_train_episodes": 1, "correction_train_frames": 10, "human_corrections_included": True},
            {"source_revision": "different"},
            {"dataset_kind": "unrecognized_public"},
            {"preserved_train_source_episode_ids": []},
            {"holdout_dataset_dir": self.manifest["dataset_dir"]},
            {"train_frames": 1},
        ]
        for values in mutations:
            self.path.write_text(json.dumps({**self.manifest, **values}))
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.validate()

    def test_expanded_rejects_holdout_inventory_and_rehashed_wrong_metadata(self):
        from prepare_corrections import file_record
        holdout = self.root / self.manifest["holdout_dataset_dir"]
        extra = holdout / "unexpected.txt"
        extra.write_text("not inventoried")
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.validate()
        extra.unlink()
        info_path = holdout / "meta/info.json"
        info = json.loads(info_path.read_text())
        info_path.write_text(json.dumps({**info, "total_episodes": 37}))
        updated = file_record(self.root, info_path)
        self.manifest["files"] = [updated if item["path"] == updated["path"] else item for item in self.manifest["files"]]
        self.path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "metadata counts"):
            self.validate()


@unittest.skipIf(pa is None, "CPU Arrow environment required")
class MixedExpandedTrainingTests(ExpandedTrainingTests):
    def setUp(self):
        super().setUp()
        self.base_path = self.path
        record(self.root, "human-one", scenario="shift-1")
        record(self.root, "human-two", scenario="shift-2")
        record(self.root, "human-holdout", split="holdout", scenario="shift-heldout")
        self.path = prepare(self.root, "mixed", base_manifest=self.base_path)
        self.manifest = json.loads(self.path.read_text())

    # Parent tests describe the public-only export; exercise mixed-specific gates instead.
    def test_expanded_public_accepts_separate_holdout_and_preserves_original(self):
        manifest, parent, _ = self.validate()
        self.assertEqual((manifest["train_episodes"], manifest["train_frames"]), (39, 94))
        self.assertEqual((manifest["holdout_episodes"], manifest["holdout_frames"]), (10, 20))
        self.assertEqual((manifest["correction_train_episodes"], manifest["correction_train_frames"]), (2, 20))
        self.assertEqual(manifest["correction_holdout_episodes_excluded"], 1)
        self.assertEqual(parent, self.parent)
        self.assertEqual(train.digest(self.root / "evidence/data_manifest.json"), self.original_hash)
        self.assertFalse((self.root / "evidence/expanded-launch.json").exists())

    def test_expanded_rejects_overlap_human_corrections_and_changed_source(self):
        mutations = [
            {"holdout_source_episode_ids_excluded": self.manifest["train_source_episode_ids"][:10]},
            {"correction_train_episodes": 1}, {"correction_train_frames": 100},
            {"human_corrections_included": False}, {"source_revision": "different"},
            {"sampling": "uniform over public training frames"}, {"train_frames": 1},
            {"correction_source_directories": []},
            {"timing": {**self.manifest["timing"], "physical_control_dt_by_episode": {}}},
            {"scenarios": {"train": ["shift-heldout"], "holdout": ["shift-heldout"]}},
            {"inputs": [item for item in self.manifest["inputs"] if item["path"] != str(self.base_path.relative_to(self.root))]},
        ]
        for values in mutations:
            self.path.write_text(json.dumps({**self.manifest, **values}))
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.validate()

    def test_expanded_rejects_holdout_inventory_and_rehashed_wrong_metadata(self):
        from prepare_corrections import file_record
        dataset = self.root / self.manifest["dataset_dir"]
        info_path = dataset / "meta/info.json"
        info = json.loads(info_path.read_text())
        info_path.write_text(json.dumps({**info, "total_frames": 1}))
        updated = file_record(self.root, info_path)
        self.manifest["files"] = [updated if item["path"] == updated["path"] else item for item in self.manifest["files"]]
        self.path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "metadata counts"):
            self.validate()

    def test_rehashed_public_parquet_change_is_still_rejected(self):
        import pyarrow.parquet as pq
        from prepare_corrections import file_record
        path = self.root / self.manifest["dataset_dir"] / "data/chunk-000/episode_000000.parquet"
        table = pq.read_table(path)
        pq.write_table(table.slice(0, 1), path)
        updated = file_record(self.root, path)
        self.manifest["files"] = [updated if item["path"] == updated["path"] else item for item in self.manifest["files"]]
        self.path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "public episode prefix changed"):
            self.validate()

    def test_rehashed_revoked_human_approval_is_rejected(self):
        from prepare_corrections import file_record
        path = self.root / "corrections/human-one/metadata.json"
        metadata = json.loads(path.read_text())
        path.write_text(json.dumps({**metadata, "status": "draft"}))
        updated = file_record(self.root, path)
        self.manifest["files"] = [updated if item["path"] == updated["path"] else item for item in self.manifest["files"]]
        self.path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "human approval"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
