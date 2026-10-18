import contextlib
import dataclasses
import io
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


if __name__ == "__main__":
    unittest.main()
