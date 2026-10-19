"""Validate approval order, focused aliases and a weight-free training dry run."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_placement_teacher as training
from correction_data import digest


class PlacementTrainingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "evidence").mkdir()
        self.parent = self.root / "checkpoints/pi05_handson_lora/parent/1999"
        self.launch = self.root / "evidence/run-launch.json"
        self.base = {"task": "fixture", "approval_type": "original_human_approval"}
        self.path = self.root / "evidence/placement-teacher-pilot-manifest.json"
        self.manifest = {"repo_id": "course/libero_placement_teacher_pilot",
                         "dataset_dir": "data/lerobot/course/libero_placement_teacher_pilot",
                         "seed": 42, "train_frames": 6, "base_manifest": {"path": training.BASE_MANIFEST},
                         "teachers": [], "holdout_episodes": 10, "holdout_frames": 100,
                         "holdout_source_episode_ids_excluded": [1, 2], "scenarios": {"holdout": ["held_out"]}}
        self.write()

    def write(self):
        self.path.write_text(json.dumps(self.manifest))

    def mocks(self):
        self.addCleanup(patch.stopall)
        human = patch.object(training.train_corrections, "validate_inputs", return_value=(self.base, self.parent, self.launch)).start()
        base = patch.object(training, "load_base", return_value=self.base).start()
        export = patch.object(training, "validate_export").start()
        patch.object(training, "ROOT", self.root).start()
        return human, base, export

    def test_original_approval_failure_stops_before_teacher_validation(self):
        human, base, export = self.mocks()
        human.side_effect = ValueError("Human approval failed")
        with self.assertRaisesRegex(ValueError, "Human approval"):
            training.validate_inputs(self.path, self.parent, "run")
        human.assert_called_once_with(self.root / training.BASE_MANIFEST, self.parent, "run", root=self.root)
        base.assert_not_called()
        export.assert_not_called()

    def test_base_change_and_generated_validation_failure_propagate(self):
        _, base, export = self.mocks()
        base.return_value = {**self.base, "changed": True}
        with self.assertRaisesRegex(ValueError, "base changed"):
            training.validate_inputs(self.path, self.parent, "run")
        export.assert_not_called()
        base.return_value = self.base
        export.side_effect = ValueError("Teacher source hash differs")
        with self.assertRaisesRegex(ValueError, "source hash"):
            training.validate_inputs(self.path, self.parent, "run")

    def test_plain_manifest_identity_and_wrong_repository_rejected(self):
        _, _, export = self.mocks()
        got = training.validate_inputs(self.path, self.parent, "run")
        self.assertEqual(got, (self.manifest, self.parent, self.launch))
        export.assert_called_once_with(self.root, self.manifest, self.base)
        self.manifest["repo_id"] = "course/libero_tracking_teacher_pilot"
        self.write()
        with self.assertRaisesRegex(ValueError, "dataset identity"):
            training.validate_inputs(self.path, self.parent, "run")
        with self.assertRaisesRegex(ValueError, "in evidence"):
            training.validate_inputs(self.root / "outside.json", self.parent, "run")

    def test_real_focused_profile_checksums_and_unchanged_data(self):
        self.mocks()
        profile = self.root / "evidence/focused-profile.json"
        profile.write_text(json.dumps({"schema_version": 1, "seed": 42, "repo_id": self.manifest["repo_id"],
                                      "dataset_manifest": str(self.path.relative_to(self.root)),
                                      "dataset_manifest_sha256": digest(self.path), "virtual_frames": 12,
                                      "groups": [{"name": "all", "ranges": [[0, 6]], "virtual_count": 12}]}))
        alias = self.root / "evidence/focused-alias.json"
        alias.write_text(json.dumps({**self.manifest, "training_sampling_profile": {
            "path": str(profile.relative_to(self.root)), "sha256": digest(profile)}}))
        validated, _, _ = training.validate_inputs(alias, self.parent, "run")
        self.assertEqual(validated["repo_id"], self.manifest["repo_id"])
        profile.write_text("{}")
        with self.assertRaisesRegex(ValueError, "checksum"):
            training.validate_inputs(alias, self.parent, "run")

    def test_provenance_pins_teacher_metadata_and_holdout(self):
        self.mocks()
        directory = self.root / "teacher"
        directory.mkdir()
        metadata = directory / "metadata.json"
        details = {"source_kind": "offline_placement_recovery_teacher", "human_actions_only": False,
                   "source_state_file_sha256": "state_hash", "source_code_sha256": {"generator": "code_hash"}}
        metadata.write_text(json.dumps(details))
        self.manifest["teachers"] = [{"directory": "teacher", "metadata_sha256": digest(metadata)}]
        record = {"source_sha256": {}}
        with patch.object(training, "digest", side_effect=lambda p: digest(p) if p.name == "metadata.json" else "code"):
            training.add_provenance(record, self.manifest)
        self.assertEqual(record["teacher_provenance"][0]["metadata"], details)
        self.assertEqual(record["base_holdout"]["source_episode_ids_excluded"], [1, 2])
        self.assertFalse(record["human_actions_only"])
        metadata.write_text("{}")
        with self.assertRaisesRegex(ValueError, "changed after"):
            training.add_provenance({}, self.manifest)

    def test_dry_run_defaults_and_custom_config_never_import_trainer(self):
        self.mocks()
        cfg = SimpleNamespace(model=SimpleNamespace(discrete_state_input=False, action_horizon=10),
                              seed=42, resume=False, overwrite=False)
        for options, settings in [([], (2000, 4, 1e-5)),
                                  (["--steps", "3000", "--batch-size", "2", "--learning-rate", ".00002"], (3000, 2, 2e-5))]:
            with self.subTest(options=options), patch.object(training.train_corrections, "make_experiment", return_value=cfg) as make, \
                 patch.object(training.train_corrections, "build_record", return_value={"source_sha256": {}}), \
                 patch.object(training, "digest", return_value="code"), \
                 patch.object(training.importlib.util, "spec_from_file_location") as trainer, redirect_stdout(io.StringIO()):
                training.main(["--manifest", str(self.path), "--parent-checkpoint", str(self.parent),
                               "--experiment", "run", "--dry-run", *options])
                make.assert_called_once_with(self.manifest["repo_id"], self.parent, "run", *settings)
                trainer.assert_not_called()
                self.assertFalse(self.launch.exists())

    def test_invalid_cli_or_model_contract_never_enters_trainer(self):
        self.mocks()
        argv = ["--manifest", str(self.path), "--parent-checkpoint", str(self.parent), "--experiment", "run", "--dry-run"]
        with patch.object(training.train_corrections, "make_experiment") as make, redirect_stderr(io.StringIO()):
            for bad in [["--steps", "0"], ["--batch-size", "-1"], ["--learning-rate", "nan"]]:
                with self.subTest(bad=bad), self.assertRaises(SystemExit):
                    training.main(argv + bad)
            make.assert_not_called()
        for state, horizon, seed, resume in [(True, 10, 42, False), (False, 5, 42, False),
                                              (False, 10, 7, False), (False, 10, 42, True)]:
            cfg = SimpleNamespace(model=SimpleNamespace(discrete_state_input=state, action_horizon=horizon),
                                  seed=seed, resume=resume, overwrite=False)
            with patch.object(training.train_corrections, "make_experiment", return_value=cfg), \
                 patch.object(training.train_corrections, "build_record") as record, self.assertRaisesRegex(ValueError, "fresh"):
                training.main(argv)
            record.assert_not_called()

    def test_sampling_hook_restored_when_fake_trainer_fails(self):
        self.mocks()
        cfg = SimpleNamespace(model=SimpleNamespace(discrete_state_input=False, action_horizon=10),
                              seed=42, resume=False, overwrite=False)
        original = object()
        loader = ModuleType("openpi.training.data_loader")
        loader.create_torch_dataset = original
        package = ModuleType("openpi")
        package.training = ModuleType("openpi.training")
        package.training.data_loader = loader
        modules = {"openpi": package, "openpi.training": package.training, "openpi.training.data_loader": loader}
        sample_record = {"seed": 42, "repo_id": self.manifest["repo_id"], "real_frames": 6}

        def fail(_):
            self.assertIsNot(loader.create_torch_dataset, original)
            raise RuntimeError("fake trainer failure")

        trainer = SimpleNamespace(main=fail)
        spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda _: None))
        with patch.dict(sys.modules, modules), \
             patch.object(training.train_corrections, "make_experiment", return_value=cfg), \
             patch.object(training.train_corrections, "build_record", return_value={"source_sha256": {}}), \
             patch.object(training, "digest", return_value="code"), \
             patch("recovery_sampling.prepare_sampling", return_value=(list(range(6)), sample_record)), \
             patch.object(training.importlib.util, "spec_from_file_location", return_value=spec), \
             patch.object(training.importlib.util, "module_from_spec", return_value=trainer), \
             redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "fake trainer"):
            training.main(["--manifest", str(self.path), "--parent-checkpoint", str(self.parent), "--experiment", "run"])
        self.assertIs(loader.create_torch_dataset, original)
        self.assertEqual(json.loads(self.launch.read_text())["recovery_sampling"], sample_record)


if __name__ == "__main__":
    unittest.main()
