import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import recovery_sampling as sampling
import train_corrections as train


def fixture(root):
    evidence = root / "evidence"
    evidence.mkdir()
    base = {"repo_id": "course/fixture", "seed": 42, "train_frames": 10}
    base_path = evidence / "base.json"
    base_path.write_text(json.dumps(base))
    profile = {"schema_version": 1, "seed": 42, "repo_id": base["repo_id"],
               "dataset_manifest": "evidence/base.json", "dataset_manifest_sha256": sampling.digest(base_path),
               "virtual_frames": 40, "groups": [
                   {"name": "public", "ranges": [[0, 8]], "virtual_count": 20},
                   {"name": "recovery", "ranges": [[8, 10]], "virtual_count": 20}]}
    profile_path = evidence / "focus.json"
    profile_path.write_text(json.dumps(profile))
    manifest = {**base, "training_sampling_profile": {
        "path": "evidence/focus.json", "sha256": sampling.digest(profile_path)}}
    return manifest, profile, profile_path


class RecoverySamplingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest, self.profile, self.path = fixture(self.root)

    def write_profile(self):
        self.path.write_text(json.dumps(self.profile))
        self.manifest["training_sampling_profile"]["sha256"] = sampling.digest(self.path)

    def test_uniform_export_has_no_override(self):
        self.assertIsNone(sampling.prepare_sampling(self.root, {"train_frames": 10}))

    def test_seeded_focus_retains_every_start_and_changes_probability(self):
        first, record = sampling.prepare_sampling(self.root, self.manifest)
        second, _ = sampling.prepare_sampling(self.root, self.manifest)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 40)
        self.assertEqual(set(first), set(range(10)))
        self.assertEqual(sum(i >= 8 for i in first), 20)
        self.assertEqual(record["groups"][1]["probability"], .5)
        self.assertEqual(record["sampling_indices_sha256"],
                         hashlib.sha256(json.dumps(first).encode()).hexdigest())

    def test_rejects_profile_tamper_and_changed_underlying_export(self):
        self.path.write_text(self.path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "checksum"):
            sampling.prepare_sampling(self.root, self.manifest)
        self.write_profile()
        changed = copy.deepcopy(self.manifest)
        changed["seed"] = 43
        with self.assertRaisesRegex(ValueError, "seed"):
            sampling.prepare_sampling(self.root, changed)
        changed = copy.deepcopy(self.manifest)
        changed["train_frames"] = 11
        with self.assertRaisesRegex(ValueError, "equal the pinned"):
            sampling.prepare_sampling(self.root, changed)

    def test_rejects_overlap_missing_frames_and_insufficient_repeats(self):
        cases = [([[7, 10]], 20, "overlap"), ([[9, 10]], 20, "cover exactly"),
                 ([[8, 11]], 20, "outside"), ([[8, 10]], 1, "retain every")]
        for ranges, count, message in cases:
            with self.subTest(message=message):
                self.profile["groups"][1].update(ranges=ranges, virtual_count=count)
                self.write_profile()
                with self.assertRaisesRegex(ValueError, message):
                    sampling.prepare_sampling(self.root, self.manifest)

    def test_remap_preserves_source_future_window_and_scope(self):
        class Dataset:
            def __len__(self):
                return 10

            def __getitem__(self, i):
                # Fake episode boundary: source itself pads at frame9.
                return list(range(i, 10)) + [9] * i

        dataset = Dataset()
        def original(*args):
            return dataset
        loader = SimpleNamespace(create_torch_dataset=original)
        indices, record = sampling.prepare_sampling(self.root, self.manifest)
        self.assertIs(sampling.install_sampling(loader, indices, record), original)
        mapped = loader.create_torch_dataset(SimpleNamespace(repo_id="course/fixture"), 10, None)
        for virtual, real in enumerate(indices):
            self.assertEqual(mapped[virtual], dataset[real])
        with self.assertRaisesRegex(ValueError, "dataset or action horizon"):
            loader.create_torch_dataset(SimpleNamespace(repo_id="wrong"), 10, None)
        with self.assertRaisesRegex(ValueError, "dataset or action horizon"):
            loader.create_torch_dataset(SimpleNamespace(repo_id="course/fixture"), 5, None)
        record["real_frames"] = 11
        with self.assertRaisesRegex(ValueError, "length differs"):
            loader.create_torch_dataset(SimpleNamespace(repo_id="course/fixture"), 10, None)

    def test_bad_profile_fails_before_model_configuration(self):
        self.path.write_text("tampered")
        with patch.object(train, "ROOT", self.root), patch.object(
                train, "validate_inputs", return_value=(self.manifest, self.root, self.root / "launch.json")), patch.object(
                train, "make_experiment") as make:
            with self.assertRaisesRegex(ValueError, "checksum"):
                train.main(["--manifest", str(self.path), "--parent-checkpoint", str(self.root),
                            "--experiment", "test", "--dry-run"])
            make.assert_not_called()

    def test_trainer_failure_restores_process_local_loader(self):
        loader = ModuleType("openpi.training.data_loader")
        def original(*args):
            return range(10)
        loader.create_torch_dataset = original
        package, training = ModuleType("openpi"), ModuleType("openpi.training")
        package.__path__, training.__path__ = [], []
        package.training, training.data_loader = training, loader
        modules = {"openpi": package, "openpi.training": training,
                   "openpi.training.data_loader": loader}
        def fail(cfg):
            self.assertIsNot(loader.create_torch_dataset, original)
            raise RuntimeError("test trainer failure")
        trainer = SimpleNamespace(main=fail)
        spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None))
        with patch.dict(sys.modules, modules), patch.object(train, "ROOT", self.root), patch.object(
                train, "validate_inputs", return_value=(self.manifest, self.root, self.root / "launch.json")), patch.object(
                train, "make_experiment", return_value=SimpleNamespace(seed=42)), patch.object(
                train, "build_record", return_value={"source_sha256": {}}), patch.object(
                train, "digest", return_value="sourcehash"), patch.object(
                train.importlib.util, "spec_from_file_location", return_value=spec), patch.object(
                train.importlib.util, "module_from_spec", return_value=trainer):
            with self.assertRaisesRegex(RuntimeError, "test trainer failure"):
                train.main(["--manifest", str(self.path), "--parent-checkpoint", str(self.root),
                            "--experiment", "test"])
        self.assertIs(loader.create_torch_dataset, original)

    def test_focused_alias_keeps_original_export_identity(self):
        from test_train_corrections import pa, training_fixture
        if pa is None:
            self.skipTest("Arrow required for real export fixture")
        root = self.root / "full"
        root.mkdir()
        original, parent = training_fixture(root)
        base = json.loads(original.read_text())
        # The small legacy fixture predates the expanded manifest's explicit seed.
        base["seed"] = 42
        base["train_frames"] = json.loads(
            (root / base["dataset_dir"] / "meta/info.json").read_text())["total_frames"]
        original.write_text(json.dumps(base))
        profile = {"schema_version": 1, "seed": base["seed"], "repo_id": base["repo_id"],
                   "dataset_manifest": str(original.relative_to(root)),
                   "dataset_manifest_sha256": sampling.digest(original),
                   "virtual_frames": base["train_frames"] * 2,
                   "groups": [{"name": "all", "ranges": [[0, base["train_frames"]]],
                               "virtual_count": base["train_frames"] * 2}]}
        path = root / "evidence/focus.json"
        path.write_text(json.dumps(profile))
        alias = root / "evidence/corrections-focused-manifest.json"
        focused = {**base, "training_sampling_profile": {
            "path": str(path.relative_to(root)), "sha256": sampling.digest(path)}}
        alias.write_text(json.dumps(focused))
        actual, _, _ = train.validate_inputs(alias, parent, "new", root=root)
        self.assertEqual(actual["repo_id"], base["repo_id"])
        self.assertIn("training_sampling_profile", actual)
        focused["train_frames"] += 1
        alias.write_text(json.dumps(focused))
        with self.assertRaisesRegex(ValueError, "equal the pinned"):
            train.validate_inputs(alias, parent, "new", root=root)


if __name__ == "__main__":
    unittest.main()
