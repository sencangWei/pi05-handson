"""Configuration and CLI checks only; never load weights or start training/inference."""
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import serve_checkpoint
import train_release_focus as training


class EntryValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.addCleanup(patch.stopall)
        patch.object(training, "ROOT", self.root).start()
        self.parent = self.root / "checkpoints/pi05_handson_lora/parent/9999"
        (self.parent / "params").mkdir(parents=True)
        (self.parent / "params/manifest.ocdbt").write_bytes(b"small manifest")
        self.stats = self.parent / "assets/course/libero_black_bowl/norm_stats.json"
        self.stats.parent.mkdir(parents=True)
        self.stats.write_text("{}")

    def test_experiment_names_are_ascii_and_cannot_escape(self):
        for name in ("course_001", "Focus-02", "a" * 80):
            self.assertEqual(training.validate_experiment(name), name)
        for name in ("", "../other", "/tmp/run", ".hidden", "a/b", "a b", "中文", "a\n", "a" * 81):
            with self.subTest(name=name), self.assertRaises(ValueError):
                training.validate_experiment(name)

    def test_positive_numeric_values(self):
        self.assertEqual(training.positive_int("3"), 3)
        self.assertEqual(training.positive_learning_rate("0.0002"), 0.0002)
        for value in (0, -1, "1.2", True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                training.positive_int(value)
        for value in ("nan", "inf", "-inf", 0, -0.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                training.positive_learning_rate(value)

    def test_parent_requires_course_numeric_step_and_manifest(self):
        self.assertEqual(training.validate_checkpoint(self.parent), self.parent)
        for path in (self.root / "outside/9999", self.parent.parent,
                     self.parent.parent / "latest", self.parent / "extra/1",
                     self.parent.parent / "10000"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                training.validate_checkpoint(path)

    def test_parent_symlink_cannot_escape_course_checkpoints(self):
        outside = self.root / "outside/123"
        (outside / "params").mkdir(parents=True)
        (outside / "params/manifest.ocdbt").write_text("outside")
        link = self.parent.parent / "123"
        link.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            training.validate_checkpoint(link)

    def test_server_requires_course_normalization_asset(self):
        self.assertEqual(serve_checkpoint.checkpoint_assets(self.parent), self.parent)
        self.stats.unlink()
        with self.assertRaises(ValueError):
            serve_checkpoint.checkpoint_assets(self.parent)

    def test_invalid_cli_never_imports_training_or_model_stack(self):
        original_import = __import__

        def guard(name, *args, **kwargs):
            if name.split(".")[0] in {"jax", "flax", "openpi", "course_config"}:
                self.fail(f"Heavy import before CLI validation: {name}")
            return original_import(name, *args, **kwargs)

        cases = [
            (training.main, ["--experiment", "../bad"]),
            (training.main, ["--steps", "0"]),
            (training.main, ["--batch-size", "-1"]),
            (training.main, ["--learning-rate", "nan"]),
            (training.main, ["--parent-checkpoint", "/tmp/outside/9999"]),
            (serve_checkpoint.main, []),
            (serve_checkpoint.main, ["--checkpoint", "/tmp/outside/9999"]),
            (serve_checkpoint.main, ["--checkpoint", str(self.parent), "--port", "65536"]),
        ]
        with patch("builtins.__import__", side_effect=guard), redirect_stderr(io.StringIO()):
            for main, argv in cases:
                with self.subTest(main=main.__module__, argv=argv), self.assertRaises(SystemExit) as error:
                    main(argv)
                self.assertEqual(error.exception.code, 2)

    def test_existing_checkpoint_or_launch_manifest_refuses_before_config(self):
        for kind in ("checkpoint", "manifest"):
            with self.subTest(kind=kind):
                name = f"exists_{kind}"
                path = (self.parent.parent.parent / name if kind == "checkpoint" else
                        self.root / "evidence" / f"{name}-launch.json")
                if kind == "checkpoint":
                    path.mkdir()
                else:
                    path.parent.mkdir(exist_ok=True)
                    path.write_text("original")
                with patch.object(training, "make_experiment") as make, self.assertRaises(FileExistsError):
                    training.main(["--experiment", name, "--parent-checkpoint", str(self.parent), "--dry-run"])
                make.assert_not_called()
                if kind == "manifest":
                    self.assertEqual(path.read_text(), "original")

    def test_record_uses_selected_configuration_and_parent_norm(self):
        stats = self.root / "assets/pi05_handson_lora/course/libero_black_bowl/norm_stats.json"
        stats.parent.mkdir(parents=True)
        stats.write_text("{}")
        for name in ("train_release_focus.py", "course_config.py"):
            source = self.root / "scripts" / name
            source.parent.mkdir(exist_ok=True)
            source.write_text("test source")
        cfg = SimpleNamespace(exp_name="custom", num_train_steps=1201, batch_size=2, seed=42,
                              model=SimpleNamespace(action_horizon=10),
                              checkpoint_dir=self.parent.parent.parent / "custom", save_interval=1000,
                              lr_schedule=SimpleNamespace(warmup_steps=20, peak_lr=0.0002, decay_lr=0.0002),
                              wandb_enabled=False)
        with patch.object(training.subprocess, "check_output", return_value="abc\n"):
            record = training.build_record(cfg, self.parent, {"original_frames": 10}, dry_run=True)
            self.assertEqual(record["new_updates"], 1201)
            self.assertEqual(record["batch_size"], 2)
            self.assertEqual(record["final_step_index"], 1200)
            self.assertEqual(record["checkpoint_save_steps"], [1000, 1200])
            self.assertEqual(record["learning_rate"]["peak"], 0.0002)
            self.assertEqual(record["parent_checkpoint"], str(self.parent))
            self.assertEqual(record["original_frames"], 10)
            for steps, expected in ((1, [0]), (1000, [999]), (1001, [1000]), (2000, [1000, 1999])):
                cfg.num_train_steps, cfg.save_interval = steps, min(1000, steps)
                self.assertEqual(training.build_record(cfg, self.parent, {}, dry_run=True)["checkpoint_save_steps"], expected)
            self.stats.write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, "Normalization differs"):
                training.build_record(cfg, self.parent, {}, dry_run=True)

    def test_server_passes_selected_checkpoint_and_identity_to_fake_policy(self):
        cfg = SimpleNamespace(name="pi05_handson_lora")
        make_config = Mock(return_value=cfg)
        policy = SimpleNamespace(metadata={})
        create_policy = Mock(return_value=policy)
        server = Mock()
        modules = {
            "course_config": SimpleNamespace(make_config=make_config),
            "jax": SimpleNamespace(devices=lambda _: ["fake device"]),
            "openpi.policies.policy_config": SimpleNamespace(create_trained_policy=create_policy),
            "openpi.serving.websocket_policy_server": SimpleNamespace(WebsocketPolicyServer=server),
        }
        with patch.dict(sys.modules, modules), redirect_stdout(io.StringIO()):
            serve_checkpoint.main(["--checkpoint", str(self.parent), "--port", "8756"])
        make_config.assert_called_once_with(experiment="parent")
        create_policy.assert_called_once_with(cfg, self.parent)
        self.assertEqual(policy.metadata["model_stage"], "parent/9999")
        self.assertEqual(policy.metadata["checkpoint_step_index"], "9999")
        self.assertEqual(policy.metadata["policy_sampling_seed_at_server_start"], 0)
        self.assertEqual(policy.metadata["params_manifest_sha256"], training.digest(self.parent / "params/manifest.ocdbt"))
        self.assertEqual(policy.metadata["norm_stats_sha256"], training.digest(self.stats))
        server.assert_called_once_with(policy, host="127.0.0.1", port=8756, metadata=policy.metadata)
        server.return_value.serve_forever.assert_called_once_with()


@unittest.skipUnless(importlib.util.find_spec("flax"), "Training configuration dependencies unavailable")
class TrainingConfigurationTests(unittest.TestCase):
    def test_defaults_and_uniform_preserve_original_training_contract(self):
        cfg = training.make_experiment()
        self.assertEqual(cfg.exp_name, "release_focus_2000")
        self.assertEqual(training.make_experiment(True).exp_name, "release_uniform_2000")
        self.assertEqual((cfg.num_train_steps, cfg.batch_size, cfg.seed), (2000, 4, 42))
        self.assertEqual((cfg.save_interval, cfg.keep_period), (1000, 1))
        self.assertEqual((cfg.lr_schedule.peak_lr, cfg.lr_schedule.decay_lr), (5e-5, 5e-5))
        self.assertEqual(cfg.model.action_horizon, 10)
        self.assertEqual(cfg.data.repo_id, "course/libero_black_bowl")
        self.assertEqual(cfg.weight_loader.params_path, str(training.PARENT / "params"))
        self.assertFalse(cfg.resume)
        self.assertFalse(cfg.overwrite)
        self.assertFalse(cfg.wandb_enabled)

    def test_custom_settings_update_actual_configuration(self):
        cfg = training.make_experiment(experiment="custom", steps=3, batch_size=2, learning_rate=0.0001)
        self.assertEqual((cfg.exp_name, cfg.num_train_steps, cfg.batch_size), ("custom", 3, 2))
        self.assertEqual(cfg.save_interval, 3)
        self.assertEqual((cfg.lr_schedule.peak_lr, cfg.lr_schedule.decay_lr), (0.0001, 0.0001))
        self.assertEqual(cfg.lr_schedule.decay_steps, 21)
        self.assertFalse(cfg.resume)


if __name__ == "__main__":
    unittest.main()
