"""Launcher checks use temporary files and mocked children; no GUI or jobs."""
import fcntl
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import lab_launcher


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "实验目录 with spaces"
        self.root.mkdir()
        self.notification = patch.object(lab_launcher, "notify")
        self.notify = self.notification.start()
        self.addCleanup(self.notification.stop)

    def test_launch_uses_sim_python_and_keeps_lock_until_child_exit(self):
        def child(command, **kwargs):
            self.assertEqual(command, [str(self.root / ".venv-sim/bin/python"), "-u", "scripts/simulation_lab.py"])
            self.assertEqual(kwargs["cwd"], self.root)
            self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
            self.assertEqual(len(kwargs["pass_fds"]), 1)
            with (self.root / "logs/lab-launcher.lock").open("a+") as other:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            kwargs["stdout"].write("fake GUI output\n")
            return subprocess.CompletedProcess(command, 0)

        with patch.object(lab_launcher.subprocess, "run", side_effect=child) as run:
            self.assertEqual(lab_launcher.launch_lab(self.root), 0)
            self.assertEqual(lab_launcher.launch_lab(self.root), 0)
        self.assertEqual(run.call_count, 2)
        logs = list((self.root / "logs").glob("*.log"))
        self.assertEqual(len(logs), 2)
        self.assertTrue(all("fake GUI output" in log.read_text() for log in logs))
        self.notify.assert_not_called()

    def test_existing_lock_shows_message_without_launching(self):
        (self.root / "logs").mkdir()
        with (self.root / "logs/lab-launcher.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(lab_launcher.subprocess, "run") as run:
                self.assertEqual(lab_launcher.launch_lab(self.root), 0)
            run.assert_not_called()
        self.assertIn("已在运行", self.notify.call_args.args[0])
        self.assertEqual(list((self.root / "logs").glob("*.log")), [])

    def test_child_failure_dialog_names_actual_log(self):
        with patch.object(lab_launcher.subprocess, "run", return_value=subprocess.CompletedProcess([], 3)):
            self.assertEqual(lab_launcher.launch_lab(self.root), 3)
        log = next((self.root / "logs").glob("*.log"))
        self.assertIn(str(log), self.notify.call_args.args[0])
        self.assertTrue(self.notify.call_args.kwargs["error"])

    def test_missing_python_is_logged_and_reported(self):
        with patch.object(lab_launcher.subprocess, "run", side_effect=FileNotFoundError("missing python")):
            self.assertEqual(lab_launcher.launch_lab(self.root), 1)
        log = next((self.root / "logs").glob("*.log"))
        self.assertIn("missing python", log.read_text())
        self.assertIn(str(log), self.notify.call_args.args[0])

    def test_install_respects_localized_desktop_and_sets_trust(self):
        desktop = self.root / "中文桌面"
        with patch.object(lab_launcher.subprocess, "check_output", return_value=str(desktop) + "\n") as lookup:
            with patch.object(lab_launcher.subprocess, "run") as run:
                shortcut = lab_launcher.install_shortcut(self.root)
        lookup.assert_called_once_with(["xdg-user-dir", "DESKTOP"], text=True)
        self.assertEqual(shortcut.parent, desktop)
        self.assertEqual(shortcut.stat().st_mode & 0o777, 0o755)
        content = shortcut.read_text()
        self.assertIn("Name=π0.5 仿真实验室\n", content)
        self.assertIn(f'Exec="/usr/bin/python3" "{self.root}/scripts/lab_launcher.py"\n', content)
        self.assertIn("Terminal=false\n", content)
        run.assert_called_once_with(["gio", "set", str(shortcut), "metadata::trusted", "true"], check=True,
                                    timeout=5, capture_output=True, text=True)

    def test_install_preserves_unrelated_shortcut_and_symlink_target(self):
        desktop = self.root / "desktop"
        desktop.mkdir()
        shortcut = desktop / "pi05-simulation-lab.desktop"
        target = self.root / "unrelated.desktop"
        target.write_text("[Desktop Entry]\nExec=unrelated\n")
        for symlink in (False, True):
            with self.subTest(symlink=symlink):
                if symlink:
                    shortcut.symlink_to(target)
                else:
                    shortcut.write_text(target.read_text())
                with patch.object(lab_launcher.subprocess, "check_output", return_value=str(desktop)):
                    with patch.object(lab_launcher.subprocess, "run") as run:
                        with self.assertRaises(FileExistsError):
                            lab_launcher.install_shortcut(self.root)
                self.assertEqual(shortcut.read_text(), "[Desktop Entry]\nExec=unrelated\n")
                run.assert_not_called()
                shortcut.unlink()
        self.assertEqual(target.read_text(), "[Desktop Entry]\nExec=unrelated\n")

    def test_reinstall_updates_owned_shortcut_and_trust_failure_keeps_it(self):
        desktop = self.root / "desktop"
        with patch.object(lab_launcher.subprocess, "check_output", return_value=str(desktop)):
            with patch.object(lab_launcher.subprocess, "run"):
                shortcut = lab_launcher.install_shortcut(self.root)
            for error in (subprocess.CalledProcessError(1, "gio"), subprocess.TimeoutExpired("gio", 5)):
                with self.subTest(error=type(error).__name__):
                    with patch.object(lab_launcher.subprocess, "run", side_effect=error):
                        with patch("sys.stderr"):
                            self.assertEqual(lab_launcher.install_shortcut(self.root), shortcut)
                    self.assertIn("Name=π0.5 仿真实验室", shortcut.read_text())
                    self.assertEqual(shortcut.stat().st_mode & 0o777, 0o755)
        self.notify.assert_not_called()

    def test_exec_escapes_desktop_field_codes_and_reserved_characters(self):
        self.assertEqual(lab_launcher.desktop_quote('/tmp/a%b "$`\\c'),
                         '"/tmp/a%%b \\\\"\\\\$\\\\`\\\\\\\\c"')
        with self.assertRaises(ValueError):
            lab_launcher.desktop_quote("/tmp/new\nline")

    def test_install_flag_does_not_launch_gui(self):
        with patch.object(lab_launcher, "install_shortcut", return_value=Path("/tmp/lab.desktop")) as install:
            with patch.object(lab_launcher, "launch_lab") as launch:
                self.assertEqual(lab_launcher.main(["--install-shortcut"]), 0)
        install.assert_called_once_with()
        launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
