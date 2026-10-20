"""Demo lifecycle and deletion boundaries, with no model or desktop capture."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from demo_recording import DemoRecording, KIND


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.demo = DemoRecording(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def saved(self, name, status="saved", kind=KIND):
        directory = self.root / "demos" / name
        directory.mkdir(parents=True)
        (directory / "demo.json").write_text(json.dumps({"kind": kind, "status": status}))
        (directory / "demo.mp4").write_bytes(b"saved-demo")
        return directory

    def test_deletion_recycles_latest_then_previous_and_preserves_training(self):
        old, new = self.saved("demo-100"), self.saved("demo-200")
        self.saved("demo-300", status="failed")
        self.saved("demo-400", kind="foreign")
        raw = self.root / "corrections/human_1/frame.npz"
        raw.parent.mkdir(parents=True)
        raw.write_bytes(b"human-source")
        self.demo.delete_last()
        self.assertFalse(new.exists())
        self.assertEqual(self.demo.last, old)
        recycled = list((self.root / ".local/demo-trash").glob("demo-200-*"))
        self.assertEqual((recycled[0] / "demo.mp4").read_bytes(), b"saved-demo")
        self.demo.delete_last()
        self.assertIsNone(self.demo.last)
        self.assertEqual(raw.read_bytes(), b"human-source")
        with self.assertRaises(ValueError):
            self.demo.delete_last()

    def test_recording_disables_deletion_and_symlinks_are_not_followed(self):
        self.demo.process = Mock()
        with self.assertRaises(ValueError):
            self.demo.delete_last()
        self.demo.process = None
        outside = self.root / "other"
        outside.mkdir()
        self.demo.directory.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            DemoRecording(self.root)

    def test_start_stop_finalizes_video_and_survives_reopening(self):
        process = Mock(returncode=0, stdin=io.BytesIO())
        process.poll.return_value = None
        probe = Mock(returncode=0, stdout=json.dumps({"streams": [{"codec_name": "h264", "width": 1000, "height": 600}],
                                                     "format": {"duration": "2.0"}}))
        region = dict(x=10, y=20, width=1000, height=600, window_ids=[1, 2])
        with patch("demo_recording.capture_region", return_value=region), patch("demo_recording.shutil.which", return_value="installed"):
            with patch.dict("os.environ", {"DISPLAY": ":0"}), patch("demo_recording.subprocess.Popen", return_value=process) as popen:
                self.demo.start(1, "evaluations/session/scene-1")
            command = popen.call_args.args[0]
            self.assertIn(":0+10,20", command)
            self.assertIn("1000x600", command)
            self.assertTrue(self.demo.recording)
            with self.assertRaises(ValueError):
                self.demo.start(1, "other")
            with patch("demo_recording.subprocess.run", return_value=probe):
                self.demo.stop()
        self.assertFalse(self.demo.recording)
        metadata = json.loads((self.demo.last / "demo.json").read_text())
        self.assertEqual(metadata["status"], "saved")
        self.assertEqual(metadata["timing"], "wall clock, including pauses")
        self.assertEqual(DemoRecording(self.root).last, self.demo.last)
        process.wait.assert_called_once_with(timeout=8)

    def test_encoder_failure_is_preserved_and_not_a_saved_demo(self):
        directory = self.saved("demo-100", status="recording")
        self.demo.process = Mock(returncode=1, stdin=io.BytesIO())
        self.demo.process.poll.return_value = 1
        self.demo.current, self.demo.log = directory, io.BytesIO()
        with patch("demo_recording.subprocess.run", return_value=Mock(returncode=1)):
            self.demo.poll()
        self.assertFalse(self.demo.recording)
        self.assertIsNone(self.demo.last)
        self.assertTrue((directory / "demo.mp4").is_file())
        self.assertEqual(json.loads((directory / "demo.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
