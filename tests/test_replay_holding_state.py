"""A saved holding state must not silently inherit the reset gripper accumulator."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import replay_saved_scene as replay


def test_holding_without_canonical_initialization_fails_before_simulation_or_output(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    (source / "scene.json").write_text(json.dumps({"task_language": replay.TASK_LANGUAGE,
                                                "source_kind": "exact_saved_failure_holding_state"}))
    monkeypatch.setattr(sys, "argv", ["replay", "--source-scene", str(source), "--output", str(output), "--smoke"])
    monkeypatch.setattr(replay, "LiberoTask", lambda: pytest.fail("must reject before simulator construction"))
    with pytest.raises(ValueError, match="require --initialize-saved-controller"):
        replay.main()
    assert not output.exists()
