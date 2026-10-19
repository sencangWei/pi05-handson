"""Check this immutable pilot's real training reader without loading model weights."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ.update(JAX_PLATFORMS="cpu", HF_HUB_OFFLINE="1",
                  HF_LEROBOT_HOME=str(ROOT / "data/lerobot"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from correction_data import digest, load_frames, write_json  # noqa: E402
from train_corrections import make_experiment  # noqa: E402
from openpi.training.data_loader import create_torch_dataset  # noqa: E402


def main():
    path = ROOT / "evidence/corrections-public37_manual_tempo_pilot_v1-manifest.json"
    manifest = json.loads(path.read_text())
    parent = ROOT / "checkpoints/pi05_handson_lora/my_first_200/9999"
    cfg = make_experiment(manifest["repo_id"], parent, "manual_tempo_pilot_ready", 2000, 4, 1e-5)
    data = create_torch_dataset(cfg.data.create(cfg.assets_dirs, cfg.model), 10, cfg.model)
    assert len(data) == manifest["train_frames"] == 5446
    directory = ROOT / manifest["dataset_dir"]
    episodes = [json.loads(line) for line in (directory / "meta/episodes.jsonl").read_text().splitlines()]
    assert len(episodes) == 43
    offset, windows, human_frames = 0, 0, 0
    for episode, record in enumerate(episodes):
        table = pq.read_table(directory / f"data/chunk-000/episode_{episode:06d}.parquet")
        actions = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
        states = np.asarray(table["state"].to_pylist(), dtype=np.float32)
        count = record["length"]
        assert len(actions) == count
        for start in sorted({0, max(0, count - 11), count - 1}):
            sample = data[offset + start]
            expected = actions[np.minimum(np.arange(start, start + 10), count - 1)]
            np.testing.assert_array_equal(np.asarray(sample["actions"]), expected)
            np.testing.assert_array_equal(np.asarray(sample["state"]), states[start])
            assert sample["prompt"] == manifest["task"]
            assert np.asarray(sample["actions"]).shape == (10, 7)
            assert np.asarray(sample["state"]).shape == (8,)
            windows += 1
        if episode >= 37:
            clip = manifest["selected_source_ranges"][episode - 37]
            source = ROOT / clip["source_directory"]
            metadata = json.loads((source / "metadata.json").read_text())
            assert digest(source / "metadata.json") == clip["metadata_sha256"]
            frames = load_frames(source, metadata)[clip["start"]:clip["stop"]]
            np.testing.assert_array_equal(actions, np.stack([f["action"] for f in frames]))
            np.testing.assert_array_equal(states, np.stack([f["state"] for f in frames]))
            human_frames += len(frames)
        offset += count
    assert human_frames == 1046
    report = {"status": "PASS", "manifest_sha256": digest(path),
              "loader": "openpi.training.data_loader.create_torch_dataset",
              "train_episodes": 43, "train_frames": 5446, "human_frames_exact": human_frames,
              "episode_start_tail_end_windows_checked": windows, "action_shape": [10, 7],
              "state_shape": [8], "training_started": False, "model_weights_loaded": False,
              "limits": "Reader, labels and configuration checks only; no GPU training or rollout acceptance."}
    write_json(Path(__file__).parent / "loader-check.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
