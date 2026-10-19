"""Read real generated episodes through the pinned native training loader, without weights."""
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
os.environ['HF_LEROBOT_HOME'] = str(ROOT / 'data/lerobot')
os.environ['HF_HUB_OFFLINE'] = '1'
# The native loader reads the local cache location during import.
from train_corrections import make_experiment  # noqa: E402
from openpi.training.data_loader import create_torch_dataset, transform_dataset  # noqa: E402

records = []
for rate in [1]:
    manifest_path = ROOT / 'evidence/placement-teacher-negative_quadrants_v1-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    parent = ROOT / 'checkpoints/pi05_handson_lora/retrain_teacher_1x_20261019/1999'
    cfg = make_experiment(manifest['repo_id'], parent, 'negative_quadrants_20261019', 2000, 4, 1e-5)
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    raw = create_torch_dataset(data_config, 10, cfg.model)
    transformed = transform_dataset(raw, data_config)
    offset = manifest['original_train_frames'] + manifest['correction_train_frames']
    samples = []
    for teacher in manifest['teachers']:
        directory = ROOT / teacher['directory']
        count = teacher['frame_count']
        for local in sorted({0, count // 2, count - 10, count - 1}):
            sample = raw[offset + local]
            expected = []
            for j in range(10):
                with np.load(directory / f'frame-{min(local + j, count - 1):06d}.npz', allow_pickle=False) as frame:
                    expected.append(frame['action'])
            assert np.array_equal(np.asarray(sample['actions']), np.asarray(expected))
            with np.load(directory / f'frame-{local:06d}.npz', allow_pickle=False) as frame:
                assert np.array_equal(np.asarray(sample['state']), frame['state'])
                for key in ['image', 'wrist_image']:
                    rgb = np.rint(np.asarray(sample[key]).transpose(1, 2, 0) * 255).astype(np.uint8)
                    assert np.array_equal(rgb, frame[key])
            model = transformed[offset + local]
            assert np.asarray(model['actions']).shape == (10, 32)
            assert np.asarray(model['state']).shape == (32,)
            assert np.asarray(model['tokenized_prompt']).shape == (200,)
            assert np.isfinite(model['actions']).all()
            samples.append({'index': offset + local, 'episode': teacher['episode_index'],
                            'local_frame': local, 'ten_actions_exact': True, 'state_and_both_RGB_exact': True,
                            'model_action_shape': [10, 32], 'used_prompt_tokens': int(np.asarray(model['tokenized_prompt_mask']).sum())})
        offset += count
    assert len(raw) == offset == manifest['train_frames']
    records.append({'source_kind': 'placement_teacher_v1', 'repo_id': manifest['repo_id'], 'frames': len(raw),
                    'samples': samples, 'state_input': cfg.model.discrete_state_input})
    print('LOADER_PASS', rate, len(raw), len(samples), flush=True)
(ROOT / 'evidence/negative-recovery-20261019/loader-check.json').write_text(json.dumps({
    'status': 'PASS', 'method': 'Native pinned OpenPI create_torch_dataset/transform_dataset with real Arrow RGB and frames; no weights/GPU/optimizer.',
    'records': records}, indent=2) + '\n')
