"""Protect native action windows when contrasting original and faster trajectories."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_teacher_sampling import phase_groups


def fixture(rate):
    references = []
    for _ in range(2):
        ref = np.array(list(range(20)) + list(range(20, 60, rate)) + list(range(60, 80)))
        references.append(({"frame_count": len(ref), "reference_carry_interval": [10, 70],
                            "protected_source_steps": 10}, ref))
    manifest = {"original_train_frames": 20, "correction_train_frames": 30,
                "teachers": [{"frame_count": len(r)} for _, r in references],
                "train_frames": 50 + sum(len(r) for _, r in references)}
    return manifest, references


@pytest.mark.parametrize("rate", [1, 2])
def test_every_frame_once_in_group_and_all_carry_windows_protected(rate):
    manifest, references = fixture(rate)
    groups = phase_groups(manifest, references)
    assigned = []
    for group in groups:
        assigned.extend(i for a, b in group["ranges"] for i in range(a, b))
    assert sorted(assigned) == list(range(manifest["train_frames"]))
    assert [g["virtual_count"] for g in groups] == [60000, 20000, 15000, 5000]
    offset = 50
    for _, ref in references:
        for a, b in groups[2]["ranges"]:
            for i in range(max(a, offset), min(b, offset + len(ref))):
                assert len(ref[i - offset:i - offset + 10]) == 10
                assert np.all((ref[i - offset:i - offset + 10] >= 20)
                              & (ref[i - offset:i - offset + 10] < 60))
        offset += len(ref)


def test_shorter_teacher_keeps_same_phase_mass_without_stitching():
    slow, slow_ref = fixture(1)
    fast, fast_ref = fixture(2)
    slow_group = phase_groups(slow, slow_ref)[2]
    fast_group = phase_groups(fast, fast_ref)[2]
    assert slow_group["virtual_count"] == fast_group["virtual_count"]
    assert sum(b - a for a, b in fast_group["ranges"]) < sum(b - a for a, b in slow_group["ranges"])


def test_bad_reference_contract_rejected():
    manifest, refs = fixture(2)
    refs[0][0]["protected_source_steps"] = 0
    with pytest.raises(ValueError, match="protected"):
        phase_groups(manifest, refs)
