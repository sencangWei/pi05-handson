"""New recovery teachers must not weaken human approval or accept failed rollouts."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_placement_teacher as export
from correction_data import digest, write_json


def frames(count=70):
    state = np.array([0, 0, 1.1, 3.14, 0, 0, .04, -.04], np.float32)
    return [{"state": state.copy(), "post_state": state.copy(),
             "action": np.array([0, 0, 0, 0, 0, 0, -1], np.float32),
             "image": np.zeros((256, 256, 3), np.uint8),
             "wrist_image": np.zeros((256, 256, 3), np.uint8),
             "post_bowl_pos": np.array([0, 0, .9], np.float32),
             "done": np.array(True)} for _ in range(count)]


def test_release_requires_sixty_consecutive_real_open_clear_frames():
    data = frames()
    export.validate_trajectory(data)
    data[-40]["action"][6] = 1
    with pytest.raises(ValueError, match="sixty"):
        export.validate_trajectory(data)


@pytest.mark.parametrize("field,value", [("done", np.array(False)),
                                         ("post_bowl_pos", np.array([0, 0, 1.08], np.float32))])
def test_done_alone_or_unstable_release_is_insufficient(field, value):
    data = frames()
    data[-25][field] = value
    with pytest.raises(ValueError, match="sixty"):
        export.validate_trajectory(data)


def test_discontinuous_data_is_rejected():
    data = frames()
    data[3]["state"][0] += .02
    with pytest.raises(ValueError, match="continuous"):
        export.validate_trajectory(data)


def test_oversized_or_nonfinite_action_is_rejected():
    data = frames()
    data[0]["action"][0] = 1.01
    with pytest.raises(ValueError, match="inside"):
        export.validate_trajectory(data)


def test_failed_rollout_and_fake_human_identity_are_rejected(tmp_path):
    metadata = {"schema_version": 1, "source_kind": "user_run_policy_rollout",
                "human_actions_only": False, "approval_type": None,
                "ready": True, "status": "validated", "split": "train",
                "task": "fixture", "control_dt": .05}
    directory = tmp_path / "teacher"
    directory.mkdir()
    (directory / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="non-human"):
        export.load_teacher(tmp_path, directory, {"task": "fixture"})
    metadata.update(source_kind="offline_placement_recovery_teacher", human_actions_only=True)
    (directory / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="non-human"):
        export.load_teacher(tmp_path, directory, {"task": "fixture"})


def test_bowl_drift_is_checked_even_when_on_predicate_stays_true():
    data = frames()
    data[-1]["post_bowl_pos"][0] = .008
    with pytest.raises(ValueError, match="drift"):
        export.validate_trajectory(data)


def test_real_small_append_export_keeps_public_human_prefix_and_verifies_rgb(tmp_path):
    # Reuse the independently tested miniature public37/approved17/reviewed19 fixture.
    sys.path.insert(0, str(Path(__file__).parent))
    from test_tracking_teacher_export import data as base_fixture
    root, base, _, _ = base_fixture.__wrapped__(tmp_path)
    teacher = root / ".local/placement-fixture"
    teacher.mkdir(parents=True)
    source = root / "evaluations/fixture-scene"
    source.mkdir(parents=True)
    initial = np.array([1., 2., 3.])
    np.save(source / "final_state.npy", initial)
    np.save(teacher / "initial_sim_state.npy", initial)
    scene = {"layout_id": 24, "task_language": base["task"], "task_metadata": {"fixture": "test"}}
    write_json(source / "scene.json", scene)
    hashes = {}
    for name in ("generate_placement_recovery_teacher.py", "generate_tracking_teacher.py", "simulation_lab.py"):
        path = root / "scripts" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("# synthetic fixture, never executed\n")
        hashes[name] = digest(path)
    files = []
    for i, frame in enumerate(frames()):
        path = teacher / f"frame-{i:06d}.npz"
        frame["done"] = np.array(i >= 10)
        frame["image"][0, 0] = [17, 28, 39]
        np.savez_compressed(path, **frame)
        files.append({"path": path.name, "sha256": digest(path)})
    write_json(teacher / "metadata.json", {
        "schema_version": 1, "source_kind": "offline_placement_recovery_teacher",
        "human_actions_only": False, "approval_type": None, "ready": True, "status": "validated",
        "split": "train", "task": base["task"], "control_dt": .05, "scenario_key": "fixture",
        "scenario": scene, "task_metadata": scene["task_metadata"],
        "source_scene": str(source.relative_to(root)),
        "source_files": {"scene.json": digest(source / "scene.json"), "final_state.npy": digest(source / "final_state.npy")},
        "source_state_file": str((source / "final_state.npy").relative_to(root)),
        "source_state_file_sha256": digest(source / "final_state.npy"),
        "initial_sim_state_sha256": export.digest_bytes(initial.tobytes()),
        "source_code_sha256": hashes, "quality_gates": dict.fromkeys(export.REQUIRED_GATES, True),
        "first_ten_step_release_retreat_control_steps": 20,
        "quality_metrics": {"simulation_duration_s": 3.5}, "frame_count": 70, "files": files})
    metadata_path = teacher / "metadata.json"
    saved = json.loads(metadata_path.read_text())
    incomplete = {**saved, "quality_gates": {k: v for k, v in saved["quality_gates"].items()
                                            if k != "no_unexpected_deep_target_contact"}}
    write_json(metadata_path, incomplete)
    with pytest.raises(ValueError, match="quality gates"):
        export.load_teacher(root, teacher, base)
    write_json(metadata_path, {**saved, "source_code_sha256": {k: v for k, v in hashes.items()
                                                             if k != "generate_placement_recovery_teacher.py"}})
    with pytest.raises(ValueError, match="generator source hashes"):
        export.load_teacher(root, teacher, base)
    write_json(metadata_path, saved)
    path = export.prepare(root, "test", [teacher])
    manifest = json.loads(path.read_text())
    export.validate_export(root, manifest, base)
    assert manifest["train_episodes"] == 57
    assert manifest["teacher_train_frames"] == 70
    assert manifest["selected_source_ranges"] == base["selected_source_ranges"]
    assert manifest["holdout_episodes"] == 10
    for i in range(56):
        relative = f"data/chunk-000/episode_{i:06d}.parquet"
        assert digest(root / base["dataset_dir"] / relative) == digest(root / manifest["dataset_dir"] / relative)
    import pyarrow as pa
    import pyarrow.parquet as pq
    parquet = root / manifest["dataset_dir"] / "data/chunk-000/episode_000056.parquet"
    table = pq.read_table(parquet)
    values = table["actions"].to_pylist()
    values[0][0] = .9
    column = table.schema.field("actions")
    table = table.set_column(table.schema.get_field_index("actions"), column, pa.array(values, type=column.type))
    pq.write_table(table, parquet)
    manifest["files"] = [export.file_record(root, parquet) if r["path"] == str(parquet.relative_to(root)) else r
                         for r in manifest["files"]]
    with pytest.raises(ValueError, match="RGB/state/action"):
        export.validate_export(root, manifest, base)
