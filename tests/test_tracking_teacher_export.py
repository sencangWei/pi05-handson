"""Isolated append-only export, provenance rejection and native future windows."""
import contextlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))
import prepare_tracking_teacher as export
import train_tracking_teacher as train
from correction_data import CorrectionRecorder, digest, write_json
from prepare_corrections import prepare as prepare_human
from prepare_expanded_public import prepare as expand
from test_expanded_public import fixture as public_fixture
from generate_tracking_teacher import quality_gates, reference_schedule


def metrics():
    return {"terminal_on_last_ten": True, "terminal_min_gap_m": .08,
            "terminal_open_command_last_ten": True, "final_retreat_m": .2,
            "carry_position_error_p95_m": .001, "carry_orientation_error_p95_deg": .5,
            "carry_min_bowl_height_delta_to_reference_m": 0., "carry_relative_bowl_eef_error_max_m": .001,
            "carry_control_step_speedup": 1.6, "continuous_physics_trajectory": True,
            "actions_finite_and_bounded": True, "reference_coverage_complete": True}


@pytest.fixture
def data(tmp_path):
    public_fixture(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        public = expand(tmp_path)
    image = np.zeros((256, 256, 3), np.uint8)
    state = np.array([0, 0, 1.1, 3.14, 0, 0, .04, -.04], np.float32)
    sources, clips = [], []
    for i in range(17):
        source = tmp_path / f"corrections/human_{i}"
        initial = np.array([float(i), 1, 2])
        metadata = {"task": json.loads(public.read_text())["task"], "split": "train",
                    "scenario_key": f"scenario-{i}", "source_scene": "fixture/scene", "scenario": {"layout_id": 24},
                    "control_dt": .05, "initial_sim_state_sha256": export.digest_bytes(initial.tobytes())}
        recorder = CorrectionRecorder(source, metadata)
        np.save(source / "initial_sim_state.npy", initial)
        count = 100 if i < 2 else 10
        for _ in range(count):
            recorder.append(state, [-.1 if i == 1 else .1, 0, 0, 0, 0, 0, -1], {"image": image, "wrist_image": image},
                            state, [0, 0, .9], True)
        recorder.finish(True)
        sources.append(source)
        for start in (range(0, 30, 10) if i == 0 else [0]):
            clips.append({"source_directory": str(source.relative_to(tmp_path)),
                          "metadata_sha256": digest(source / "metadata.json"), "source_frame_count": count,
                          "start": start, "stop": start + 10, "reason": "fixture reviewed release", "kind": "placement_release"})
    selection = tmp_path / "evidence/fixture-selection.json"
    from prepare_corrections import SELECTION_RULE_V2
    write_json(selection, {"schema_version": 2, "task": metadata["task"], "rule": SELECTION_RULE_V2, "clips": clips})
    base_path = prepare_human(tmp_path, "public37_reviewed19_v4", public, selection)
    base = json.loads(base_path.read_text())
    teacher_dirs = []
    for i, source in enumerate(sources[:2]):
        directory = tmp_path / f".local/teachers/{i}"
        directory.mkdir(parents=True)
        original = json.loads((source / "metadata.json").read_text())
        reference = [index for index, _ in reference_schedule(original["frame_count"], (10, 90))]
        count, files = len(reference), []
        initial = np.load(source / "initial_sim_state.npy", allow_pickle=False)
        np.save(directory / "initial_sim_state.npy", initial)
        np.save(directory / "reference-index.npy", np.asarray(reference))
        for j in range(count):
            path = directory / f"frame-{j:06d}.npz"
            x_action = .2 + .1*i if 10 <= reference[j] < 90 else (-.1 if i == 1 else .1)
            np.savez_compressed(path, state=state, action=np.array([x_action, 0, 0, 0, 0, 0, -1], np.float32),
                                image=image + i, wrist_image=image + i, post_state=state,
                                post_bowl_pos=np.array([0, 0, .9], np.float32), done=np.asarray(True))
            files.append({"path": path.name, "sha256": digest(path)})
        write_json(directory / "metadata.json", {
            "schema_version": 1, "source_kind": "offline_tracking_teacher", "human_actions_only": False,
            "generator_version": 2, "feedback_scope": "audited_carry_only", "noncarry_action_policy": "exact_source_action",
            "status": "validated", "ready": True, "split": "train", "task": base["task"], "control_dt": .05,
            "scenario_key": original["scenario_key"], "scenario": original["scenario"],
            "source_directory": str(source), "source_metadata_sha256": digest(source / "metadata.json"),
            "source_initial_sim_state_sha256": original["initial_sim_state_sha256"],
            "initial_sim_state_sha256": export.digest_bytes(initial.tobytes()),
            "source_frame_count": original["frame_count"], "frame_count": count, "files": files,
            "reference_carry_interval": [10, 90], "protected_source_steps": 10,
            "reference_rate": 2, "requested_reference_rate": 2, "role": "carry_rate2_treatment",
            "quality_gates": quality_gates(metrics(), 2), "quality_metrics": metrics(),
            "source_code_sha256": {"fixture": "a" * 64}})
        teacher_dirs.append(directory)
    parent = tmp_path / "checkpoints/pi05_handson_lora/parent/1999"
    (parent / "params").mkdir(parents=True)
    (parent / "params/manifest.ocdbt").write_text("fixture only; never loaded")
    norm = parent / "assets/course/libero_black_bowl/norm_stats.json"
    norm.parent.mkdir(parents=True)
    norm.write_bytes((tmp_path / base["normalization"]["path"]).read_bytes())
    return tmp_path, base, teacher_dirs, parent


def test_real_small_export_preserves_all_base_and_holdout_files(data):
    root, base, teachers, parent = data
    originals = {p: digest(p) for p in root.rglob("*") if p.is_file()}
    path = export.prepare(root, "pilot", teachers)
    manifest, _, _ = train.validate_inputs(path, parent, "new", root=root)
    assert manifest["train_episodes"] == 58 and manifest["teacher_train_frames"] == 140
    assert manifest["correction_train_episodes"] == 19
    assert len(set(manifest["correction_source_directories"])) == 17
    assert manifest["holdout_episodes"] == 10
    assert all(digest(p) == sha for p, sha in originals.items())
    for i in range(56):
        relative = f"data/chunk-000/episode_{i:06d}.parquet"
        assert digest(root / manifest["dataset_dir"] / relative) == digest(root / base["dataset_dir"] / relative)
    with pytest.raises(FileExistsError):
        export.prepare(root, "pilot", teachers)


@pytest.mark.parametrize("mutation", [{"ready": False}, {"human_actions_only": True},
                                     {"status": "approved"}, {"quality_gates": {"geometry": False}}])
def test_unvalidated_or_fake_human_teacher_is_rejected_before_export(data, mutation):
    root, _, teachers, _ = data
    path = teachers[0] / "metadata.json"
    metadata = json.loads(path.read_text())
    write_json(path, {**metadata, **mutation})
    with pytest.raises(ValueError, match="ready, validated, non-human"):
        export.prepare(root, "bad", teachers)
    assert not (root / "data/lerobot/course/libero_tracking_teacher_bad").exists()


def test_frame_sha_tamper_and_rehashed_wrong_payload_are_rejected(data):
    root, _, teachers, parent = data
    path = export.prepare(root, "pilot", teachers)
    frame = teachers[0] / "frame-000000.npz"
    frame.write_bytes(frame.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="Artifact differs|checksum"):
        train.validate_inputs(path, parent, "new", root=root)


def test_rehashed_wrong_export_action_still_rejected_against_actual_raw_frames(data):
    import pyarrow as pa
    root, _, teachers, parent = data
    path = export.prepare(root, "pilot", teachers)
    manifest = json.loads(path.read_text())
    parquet = root / manifest["dataset_dir"] / "data/chunk-000/episode_000056.parquet"
    table = pq.read_table(parquet)
    values = table["actions"].to_pylist()
    values[0][0] = .9
    position = table.schema.get_field_index("actions")
    table = table.set_column(position, table.schema.field(position), pa.array(values, type=table.schema.field(position).type))
    pq.write_table(table, parquet)
    manifest["files"] = [export.file_record(root, parquet) if item["path"] == str(parquet.relative_to(root)) else item
                         for item in manifest["files"]]
    write_json(path, manifest)
    with pytest.raises(ValueError, match="schema/state/action/RGB"):
        train.validate_inputs(path, parent, "new", root=root)


def test_native_lerobot_ten_action_windows_stay_in_teacher_episode(data, monkeypatch):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets import lerobot_dataset
    root, _, teachers, _ = data
    path = export.prepare(root, "pilot", teachers)
    manifest = json.loads(path.read_text())
    monkeypatch.setattr(lerobot_dataset, "HF_LEROBOT_HOME", root / "data/lerobot")
    dataset = LeRobotDataset(manifest["repo_id"], root=root / manifest["dataset_dir"],
                             delta_timestamps={"actions": [i / 10 for i in range(10)]})
    # The public fixture intentionally has fake HF schema metadata; supply its
    # image feature types, leaving native episode/window indexing untouched.
    from datasets import Image
    for key in ("image", "wrist_image"):
        dataset.hf_dataset = dataset.hf_dataset.cast_column(key, Image())
    episodes = [json.loads(line) for line in (root / manifest["dataset_dir"] / "meta/episodes.jsonl").read_text().splitlines()]
    first = sum(e["length"] for e in episodes[:56])
    for index, expected in [(first + 68, .1), (first + 69, .1), (first + 70, -.1)]:
        window = dataset[index]["actions"].numpy()
        assert window.shape == (10, 7)
        np.testing.assert_allclose(window[:, 0], expected)


def test_trainer_validates_original_human_base_before_generated_manifest(data):
    root, _, _, parent = data
    missing = root / "evidence/nonexistent.json"
    with patch.object(train.train_corrections, "validate_inputs", side_effect=ValueError("base validation first")) as check:
        with pytest.raises(ValueError, match="base validation first"):
            train.validate_inputs(missing, parent, "new", root=root)
        check.assert_called_once_with(root / export.BASE_MANIFEST, parent, "new", root=root)


def test_dry_run_never_imports_trainer_or_writes_launch(data):
    root, _, teachers, parent = data
    path = export.prepare(root, "pilot", teachers)
    cfg = SimpleNamespace(model=SimpleNamespace(discrete_state_input=False, action_horizon=10), seed=42)
    with (patch.object(train, "ROOT", root), patch.object(train.train_corrections, "make_experiment", return_value=cfg) as make,
          patch.object(train.train_corrections, "build_record", return_value={"source_sha256": {}}),
          patch.object(train, "digest", return_value="fixture"),
          patch.object(train.importlib.util, "spec_from_file_location") as load_trainer,
          contextlib.redirect_stdout(io.StringIO()) as output):
        train.main(["--manifest", str(path), "--parent-checkpoint", str(parent), "--experiment", "new", "--dry-run"])
        assert make.call_args.args[-3:] == (2000, 4, 1e-5)
        load_trainer.assert_not_called()
    assert "DRY_RUN_PASS" in output.getvalue()
    assert not (root / "evidence/new-launch.json").exists()


@pytest.mark.parametrize("mutation", ["missing_gate", "contradicting_metrics", "mixed_rate"])
def test_complete_quality_schema_metrics_and_same_rate_required(data, mutation):
    root, _, teachers, _ = data
    path = teachers[0] / "metadata.json"
    metadata = json.loads(path.read_text())
    if mutation == "missing_gate":
        metadata["quality_gates"].pop("carry_not_dropped")
    elif mutation == "contradicting_metrics":
        metadata["quality_metrics"]["carry_relative_bowl_eef_error_max_m"] = .1
    else:
        set_rate_one(teachers[0])
        metadata = json.loads(path.read_text())
    write_json(path, metadata)
    with pytest.raises(ValueError, match="quality|same reference rate"):
        export.prepare(root, "bad", teachers)


def test_rate_one_pair_has_no_faster_than_original_requirement(data):
    root, _, teachers, parent = data
    for directory in teachers:
        set_rate_one(directory)
    path = export.prepare(root, "control", teachers)
    manifest, _, _ = train.validate_inputs(path, parent, "new", root=root)
    assert manifest["teacher_train_episodes"] == 2


def set_rate_one(directory):
    """Create a separate fixture observation/action sequence, not a production retimer."""
    path = directory / "metadata.json"
    metadata = json.loads(path.read_text())
    original_count = metadata["source_frame_count"]
    payload = (directory / "frame-000000.npz").read_bytes()
    for i in range(metadata["frame_count"], original_count):
        (directory / f"frame-{i:06d}.npz").write_bytes(payload)
    metadata.update(frame_count=original_count, reference_rate=1, requested_reference_rate=1, role="carry_rate1_control")
    metadata["files"] = [{"path": f"frame-{i:06d}.npz", "sha256": digest(directory / f"frame-{i:06d}.npz")}
                         for i in range(original_count)]
    metadata["quality_metrics"]["carry_control_step_speedup"] = 1.
    metadata["quality_gates"] = quality_gates(metadata["quality_metrics"], 1)
    np.save(directory / "reference-index.npy", np.arange(original_count))
    write_json(path, metadata)


def test_sampling_alias_uses_the_fully_validated_plain_teacher_export(data):
    from prepare_teacher_sampling import prepare as prepare_profile
    from recovery_sampling import prepare_sampling
    root, _, teachers, parent = data
    plain = export.prepare(root, "pilot", teachers)
    alias, record = prepare_profile(root, plain, "pilot_focus")
    manifest, _, _ = train.validate_inputs(alias, parent, "new", root=root)
    indices, again = prepare_sampling(root, manifest)
    assert again == record and len(indices) == 100000
    assert set(indices) == set(range(manifest["train_frames"]))


def test_trainer_failure_restores_the_process_scoped_sampling_override(tmp_path):
    import recovery_sampling
    package, training, loader = ModuleType("openpi"), ModuleType("openpi.training"), ModuleType("openpi.training.data_loader")
    package.__path__, training.__path__ = [], []
    def original(*args):
        return None
    loader.create_torch_dataset = original
    def install(module, *args):
        module.create_torch_dataset = lambda *unused: "temporary override"
        return original
    def fail(cfg):
        assert loader.create_torch_dataset is not original
        raise RuntimeError("fixture trainer failure")
    cfg = SimpleNamespace(model=SimpleNamespace(discrete_state_input=False, action_horizon=10), seed=42)
    manifest = {"repo_id": "fixture", "base_manifest": {}, "teachers": []}
    spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None))
    launch = tmp_path / "launch.json"
    with (patch.object(train, "ROOT", tmp_path),
          patch.object(train, "validate_inputs", return_value=(manifest, tmp_path, launch)),
          patch.object(train.train_corrections, "make_experiment", return_value=cfg),
          patch.object(train.train_corrections, "build_record", return_value={"source_sha256": {}}),
          patch.object(train, "digest", return_value="fixture"),
          patch.object(recovery_sampling, "prepare_sampling", return_value=([0], {"seed": 42})),
          patch.object(recovery_sampling, "install_sampling", side_effect=install),
          patch.dict(sys.modules, {"openpi": package, "openpi.training": training, "openpi.training.data_loader": loader}),
          patch.object(train.importlib.util, "spec_from_file_location", return_value=spec),
          patch.object(train.importlib.util, "module_from_spec", return_value=SimpleNamespace(main=fail)),
          contextlib.redirect_stdout(io.StringIO())):
        with pytest.raises(RuntimeError, match="fixture trainer failure"):
            train.main(["--manifest", str(tmp_path), "--parent-checkpoint", str(tmp_path), "--experiment", "fixture"])
    assert loader.create_torch_dataset is original


@pytest.mark.parametrize("mutation", [{"generator_version": 1}, {"feedback_scope": "whole_episode"},
                                     {"noncarry_action_policy": "feedback"}, {"feedback_scope": None}])
def test_legacy_full_feedback_or_missing_v2_contract_is_rejected(data, mutation):
    root, _, teachers, _ = data
    path = teachers[0] / "metadata.json"
    metadata = json.loads(path.read_text())
    write_json(path, {**metadata, **mutation})
    with pytest.raises(ValueError, match="generator v2"):
        export.load_teachers(root, teachers, json.loads((root / export.BASE_MANIFEST).read_text()))


def test_rehashed_noncarry_action_change_cannot_pass_ready_metadata(data):
    root, base, teachers, _ = data
    path = teachers[0] / "frame-000000.npz"
    with np.load(path, allow_pickle=False) as archive:
        frame = {key: archive[key] for key in archive.files}
    frame["action"][0] = .4
    np.savez_compressed(path, **frame)
    metadata_path = teachers[0] / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["files"][0]["sha256"] = digest(path)
    write_json(metadata_path, metadata)
    with pytest.raises(ValueError, match="noncarry action differs"):
        export.load_teachers(root, teachers, base)


def test_self_consistent_initial_state_must_still_match_original_source(data):
    root, base, teachers, _ = data
    path = teachers[0] / "initial_sim_state.npy"
    state = np.load(path, allow_pickle=False)
    state[0] += .01
    np.save(path, state)
    metadata_path = teachers[0] / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["initial_sim_state_sha256"] = export.digest_bytes(state.tobytes())
    write_json(metadata_path, metadata)
    with pytest.raises(ValueError, match="source identity"):
        export.load_teachers(root, teachers, base)


def test_missing_feedback_scope_is_rejected_before_reading_frames(tmp_path):
    directory = tmp_path / "teacher"
    directory.mkdir()
    write_json(directory / "metadata.json", {"generator_version": 2, "noncarry_action_policy": "exact_source_action"})
    with pytest.raises(ValueError, match="generator v2"):
        export.load_teachers(tmp_path, [directory, directory], {"selected_source_ranges": []})
