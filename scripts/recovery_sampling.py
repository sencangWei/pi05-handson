"""Pinned recovery observation sampling; future action windows remain unchanged."""
import hashlib
import json
import random

from prepare_corrections import contained
from train_release_focus import RepeatedFrames


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_sampling(root, manifest):
    """Return a virtual start-index map only for an explicitly pinned data export."""
    reference = manifest.get("training_sampling_profile")
    if reference is None:
        return None
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("Invalid recovery sampling profile reference")
    path = contained(root, reference["path"])
    if path.parent != root / "evidence" or digest(path) != reference["sha256"]:
        raise ValueError("Recovery sampling profile checksum or location differs")
    profile = json.loads(path.read_text())
    if (profile.get("schema_version") != 1 or type(profile.get("seed")) is not int
            or profile["seed"] != manifest["seed"] or profile.get("repo_id") != manifest["repo_id"]):
        raise ValueError("Recovery sampling schema, seed or dataset differs")
    base_path = contained(root, profile["dataset_manifest"])
    if base_path.parent != root / "evidence" or digest(base_path) != profile["dataset_manifest_sha256"]:
        raise ValueError("Recovery sampling base manifest checksum differs")
    base = json.loads(base_path.read_text())
    if base != {key: value for key, value in manifest.items() if key != "training_sampling_profile"}:
        raise ValueError("Focused data must equal the pinned export except for sampling metadata")
    groups = profile.get("groups")
    total = profile.get("virtual_frames")
    if (not isinstance(groups, list) or not groups or type(total) is not int or total < 1):
        raise ValueError("Invalid recovery sampling groups or virtual count")
    assigned, indices, summaries, names = set(), [], [], set()
    rng = random.Random(profile["seed"])
    for group in groups:
        if (not isinstance(group, dict) or set(group) != {"name", "ranges", "virtual_count"}
                or not isinstance(group["name"], str) or not group["name"] or group["name"] in names
                or not isinstance(group["ranges"], list) or not group["ranges"]):
            raise ValueError("Invalid recovery sampling group")
        names.add(group["name"])
        starts = []
        for bounds in group["ranges"]:
            if (not isinstance(bounds, list) or len(bounds) != 2
                    or any(type(value) is not int for value in bounds)
                    or not 0 <= bounds[0] < bounds[1] <= manifest["train_frames"]):
                raise ValueError("Recovery sampling range outside training data")
            current = set(range(*bounds))
            if assigned & current:
                raise ValueError("Recovery sampling groups overlap")
            assigned.update(current)
            starts.extend(range(*bounds))
        count = group["virtual_count"]
        if type(count) is not int or count < len(starts):
            raise ValueError("Recovery sampling must retain every real observation start")
        indices.extend(starts)
        indices.extend(rng.choices(starts, k=count - len(starts)))
        summaries.append({**group, "real_frames": len(starts), "probability": count / total})
    if len(indices) != total or assigned != set(range(manifest["train_frames"])):
        raise ValueError("Recovery sampling must cover exactly all training frames")
    return indices, {
        "profile": {**reference}, "base_manifest": profile["dataset_manifest"],
        "base_manifest_sha256": profile["dataset_manifest_sha256"],
        "repo_id": manifest["repo_id"], "real_frames": manifest["train_frames"],
        "virtual_frames": total, "seed": profile["seed"], "groups": summaries,
        "sampling_indices_sha256": hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
        "meaning": "Repeated observation starts, shuffled by the existing loader; no extra scenarios, no action stitching",
    }


def install_sampling(data_loader, indices, record):
    """Scope the remap to one trainer process and one pinned repository/horizon."""
    original = data_loader.create_torch_dataset

    def create(data_config, action_horizon, model_config):
        if data_config.repo_id != record["repo_id"] or action_horizon != 10:
            raise ValueError("Recovery sampling map used with a different dataset or action horizon")
        dataset = original(data_config, action_horizon, model_config)
        if len(dataset) != record["real_frames"]:
            raise ValueError("Recovery sampling real dataset length differs")
        return RepeatedFrames(dataset, indices)

    data_loader.create_torch_dataset = create
    return original
