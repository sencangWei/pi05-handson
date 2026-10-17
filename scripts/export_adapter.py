"""Export this course's LoRA and evidence; run with JAX_PLATFORMS=cpu.

This is an inference backup, not an Adam/DataLoader training-state backup.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import ml_dtypes
import numpy as np
from flax.traverse_util import flatten_dict
from openpi.models.model import restore_params

ASSET = Path("course/libero_black_bowl/norm_stats.json")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parameter_manifest(flat):
    result = {}
    for key, value in sorted(flat.items()):
        array = np.asarray(value)
        result[key] = {"shape": list(array.shape), "dtype": str(array.dtype),
                       "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}
    return result


def identity(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def load_flat(path):
    return flatten_dict(restore_params(path, restore_type=np.ndarray), sep="/")


def require_equal(actual, expected, description):
    if actual != expected:
        differing = sorted(key for key in set(actual) | set(expected)
                           if actual.get(key) != expected.get(key))
        raise ValueError(f"{description} mismatch: {differing[:10]}")


def frozen_base(base):
    # Upstream train.init_train_state casts every frozen parameter to bfloat16.
    if any("lora" in key for key in base):
        raise ValueError("Expected original non-LoRA base parameters")
    return {key: value.astype(ml_dtypes.bfloat16) for key, value in base.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Full checkpoint root, e.g. .../199")
    parser.add_argument("--base-params", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    original = load_flat(args.checkpoint / "params")
    original_manifest = parameter_manifest(original)
    adapters = {key: value for key, value in original.items() if "lora" in key}
    if not adapters or any(value.dtype != np.float32 for value in adapters.values()):
        raise ValueError("This course exporter requires nonempty float32 LoRA parameters")
    frozen_manifest = {key: value for key, value in original_manifest.items() if "lora" not in key}
    # Hold only the small LoRA arrays while loading the full original base on CPU.
    del original
    base = load_flat(args.base_params)
    base_manifest = parameter_manifest(base)
    require_equal(parameter_manifest(frozen_base(base)), frozen_manifest, "Frozen parameters vs base cast")
    del base
    norm_path = args.checkpoint / "assets" / ASSET
    if not norm_path.is_file():
        raise FileNotFoundError(norm_path)
    output.mkdir(parents=True)
    np.savez_compressed(output / "adapter.npz", **adapters)
    shutil.copyfile(norm_path, output / "norm_stats.json")
    metadata = {
        "format": "pi05-course-lora-v1", "purpose": "inference only; no optimizer or loader state",
        "checkpoint_step_index": args.checkpoint.name,
        "base": {"reference": "official_openpi_pi05_base_jax", "tree_sha256": identity(base_manifest),
                 "parameters": base_manifest, "manifest_sha256": file_hash(args.base_params / "manifest.ocdbt")},
        "model": {"pi05": True, "paligemma_variant": "gemma_2b_lora",
                  "action_expert_variant": "gemma_300m_lora", "action_horizon": 10,
                  "discrete_state_input": False, "freeze": "all non-LoRA parameters, cast bfloat16"},
        "original_parameters": original_manifest, "original_tree_sha256": identity(original_manifest),
        "files": {name: file_hash(output / name) for name in ("adapter.npz", "norm_stats.json")},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(output), "lora_tensors": len(adapters),
                      "frozen_base_exact_match": True, "original_tree_sha256": identity(original_manifest)}))


if __name__ == "__main__":
    main()
