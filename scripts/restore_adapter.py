"""Rebuild inference params from this course adapter; run with JAX_PLATFORMS=cpu."""
import argparse
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import orbax.checkpoint as ocp
from flax.traverse_util import unflatten_dict

from export_adapter import ASSET, file_hash, frozen_base, identity, load_flat, parameter_manifest, require_equal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("--base-params", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    metadata = json.loads((args.adapter / "metadata.json").read_text())
    if metadata["format"] != "pi05-course-lora-v1":
        raise ValueError("Unsupported adapter format")
    if identity(metadata["original_parameters"]) != metadata["original_tree_sha256"]:
        raise ValueError("Original parameter manifest identity mismatch")
    if identity(metadata["base"]["parameters"]) != metadata["base"]["tree_sha256"]:
        raise ValueError("Base parameter manifest identity mismatch")
    for name in ("adapter.npz", "norm_stats.json"):
        if file_hash(args.adapter / name) != metadata["files"][name]:
            raise ValueError(f"Artifact hash mismatch: {name}")
    base = load_flat(args.base_params)
    require_equal(parameter_manifest(base), metadata["base"]["parameters"], "Original base identity")
    flat = frozen_base(base)
    del base
    expected_adapters = {key: value for key, value in metadata["original_parameters"].items() if "lora" in key}
    with np.load(args.adapter / "adapter.npz", allow_pickle=False) as archive:
        adapters = {key: archive[key] for key in archive.files}
    if not adapters or any(value.dtype != np.float32 for value in adapters.values()):
        raise ValueError("Expected nonempty float32 LoRA arrays")
    require_equal(parameter_manifest(adapters), expected_adapters, "LoRA tensors")
    if set(flat) & set(adapters):
        raise ValueError("Base and adapter tensor keys overlap")
    flat.update(adapters)
    require_equal(parameter_manifest(flat), metadata["original_parameters"], "Reconstructed full parameter tree")
    output.mkdir(parents=True)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(output / "params", {"params": unflatten_dict(flat, sep="/")})
    del flat, adapters
    # Read back OCDBT bytes, rather than treating a successful save call as proof.
    restored = parameter_manifest(load_flat(output / "params"))
    require_equal(restored, metadata["original_parameters"], "Saved-and-restored full parameter tree")
    stats_destination = output / "assets" / ASSET
    stats_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.adapter / "norm_stats.json", stats_destination)
    if file_hash(stats_destination) != metadata["files"]["norm_stats.json"]:
        raise ValueError("Restored normalization hash mismatch")
    report = {"status": "PASS", "inference_only": True, "optimizer_state_restored": False,
              "tensor_count": len(restored), "all_original_tensor_hashes_match": True,
              "original_tree_sha256": metadata["original_tree_sha256"],
              "restored_tree_sha256": identity(restored),
              "adapter_metadata_sha256": file_hash(args.adapter / "metadata.json"),
              "normalization_sha256": file_hash(stats_destination)}
    (output / "adapter_restore_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
