"""Freeze two extra training positions and two held-out positions before training."""
import argparse
from pathlib import Path

import numpy as np

from correction_data import digest, write_json

ROOT = Path(__file__).resolve().parents[1]
CASES = [("train_negpos_near", "train", -.03, .035),
         ("train_negpos_far", "train", -.09, .035),
         ("holdout_negpos_mid", "holdout", -.055, .05),
         ("holdout_negpos_far", "holdout", -.085, .055)]


def main():
    import imageio.v2 as imageio
    from generate_tracking_teacher import camera_images
    from simulation_lab import LiberoTask, TASK_LANGUAGE, move_bowl, state_digest
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    task, records = LiberoTask(), []
    for name, split, dx, dy in CASES:
        directory = output / name
        directory.mkdir()
        env, obs = task.create(24)
        try:
            obs, edit = move_bowl(env, obs, dx, dy)
            state = env.get_sim_state().copy()
            scene = {"layout_id": 24, "seed": 42, "bowl_edit": edit, "task_metadata": task.metadata,
                     "initial_state_sha256": state_digest(state), "task_language": TASK_LANGUAGE,
                     "split": split, "source_kind": "predeclared_simulation_initial_state",
                     "formal_benchmark": False, "setup_control_steps_before_recording": 10,
                     "control_actions_after_initial_state_saved": 0,
                     "source_code_sha256": {"prepare_negative_recovery_scenes.py": digest(Path(__file__)),
                                            "simulation_lab.py": digest(ROOT / "scripts/simulation_lab.py")}}
            np.save(directory / "initial_state.npy", state, allow_pickle=False)
            write_json(directory / "scene.json", scene)
            for key, image in camera_images(obs).items():
                imageio.imwrite(directory / f"initial-{key}.png", image)
            records.append({"case": name, "split": split, "directory": str(directory.relative_to(ROOT)),
                            "scene_sha256": digest(directory / "scene.json"),
                            "state_file_sha256": digest(directory / "initial_state.npy"), "offset": edit})
        finally:
            env.close()
    write_json(output / "manifest.json", {"cases": records, "holdout_teacher_generation_allowed": False})
    print(output / "manifest.json", flush=True)


if __name__ == "__main__":
    main()
