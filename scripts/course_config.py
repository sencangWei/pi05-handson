"""A small, explicit pi0.5 LoRA lesson using only a public LIBERO subset."""
import os
from pathlib import Path

from flax import nnx
from openpi.models.pi0_config import Pi0Config
from openpi.shared.nnx_utils import PathRegex
from openpi.training import config, optimizer, weight_loaders

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "course/libero_black_bowl"
NAME = "pi05_handson_lora"
TASK = "pick up the black bowl next to the plate and place it on the plate"
HORIZON = 10


def make_config(steps=200, experiment="lesson01_200", *, resume=False):
    model = Pi0Config(
        pi05=True, paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora", action_horizon=HORIZON,
        discrete_state_input=False,
    )
    params = os.environ.get(
        "PI05_BASE_PARAMS",
        "/home/robot/pi0.5/versions/001_official_openpi_pi05_base_jax/params",
    )
    # Deliberately stricter than the upstream default: only LoRA is trainable.
    # Freezing the image encoder saves memory while other GPU tasks stay alive.
    frozen = nnx.All(nnx.Param, nnx.Not(PathRegex(".*lora.*")))
    return config.TrainConfig(
        name=NAME, project_name="pi05-simulation-course", exp_name=experiment,
        model=model, freeze_filter=frozen, ema_decay=None,
        data=config.LeRobotLiberoDataConfig(
            repo_id=REPO_ID, base_config=config.DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(params),
        batch_size=1, num_workers=0, seed=42, num_train_steps=steps,
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=20, peak_lr=5e-5, decay_steps=max(steps, 21), decay_lr=5e-5,
        ),
        optimizer=optimizer.AdamW(clip_gradient_norm=1.0),
        assets_base_dir=str(ROOT / "assets"),
        checkpoint_base_dir=str(ROOT / "checkpoints"),
        save_interval=steps, keep_period=None, log_interval=10,
        wandb_enabled=False, resume=resume, overwrite=False,
        policy_metadata={"course": "pi05 LoRA", "task": TASK,
                         "action_horizon": HORIZON, "seed": 42},
    )

