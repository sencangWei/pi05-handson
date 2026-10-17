import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 在导入仿真库之前设置渲染方式和独立配置目录。
os.environ["LIBERO_CONFIG_PATH"] = str(ROOT / ".local/libero")
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

import yaml

libero_root = ROOT / "openpi/third_party/libero/libero/libero"
dataset_root = ROOT / "data/libero_raw"
dataset_root.mkdir(parents=True, exist_ok=True)

paths = {
    "benchmark_root": libero_root,
    "bddl_files": libero_root / "bddl_files",
    "init_states": libero_root / "init_files",
    "assets": libero_root / "assets",
    "datasets": dataset_root,
}
for name, path in paths.items():
    assert path.is_dir(), f"缺少目录: {name}: {path}"

config_file = Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"
config_file.parent.mkdir(parents=True, exist_ok=True)
config_file.write_text(
    yaml.safe_dump({k: str(v) for k, v in paths.items()})
)

import numpy as np
import torch
from PIL import Image
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

prompt = "pick up the black bowl next to the plate and place it on the plate"
suite = benchmark.get_benchmark_dict()["libero_spatial"]()
matches = [
    (i, suite.get_task(i))
    for i in range(suite.n_tasks)
    if suite.get_task(i).language.strip().casefold() == prompt
]
assert len(matches) == 1, f"任务匹配异常: {matches}"
task_id, task = matches[0]

# 仅加载刚克隆的官方 LIBERO 初始状态文件。
initial_states = torch.load(
    paths["init_states"] / task.problem_folder / task.init_states_file,
    weights_only=False,
)

env = OffScreenRenderEnv(
    bddl_file_name=str(
        paths["bddl_files"] / task.problem_folder / task.bddl_file
    ),
    camera_heights=256,
    camera_widths=256,
)
try:
    env.seed(42)
    env.reset()
    obs = env.set_init_state(initial_states[0])

    # 运行 10 步，让场景稳定；末端位姿增量为零，夹爪指令为 -1。
    for _ in range(10):
        obs, reward, done, info = env.step([0.0] * 6 + [-1.0])

    output = ROOT / "previews"
    output.mkdir(exist_ok=True)
    for key, name in [
        ("agentview_image", "sim-agentview.png"),
        ("robot0_eye_in_hand_image", "sim-wrist.png"),
    ]:
        rgb = np.ascontiguousarray(obs[key][::-1, ::-1])
        Image.fromarray(rgb).save(output / name)
        print("截图:", output / name, "形状:", rgb.shape)

    print("任务编号:", task_id)
    print("文字指令:", task.language)
    print("末端位置:", obs["robot0_eef_pos"])
    print("夹爪位置:", obs["robot0_gripper_qpos"])
    print("仿真画面检查通过")
finally:
    env.close()
