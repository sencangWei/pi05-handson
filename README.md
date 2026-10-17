# π0.5 LoRA：从示范数据到 LIBERO 仿真执行

本项目记录一次亲手完成的机器人模仿学习实践：公开示范检查、数据准备、归一化、LoRA 微调、闭环仿真、失败分析和模型备份。使用预训练 π0.5 基座进行单任务微调。

**任务：** `pick up the black bowl next to the plate and place it on the plate`。

## 实测结果

最终 checkpoint 为 `9999`，即完成 10000 次参数更新。固定预测动作窗口为 10，最终评测每执行 1 步就重新观察。每次测试最多执行 220 步，采用 LIBERO 自带任务成功判定。

| 模型 | 执行后重新观察的间隔 | 初始状态 | 成功次数 |
| --- | --- | --- | --- |
| 200 步 | 5 步 | 0 | 0/1 |
| 2000 步 | 5 步 | 0 | 0/1 |
| 2000 步 | 1 步 | 0、1、2、3 | 1/4 |
| 10000 步 | 1 步 | 0、1、2、3 | 4/4 |
| 10000 步 | 1 步 | 10、11、12、13 | 4/4 |

最后一组布局在冻结最终模型和执行参数后测试，未用于前面的调参。最终模型在这 8 个初始状态上完成 8 次任务，没有程序异常。这是同一仿真任务的小规模观察；尚未验证掉落恢复、跨任务泛化或真机迁移。策略采样 RNG 在服务启动时初始化为 0，随后连续推进；每次 trial 不会单独重置采样 RNG。早期与后期模型的轨迹长度不同，后续 trial 的采样序列也会不同。

演示：[初始状态 0](evaluations/my_first_10000_replan1_init0to3/trial-000_init-000_success.mp4)、[初始状态 12](evaluations/my_first_10000_replan1_init10to13/trial-002_init-012_success.mp4)。失败视频也保留在 `evaluations/`，用于复盘。

视频为每个受控仿真步骤一帧、10 FPS 回放，省略推理等待时间，不代表实时运行速度。实际耗时和每次推理延迟见 JSON 与 CSV。

## 训练配置与数据

- OpenPI：`15a9616a00943ada6c20a0f158e3adb39df2ccac`。
- LIBERO：`f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`。
- 公开数据：[physical-intelligence/libero](https://huggingface.co/datasets/physical-intelligence/libero)，revision `a4336d589d589045d1c56423ffdf3b88a0e19b1f`。
- 训练集 16 条示范、1971 帧；留出集 4 条示范、473 帧。划分和文件哈希见 `evidence/data_manifest.json`。留出数据未用于训练和归一化；本轮尚未报告离线留出误差。
- Gemma 2B LoRA + Gemma 300M action expert LoRA；仅更新 LoRA 参数，其他参数冻结。
- 前 2000 次更新 batch=1；后 8000 次更新 batch=4，累计处理约 34000 个动作窗口，包含重复样本。
- seed=42，AdamW，梯度裁剪 1.0，20 步 warmup，之后学习率 5e-5，EMA 关闭。
- 恢复训练保留模型和优化器状态，但上游不会恢复 DataLoader 游标；本实验不声称逐位一致的断点续跑。
- 训练预算和 batch 同时增加，表现改善不能单独归因于其中一项。
- 8 维状态为末端 xyz、轴角 xyz、两个夹指位置；7 维动作为末端平移/旋转增量控制和夹爪命令，不能当作 7 个关节角度。
- 当前模型 `discrete_state_input=False`，使用 RGB 和语言条件，不输入本体状态 token。

## 环境与源码

Ubuntu 22.04、Python 3.11.15、JAX 0.5.3、RTX 5090 D 32 GB。MuJoCo 物理在 CPU 运行，EGL 用于渲染。训练与仿真使用独立环境；精确版本快照见 `evidence/`，训练依赖锁位于 OpenPI 子模块的 `uv.lock`。

```bash
git clone https://github.com/sencangWei/pi05-handson.git
cd pi05-handson
git submodule update --init openpi
git -C openpi submodule update --init third_party/libero

uv python install 3.11.15
uv venv --python 3.11.15 .venv
source .venv/bin/activate
cd openpi
UV_HTTP_TIMEOUT=300 GIT_LFS_SKIP_SMUDGE=1 uv sync --active --frozen
cd ..

uv venv --python 3.11.15 .venv-sim
CMAKE_POLICY_VERSION_MINIMUM=3.5 UV_HTTP_TIMEOUT=300 uv pip install \
  --python .venv-sim/bin/python -r scripts/requirements-sim.txt \
  -e openpi/packages/openpi-client -e openpi/third_party/libero
```

LIBERO 的旧打包配置在本环境产生空的 editable 映射，运行仿真时需要把其源码加入模块搜索路径：

```bash
export PYTHONPATH="$PWD/openpi/third_party/libero${PYTHONPATH:+:$PYTHONPATH}"
.venv-sim/bin/python scripts/sim_smoke.py
```

`sim_smoke.py` 会创建当前目录对应的 `.local/libero/config.yaml`；新机器不用沿用旧机器的绝对路径。已有截图和视频可以直接查看，无需重新训练。

## 模型包与恢复

最终 LoRA、归一化统计和逐张量参数哈希通过 [GitHub Release](https://github.com/sencangWei/pi05-handson/releases/tag/handson-10000-20261017) 保存。20 个 LoRA 张量约 177 MiB；官方基座和完整训练 checkpoint 不进入普通 Git 历史。

模型包是推理备份，不包含 Adam 状态。完整训练 checkpoint 仍保留在原训练机器。恢复需要匹配的官方 π0.5 JAX 基座；导出和恢复工具会核对所有基座及完整模型张量的内容哈希。

在已准备匹配基座的新目录中，设置 `PI05_BASE_PARAMS` 为其 `params` 目录：

```bash
mkdir -p exports/pi05_handson_10000
gh release download handson-10000-20261017 \
  --repo sencangWei/pi05-handson --dir exports/pi05_handson_10000
JAX_PLATFORMS=cpu .venv/bin/python scripts/restore_adapter.py \
  exports/pi05_handson_10000 --base-params "$PI05_BASE_PARAMS" \
  --output checkpoints/pi05_handson_lora/my_first_200/9999
```

输出目录必须不存在。恢复后可启动 `scripts/serve_10000.py`，在另一个终端设置以下变量后运行评测：

```bash
export PYTHONPATH="$PWD/openpi/third_party/libero${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$PWD/.local/libero"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
.venv-sim/bin/python scripts/evaluate_libero.py \
  --output evaluations/new-run --trials 4 --seed 42 \
  --initial-state-offset 10 --port 8755 --replan-steps 1 --max-steps 220
```

首次运行前执行上文的 `sim_smoke.py` 创建项目内配置。精确评测记录见 `evaluations/my_first_10000_replan1_init0to3/summary.json` 和 `evaluations/my_first_10000_replan1_init10to13/summary.json`；重跑请使用新的输出目录。

## 操作问题复盘

| 问题 | 原因与处理 |
| --- | --- |
| 下载 dm-control 超时 | 使用 `UV_HTTP_TIMEOUT=300` 重试，保留失败日志 |
| egl-probe 编译失败 | CMake 4 与旧项目策略不兼容；本次安装设置 `CMAKE_POLICY_VERSION_MINIMUM=3.5` |
| 已安装 LIBERO 却无法导入 | editable 模块映射为空；增加 LIBERO 源码搜索路径 |
| 首次导入 LIBERO 要求交互输入 | 先创建项目内的独立 config.yaml，再导入 |
| 初期抓起后掉落、重新抓取失败 | 保留失败轨迹；尚未专门训练或验证恢复能力 |
| 2000 步模型搬运后不松爪 | CSV 显示持续闭合；增加反馈频率后仅个别布局成功，后续增加训练预算 |
| Gym 与 private macro 警告 | 本轮均未导致运行异常；没有为消除提示而替换已验证的依赖 |

## 已执行的远端恢复验证

- [源码恢复记录](evidence/remote-source-restore-verification.json)：从 GitHub 新克隆课程仓库，并从官方远端获取固定版本的 OpenPI、LIBERO；首个备份提交的 100 个课程文件、140 个 OpenPI 文件、1116 个 LIBERO 文件均与本机逐文件字节一致。未使用的 ALOHA 子模块没有初始化。
- [模型恢复记录](evidence/remote-model-restore-verification.json)：重新下载 Release 的全部附件，核对文件 SHA256，使用远端克隆中的恢复脚本和匹配基座重建完整推理模型，再读回落盘参数。全部 71 个张量的形状、类型和内容哈希一致，归一化统计也一致。
- 原模型和恢复模型的参数树 SHA256 均为 `e0080b2b69ff602d461a5c7264112ea5216322cf99538abc7f22ff1d2b5b7515`。
- 本次验证未重新运行训练或仿真，也没有覆盖原模型。远端推理备份不包含优化器状态；原完整 checkpoint 继续留在训练机器。

## 目录

### 下一阶段：运动中的目标位置扰动（尚未报告仿真结果）

`experiment/target-shift` 分支为评测脚本增加可选的 `--target-shift-step` 和 `--target-shift-x-m`。不提供前一个参数时，不修改场景。第一轮协议固定模型 `9999`、初始状态 0、seed=42、每步重新观察、最多 220 步；重启推理服务后，在第 20 个控制步骤执行前将目标碗沿世界 X 轴瞬时平移 +0.05 m。

这属于抓取前的目标重定位测试，不是自然掉落实验。工具要求夹爪处于打开状态且与碗相距至少 0.12 m；仅修改目标自由关节的 X 坐标，并将其速度置零。刷新观测、清空旧动作计划后再推理，保存扰动前后截图、坐标事件和视频。若扰动未发生，不计作有效扰动成功样本。

目前只执行了 `tests/test_target_shift.py` 的 4 项隔离检查及命令行检查，没有自动启动模型、训练或 MuJoCo 仿真。仿真实测由学习者亲自运行；结果尚未产生。

- `scripts/`：数据准备、统计、训练、推理、评测、LoRA 导出与恢复。
- `evidence/`：数据划分、训练来源、环境快照、模型哈希及备份验证记录。
- `logs/`：依赖安装、训练和评测日志。
- `evaluations/`：成功/失败视频、动作 CSV、结果与服务模型身份。
- `assets/`、`previews/`：训练集归一化统计、示范和仿真截图。
- `openpi/`：固定版本的官方源码子模块，其中 LIBERO 也固定版本。

本仓库的课程脚本改编自同一作者的 `embodied-sim-training` 工程。官方源码、基座和公开数据的权利及许可分别归其发布者；采用这些材料时应遵守各上游的许可。
