# π0.5 LoRA：从示范数据到 LIBERO 仿真执行

本项目记录一次亲手完成的机器人模仿学习实践：公开示范检查、数据准备、归一化、LoRA 微调、闭环仿真、失败分析和模型备份。使用预训练 π0.5 基座进行单任务微调。

**任务：** `pick up the black bowl next to the plate and place it on the plate`。

## 实测结果

最终 checkpoint 为 `9999`，即完成 10000 次参数更新。固定预测动作窗口为 10，最终评测每执行 1 步就重新观察。每次测试最多执行 220 步，采用 LIBERO 自带任务成功判定。

| 模型 | 执行后重新观察的间隔 | 初始状态 | 官方 On 条件触发次数 |
| --- | --- | --- | --- |
| 200 步 | 5 步 | 0 | 0/1 |
| 2000 步 | 5 步 | 0 | 0/1 |
| 2000 步 | 1 步 | 0、1、2、3 | 1/4 |
| 10000 步 | 1 步 | 0、1、2、3 | 4/4 |
| 10000 步 | 1 步 | 10、11、12、13 | 4/4 |

最后一组布局在冻结最终模型和执行参数后测试，未用于前面的调参。最终模型在这 8 个初始状态上均触发了官方成功条件，没有程序异常。**指标口径更正：该任务的 On 条件检查碗与盘接触、相对高度和水平距离小于 3 cm，没有要求松爪或撤手；8 条轨迹的最后一条夹爪指令仍均为闭合。因此 8/8 仅指官方条件触发，不代表已验证完整的松爪、稳定放置和撤手。** 详见 `evidence/success-criterion-audit.json`。这是同一仿真任务的小规模观察；尚未验证掉落恢复、跨任务泛化或真机迁移。策略采样 RNG 在服务启动时初始化为 0，随后连续推进；每次 trial 不会单独重置采样 RNG。早期与后期模型的轨迹长度不同，后续 trial 的采样序列也会不同。

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

## 实验分支与可视化

### 下一阶段：运动中的目标位置扰动（尚未报告仿真结果）

`experiment/target-shift` 分支为评测脚本增加可选的 `--target-shift-step` 和 `--target-shift-x-m`。不提供前一个参数时，不修改场景。第一轮协议固定模型 `9999`、初始状态 0、seed=42、每步重新观察、最多 220 步；重启推理服务后，在第 20 个控制步骤执行前将目标碗沿世界 X 轴瞬时平移 +0.05 m。

这属于抓取前的目标重定位测试，不是自然掉落实验。工具要求夹爪处于打开状态且与碗相距至少 0.12 m；仅修改目标自由关节的 X 坐标，并将其速度置零。刷新观测、清空旧动作计划后再推理，保存扰动前后截图、坐标事件和视频。若扰动未发生，不计作有效扰动成功样本。

目前已通过位置扰动、窗口控制和曲线解析的 8 项隔离检查及命令行检查。位置扰动的仿真实测由学习者亲自运行；结果尚未产生。

### 实时三维窗口

`--viewer` 使用 MuJoCo 3.2.3 原生交互窗口，连接当前 LIBERO 场景和策略循环。窗口初始暂停：空格运行/暂停，N 执行一个控制步骤，Q 结束。鼠标可调整观察视角；结束后保留最终场景，关闭窗口或按 Q 后保存结果。用户主动关闭未完成的窗口记为 `user_stopped`，不算有效成功率样本。

保持 `serve_10000.py` 的模型服务运行，在桌面终端执行：

```bash
live_run="live_view_$(date +%Y%m%d_%H%M%S_%N)"
LIBERO_CONFIG_PATH="$PWD/.local/libero" \
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
.venv-sim/bin/python scripts/evaluate_libero.py \
  --output "evaluations/$live_run" --trials 1 --seed 42 \
  --initial-state-offset 0 --port 8755 --replan-steps 1 --max-steps 220 --viewer \
  2>&1 | tee "logs/$live_run.log"
```

Viewer 模式会在导入仿真库前选择 GLX。机器人策略仍接收原来的两路相机观测；三维窗口相机只是观察视角。GUI 允许手动修改或扰动场景，故交互结果单独标记，不直接合并到上面的固定评测结果。工具未逐项记录手动 GUI 编辑。操作规范参考 [MuJoCo 原生 viewer 文档](https://mujoco.readthedocs.io/en/3.2.3/python.html#interactive-viewer)。

首次窗口演示已由学习者运行：初始状态 0，111 个控制步骤触发官方 On 条件，无程序异常，见 [结果](evaluations/live_view_init0/trial-000_init-000_result.json) 和 [录像](evaluations/live_view_init0/trial-000_init-000_success.mp4)。GUI 的 wall_duration 包括启动暂停、交互等待和推理时间，不能当作纯运动或推理耗时。此后重复使用原输出目录触发了拒绝覆盖保护，原始结果仍保留；上面的命令每次生成新目录和新日志。

默认评测在官方 On 条件首次触发时停止，因此窗口可能停在夹爪尚未打开的状态。加上 `--post-success-steps 40` 可在首次成功信号后继续观察最多 40 个由模型预测的控制步骤，仍受 `--max-steps` 限制。原成功信号和最终信号分别记录，`release_and_retreat_success` 保持 null，直到建立并实测更完整的验收条件。续行动作由策略预测，没有额外编排松爪或抬手动作。

### 训练曲线界面

`export_training_curves.py` 将已存在的三段训练日志导入 TensorBoard，提供 loss、grad_norm 和 param_norm，共 1000 个历史记录点。横轴应使用 Step；事件的 wall_time 是导出时间，不是原训练耗时。本操作不启动训练。

## 放置收尾专项训练（已训练，初始状态 0 已观察到松爪与撤手）

训练由 JAX 后台程序执行，可从下面的中文工作台点击启动；TensorBoard 用于看训练曲线，MuJoCo 用于观察和交互仿真。

学习者已完成 `release_focus_2000` 新增 2000 次更新，并报告初始状态 0 成功。记录显示第 109 步触发 On 条件，随后执行 40 步；最后 10 步均为开爪命令且 On 条件保持成立，末端相对首次触发前观测抬高约 12.6 cm。详见 [首次收尾观察](evidence/release-focus-first-observation.json)。这是一条布局的观察，尚无多布局成功率；同预算均匀采样对照尚未训练，不能把改善单独归因于重采样。当前 TensorBoard 只显示已导入的日志；工作台的“训练曲线（快照）”可在训练中或训练后按需导入已有 Step 指标，不是持续实时推送。

续行诊断见 [记录](evidence/release-followup-diagnosis.json)：到达盘子后实际执行 40 步，其中 38 条闭爪、2 条短暂开爪，40 条向下指令，未完成松爪、撤手。16 条训练示范都有最后开爪的片段，转换数据没有截掉这些标签。

新实验检验“提高收尾片段采样频率是否改善阶段切换”这一假设，不预先保证有效：

| 项目 | 设置 |
| --- | --- |
| 初始化权重 | 原 `my_first_200/9999`，保留原模型 |
| 新实验 | `release_focus_2000` |
| 更新预算 | 新增 2000 次更新，batch=4，seed=42 |
| 优化器 | 新建 AdamW；优化器、学习率步数、随机数和数据游标重新开始，属于继续微调，不是精确断点恢复 |
| 学习率 | 前 20 步预热，随后 5e-5；仅 LoRA 可训练 |
| 采样 | 每条示范最后开爪片段，加上其之前 10 帧，共 355 帧，各出现 4 次；其余帧各 1 次 |
| 数据量 | 原 1971 帧保持不变；采样映射为 3036 项，收尾及前置片段占约 46.8% |
| 数据与统计 | 仍只使用原 16 条训练示范；4 条留出不参与；归一化沿用原统计 |
| 动作 | 预测 10 步动作；评测每次执行 1 步再观察 |
| 保存位置 | `checkpoints/pi05_handson_lora/release_focus_2000/1000`、`1999` |

### 第一步：学习者启动训练

先在仿真窗口按 Q 保存并退出；在自己启动 `serve_10000.py` 的终端按 Ctrl+C，释放该推理服务占用的显存。保留其他 GPU 任务。另一个终端执行：

```bash
cd ~/.pi05-handson
export HF_HUB_OFFLINE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.55
export OMP_NUM_THREADS=4

.venv/bin/python -u scripts/train_release_focus.py \
  2>&1 | tee "logs/release-focus-training-$(date +%Y%m%d_%H%M%S_%N).log"
```

入口会先核对训练文件哈希、索引顺序和归一化，然后打印采样设置，再启动训练。最后编号是新实验的 `1999`，表示新增 2000 次更新；参数来自此前已经训练 10000 次更新的模型。入口拒绝覆盖已有实验和已有启动记录。若训练被中断，先保留日志和目录排查，再确定恢复方式。

准备检查可单独运行 `JAX_PLATFORMS=cpu .venv/bin/python scripts/train_release_focus.py --dry-run`；它只读取数据和配置，不加载模型权重、不创建优化器，也不开始训练。

### 第二步：训练结束后加载新模型

确认 `release_focus_2000/1999` 已完成保存，再在推理终端运行：

```bash
cd ~/.pi05-handson
HF_HUB_OFFLINE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 OMP_NUM_THREADS=4 \
.venv/bin/python -u scripts/serve_release_focus.py \
  2>&1 | tee "logs/serve-release-focus-$(date +%Y%m%d_%H%M%S_%N).log"
```

它监听同一个 `127.0.0.1:8755`；旧服务必须先退出。

### 第三步：学习者在 MuJoCo 验证收尾

```bash
cd ~/.pi05-handson
release_run="release_focus_$(date +%Y%m%d_%H%M%S_%N)"
LIBERO_CONFIG_PATH="$PWD/.local/libero" \
PYTHONPATH="$PWD/openpi/third_party/libero${PYTHONPATH:+:$PYTHONPATH}" \
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
.venv-sim/bin/python -u scripts/evaluate_libero.py \
  --output "evaluations/$release_run" \
  --trials 1 --seed 42 --initial-state-offset 0 \
  --port 8755 --replan-steps 1 --max-steps 220 \
  --post-success-steps 40 --viewer \
  2>&1 | tee "logs/$release_run.log"
```

按空格开始，结束后按 Q 保存录像和结果。除了官方 On 条件，还要检查夹指是否持续打开、手是否撤离，以及碗是否仍稳定留在盘上。视频人工复核属于初步证据；当前 `release_and_retreat_success=null`，没有自动完整验收指标。短暂发出开爪命令不能算完成释放。

初始状态 0–3 用于开发复盘。采样改动是否有效，需使用 `--uniform-control` 另训 `release_uniform_2000`（相同权重来源、新优化器、预算和种子，仅采样不同），并用 `serve_release_focus.py --uniform-control` 加载对照。比较各模型时重新启动各自服务，保持采样随机数初始种子相同、相同的测试顺序和反馈设置。原模型与专项模型比较还增加了训练预算，不能单独归因于重采样。新的初始状态 14–17 预留为配置冻结后的测试；尚未测试，不报告成功率。

## 本地仿真实验面板

桌面双击 **π0.5 仿真实验室**，即可打开中文工作台；加载布局后另开 MuJoCo 三维窗口。也可运行：

```bash
cd ~/.pi05-handson
/usr/bin/python3 scripts/lab_launcher.py
```

在新机器安装桌面入口：`/usr/bin/python3 scripts/lab_launcher.py --install-shortcut`。启动失败会弹出日志位置；同一时间只允许一个面板，直接运行 `simulation_lab.py` 也有单实例保护。打开面板不会自动加载模型、创建仿真或启动训练。

### 模型与训练

- **推理 checkpoint → 启动所选模型 → 检查连接**：从本课程完整 checkpoint 中选择；加载日志在页面内显示。已有本面板模型会先停止再切换。界面没有读入新 checkpoint 时点击“刷新模型”。
- **外部服务**：原终端运行的 `serve_release_focus.py` 可以直接连接。切换模型或开始训练前，需先在它的原终端按 Ctrl+C；软件不接管或终止外部服务。
- **继续训练起点 → 设置 Steps、Batch、学习率和采样 → 开始训练**：默认新增 2000 次更新、batch=4、学习率 5e-5。采样可选收尾重点或均匀对照。每次新建唯一实验目录，初始化所选权重并重建优化器、调度器和随机数，**不是精确断点续训**。沿用原 16/4 训练和留出划分、训练集归一化与 10 步动作窗口。
- 启动训练会先保存并关闭当前三维场景、停止本面板模型，训练期间禁止运行策略。停止按钮只操作经 PID、UID、进程创建时间和命令行核验的本面板进程；中断后尚未保存的更新不恢复。
- 日志在页内刷新；**训练曲线（快照）**导出当前日志已出现的 Step 指标并打开本地 TensorBoard。想看后续数据再次点击；横轴用 Step，wall_time 为快照导出时间。
- 关闭面板后，后台模型、训练和 TensorBoard 继续运行。重新打开可从 `.local/lab-jobs.json` 恢复管理；单独关闭窗口不等于停止训练。

### 仿真操作

先在旧评测三维窗口按 Q 保存退出；模型服务可以保留，然后使用以下按钮：

1. 选择布局编号，点击 **加载/重置布局**，三维窗口显示该布局；初始为暂停。
2. X/Y 偏移以厘米显示，范围 ±10 cm，相对该布局原始碗位置；修改后再次点击 **加载/重置布局** 生效。保留碗的高度和朝向；**原位置**恢复零偏移。
3. 点击 **运行**、**暂停**或**单步**。三维窗口中的空格、N 同样可用。模型推理在后台；暂停、重置会使旧预测失效，再次运行根据新观测预测。
4. 点击 **结束并保存**后保持场景可观察，重置可开启下一轮。三维窗口按 Q 会保存并结束当前场景，操作面板仍可继续加载。
5. 首次 On 信号之后继续 40 步，或者达到 220 个控制步时，自动结束并保存；松爪和撤手仍需观察，On 信号本身不等于完整收尾。

每次启动创建独立的 `evaluations/interactive_lab_*` 目录；每个场景保存在 `scene-*_init-*` 子目录中：

- `actions.csv`：执行前 8 维状态、模型实际执行的 7 维动作、执行后末端/碗/夹指位置及 On 信号。
- `video.mp4`：有控制动作时保存；初始画面加每个控制步一帧，按 10 FPS 回放，暂停时间不进入视频。
- `scene.json`、`result.json`：布局、位置修改、模型元数据、停止原因和步数。
- `initial_state.npy`、`final_state.npy`、首尾 PNG：物理状态快照及画面；快照不包含策略随机数或控制器完整内部状态，不能声称精确续跑。
- 会话根目录的 `operations.jsonl`：运行、暂停、重置、位置修改、保存和错误事件。检测到原生窗口物理状态变化时另存快照；不保证记录原生界面的所有输入。

重置和退出先保存上一轮；编码或保存失败时保留内存记录以便重试，不直接销毁旧场景。面板实验记录为 `formal_benchmark=false`，与固定初始状态的批量评测分开；原训练、评测脚本继续使用原入口。

### 结果与录像

点击 **刷新结果**，选择一条记录即可播放录像或打开其目录；列表读取最近 200 条已保存结果。仿真页另有相机截图、重置相机、播放当前录像按钮。录像在本轮结束保存后可播放，两个相机截图需先加载场景。

官方 On 判据与人工“收尾成功/收尾失败”分开显示。播放检查后填写备注并保存复核，生成单独 `*.review.json`，不会改写原始成功字段；原结果变化后复核显示“复核过期”。未保存的活动窗口结果暂不在列表内。

验证使用真实 Tk、实际工作台与本地 checkpoint/结果目录，仿真任务为占位对象；另用隔离环境验证进程管理、训练配置和真实视频文件导出。**本轮未由代理启动真实模型、训练或仿真，首次完整运行由学习者点击完成。** 界面截图：[模型与训练](previews/lab-workbench-training-ui-check.png)、[结果与录像](previews/lab-workbench-results-ui-check.png)；截图标注为界面检查，不是新的策略演示。

当前软件管理本课程的单任务训练和交互评测；键盘遥操作采集、完整数据标注/编辑、新任务场景制作、多任务训练尚未集成。

## 目录

- `scripts/`：数据准备、统计、训练、推理、评测、LoRA 导出与恢复。
- `evidence/`：数据划分、训练来源、环境快照、模型哈希及备份验证记录。
- `logs/`：依赖安装、训练和评测日志。
- `evaluations/`：成功/失败视频、动作 CSV、结果与服务模型身份。
- `assets/`、`previews/`：训练集归一化统计、示范和仿真截图。
- `openpi/`：固定版本的官方源码子模块，其中 LIBERO 也固定版本。

本仓库的课程脚本改编自同一作者的 `embodied-sim-training` 工程。官方源码、基座和公开数据的权利及许可分别归其发布者；采用这些材料时应遵守各上游的许可。
