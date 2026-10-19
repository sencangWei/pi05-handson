# 负偏移放置与恢复：本轮重训报告（2026-10-19）

## 结论

已完成数据审计、补充8条连续离线示范、真实2000次LoRA更新，以及旧/新模型各20次配对仿真。严格成功从 **6/20 → 13/20**，官方On条件触发从7/20→18/20。结果是局部改善，**仍有失败，未达到预声明推荐切换门槛**；新模型保留为实验版，供用户手动试跑。

| 场景组 | 旧模型严格成功 | 新模型严格成功 |
| --- | ---: | ---: |
| 本轮训练位置 | 2/8 | 5/8 |
| 常规开发对照 | 4/4 | 4/4 |
| 未用于新增示范的位置 | 0/4 | 3/4 |
| 训练用失败持碗状态 | 0/4 | 1/4 |
| 合计 | 6/20 | 13/20 |

严格成功要求首次连续10步同时满足On、开爪命令、实际爪距≥6cm、末端离碗≥10cm；再由模型执行50步，全程保持这些条件，碗漂移≤5mm。全部40次均有效，没有把缺失记录或程序异常算作任务失败。两模型初始物理状态、源文件、策略seed0/1、replan=1、800步上限均配对。800步是仿真控制步，2000步是训练参数更新，二者不同。

### 后续闭爪与物理稳定的区别

新模型有5次到800步仍未触发完整释放，另2次已释放但跟进期再发短促闭爪命令：holding008/seed0第449步；holding009/seed1第552、559步。这两次的碗、On、实际爪距和撤手距离在50步中均稳定，仅命令门未过；碗最大漂移分别3.94e-7m、3.53e-7m。不能描述为碗掉落，也不能放宽预声明门后偷偷计为严格成功。持碗严格成绩为1/4。

## 对照演示与速度

[左右同步实际演示](user008-seed0-comparison.mp4)，[封面](user008-seed0-comparison.png)，[最终画面](user008-seed0-comparison.final.png)。固定展示样本在候选结果产生前选定，是训练位置user008/seed0。旧模型638步完成，新模型382步；首次10步释放确认588→332。按相同控制步对齐，短侧结束后标明冻结，20FPS对应原生物理20Hz；省略推理等待，不代表实际墙钟速度。

共同严格成功的6对里，只有user008的两个seed分别快256、160步；zero分别慢8、85步，positive057分别慢126、57步。**不能称为整体变快**。全部成功和失败的原始录像、CSV、JSON与初末状态在[本轮评测目录](../../evaluations/negative-recovery-20261019/)；逐帧校验见comparison manifest与ffprobe文件。

## 数据从哪里来，保留了什么

已批准17份人工原件，实际保留19段：10462原始帧→9121保留帧；public37另4400帧，10条公开留出示范不进训练。草稿不计入，没有删除人工纠正。已有负偏移示范存在；人工帧中worldX<0且worldY≥.35m的组合为0帧，当前失败Y约.373m。这仅说明已审计人工数据覆盖差异，不能断言公开数据也缺失或这是唯一原因。多数人工动作并非停顿。

新增6条从home开始的完整抓取搬运放置示范2055帧，2条从实际失败持碗状态开始的恢复示范458帧，共2513帧。每条均连续真实env.step，无轨迹内传送、无失败动作充当专家标签。物体GT位姿只用于离线生成，明确标为非人工teacher；模型运行时没有GT控制器。8条示范均通过连续状态、原生20Hz、双摄录像及末60步释放稳定门。示范成功不等于模型学会。

总训练数据64段16034帧。原公开/人工前缀逐字节一致；实际OpenPI loader的32个样本与原始RGB、state、未来10步动作窗口核对通过。新示范的持碗搬运阶段为21–33控制步，处理的是实际动作数据，未靠加速视频制造快动作。

## 真实训练

父模型`retrain_teacher_1x_20261019/1999` → 新模型`negative_quadrants_20261019/1999`。2000次额外更新，batch=4，seed42，20步warmup后lr=1e-5，LoRA only，动作窗口10；35%公开、25%全部批准人工、30%完整新示范、10%持碗恢复采样。状态token关闭，条件为RGB与语言；有状态记录不代表模型输入了状态。

加载父模型权重后重置Adam、调度器、RNG和数据游标，属于继续微调，非精确断点续训。已直接读取保存的step、Adam和调度计数均为2000；优化器仅含20个LoRA参数张量（Adam一、二阶矩各20个）。日志200条指标全部finite，最终异步保存完成。归一化沿用父模型，不混入留出数据。[训练审计](training-verification.json)、[损失曲线](training-loss.png)、[实际启动配置](../negative_quadrants_20261019-launch.json)。损失不是成功率。

本轮实际命令（已有输出保留，重跑须另命名实验）：

```bash
.venv/bin/python -u scripts/train_placement_teacher.py \
  --manifest evidence/placement-teacher-negative_quadrants_mix_v1-manifest.json \
  --parent-checkpoint checkpoints/pi05_handson_lora/retrain_teacher_1x_20261019/1999 \
  --experiment negative_quadrants_20261019 \
  --steps 2000 --batch-size 4 --learning-rate 1e-5
```

## 评测器修正与计分边界

新增持碗场景原先只恢复MuJoCo物理状态，没有恢复控制器爪距累计量，闭爪命令反而先张爪。已隔离A/B验证，并以显式`--initialize-saved-controller`恢复，缺参数在仿真前拒绝。2次旧seed0诊断归档排除，4次规范holding重跑；16次home旧路径保持。此缺陷不能被说成原GUI负偏移慢动作的原因。详见holding-evaluation-protocol-amendment与holding-evaluator-audit。

[最终40次计分与配对](evaluation-summary.json)、[推荐门与GUI选择](deployment-decision.json)、[剩余失败的实际动作和位移](remaining-failure-diagnostics.json)。例如seed0的user055目标碗曾升高约12cm后仍悬持；heldout_far目标碗未抬起，末100步末端中位位移约0.33mm/步。这些是观测，不能据此声称识别到某种模型心理状态或唯一根因。

这是单任务、两处留出位置、两个策略seed、一个训练seed的小规模开发实验，不是LIBERO官方benchmark或真机结果。新模型从旧模型额外训练2000步，同时改了数据和采样，不能把差异唯一归因于数据量、质量或训练时长。

## 软件中自己试

恢复界面时选中新实验模型，标题为“负偏移增补实验版（仍有失败）”，以布局24、X≈−4.0cm、Y≈+6.9cm暂停启动，执行动作数为0。点击“运行”开始；改位置后点“加载/重置布局”。旧模型可从模型列表选择，训练父模型仍保留原对照模型。GUI这次重置布局供手动查看，不替代上面的精确保存初态配对评测。

[界面窗口截图](gui-ready.png)为两个自有窗口客户区的并排拼接；已核对真实桌面两窗可见且不重叠。暂停及模型元数据见[启动记录](gui-ready.json)。

## 后续训练安排

1. 将“抓取前未进入夹口”“已持碗但搬运停滞”“已放稳后反复闭爪”分开采样，补对应失败状态之前、恢复过程及恢复后的连续稳定结尾；原批准人工数据继续保留。
2. 对本轮见过的远端位置按开发样本管理；下一轮另设未参与选片、训练及调参的新位置，不能反复看这两处后继续宣称独立最终测试。
3. 在相同父模型、数据、采样和固定开发矩阵下比较多个训练预算/中间checkpoint，按闭环指标选择，不能单凭loss更低或训练更久下结论。下一轮未自动启动。

## 备份与复查

代码/配置/报告/成功及失败录像备份到用户仓库`experiment/correction-learning`分支；最终提交在交付时重新fetch到干净基线，应用完整diff并比对树与文件内容。OpenPI与LIBERO上游源码未改。

[实验版adapter与演示发布页](https://github.com/sencangWei/pi05-handson/releases/tag/negative-quadrants-20261019)。发布的adapter从远端重新下载，结合单独的官方基座重建、保存并重新读取，71个参数张量SHA与原训练后参数全一致。[远端adapter恢复证明](remote-adapter-verification.json)。这是推理权重备份，不含官方基座、优化器状态或原始NPZ；原始数据仍保留本地，不称为已上传备份。

50项针对性测试与Ruff通过，见final-code-checks.json。只复算已有结果可执行：

```bash
.venv/bin/python scripts/summarize_negative_recovery.py \
  --plan evidence/negative-recovery-20261019/experiment-plan.json \
  --runs evaluations/negative-recovery-20261019 \
  --output evidence/negative-recovery-20261019/evaluation-summary.json
```

## 每次结果（步数包含50步后续检查）

| 场景 | seed | 旧模型完成步数 | 新模型完成步数 |
| --- | ---: | ---: | ---: |
| user008 | 0 | 638 | 382 |
| user008 | 1 | 545 | 385 |
| user009 | 0 | 800步无释放 | 305 |
| user009 | 1 | 800步无释放 | 591 |
| user054 | 0 | 800步无释放 | 800步无释放 |
| user054 | 1 | 800步无释放 | 674 |
| user055 | 0 | 800步无释放 | 800步无释放 |
| user055 | 1 | 800步无释放 | 800步无释放 |
| zero | 0 | 306 | 314 |
| zero | 1 | 237 | 322 |
| positive057 | 0 | 256 | 382 |
| positive057 | 1 | 372 | 429 |
| heldout_mid | 0 | 800步无释放 | 348 |
| heldout_mid | 1 | 800步无释放 | 405 |
| heldout_far | 0 | 800步无释放 | 800步无释放 |
| heldout_far | 1 | 800步无释放 | 607 |
| holding008 | 0 | 800步无释放 | 后续闭爪指令未过严格门 |
| holding008 | 1 | 800步无释放 | 800步无释放 |
| holding009 | 0 | 800步无释放 | 268 |
| holding009 | 1 | 800步无释放 | 后续闭爪指令未过严格门 |
