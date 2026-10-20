# 负向边缘纠错：新数据重点抽样＋旧数据回放

## 结果与使用决定

本轮实际训练已完成，checkpoint `ui_correct_1792480483802126335/1999`。这是从当前模型 `ui_correct_1792434450272918886/999` 初始化全部权重的2000次新增LoRA更新；重新初始化Adam、scheduler、RNG和数据游标，不是精确断点续训。

- 预先固定的目标位置（布局0，碗X/Y均−10cm）：严格通过 **1/2→2/2**。
- 四个开发对照位置：严格通过 **6/8→6/8**；原有单条成功发生退步，不能用总数掩盖。
- 所有10条配对试验：严格通过 **7/10→8/10**。仅为已见开发位置、两个推理种子的有限结果，不是泛化成功率。
- 两模型都严格通过的五组中，新模型全部需要更多控制步：+77～246步，平均+133.4步。**仍未解决慢的问题，不推荐直接替换旧模型**。旧模型继续在8755端口供面板使用，实验子模型可以在下拉框手动选择。
- 不依据loss宣布任务成功；本轮没有自动追加第二次训练。

## 数据没有删除

之前47条/6126帧的Parquet文件逐字节保留。新增4条已批准人工纠错共805帧，形成51条/6931帧：公开37条/4400帧、旧人工10条连续轨迹或选定片段/1726帧、新人工4条/805帧。草稿及之前暂缓数据没有悄悄加入。

抽样目标：公开50%、旧人工20%、新人工30%。以200000个虚拟观测起点索引重复和打乱原始样本；6931个真实起点全部可用。比例是长期抽样目标，不要求每个batch恰好按比例分配。数据没有增加新场景，也没有拼接、倍乘或加速动作标签。

检查：全部2531个人工动作/状态标签与原始录制完全相等；153个边界动作窗口和23个实际重抽样窗口/图像通过读回。每个动作窗口仍为10×7，状态8维。归一化沿用父模型的固定统计，未重新计算。独立公开留出10条/1170帧未入训练；本次没有评估该留出集，不能宣称其指标改善。

人工片段复核：4条完整双相机动作视频已生成，125对采样画面（包含全部命令段边界）、13张联系表已人工查看；不是逐帧目检或完整物理重放。第二、三条有较长精细下降；第三条含一次提前松爪后重夹调整。连续恢复保留，不能把它们称为完美专家数据。

## 配置

2000更新，batch4，学习率1e-5（20步warmup后固定），训练seed42，horizon10；只训练LoRA，AdamW，梯度范数裁剪1.0。总训练窗口数8000，不能称为8000条独立示范。

源码：OpenPI `15a9616a00943ada6c20a0f158e3adb39df2ccac`，课程源码 `717f5833d40fbae53a632eb130bb74cf62febaa1`。运行环境分别见training-environment.json、simulation-environment.json。environment.json为系统Python的发现快照，不是实际训练解释器。

```bash
cd ~/.pi05-handson
.venv/bin/python -u scripts/train_corrections.py \
  --manifest evidence/corrections-edge_recovery_focus_20261020-manifest.json \
  --parent-checkpoint checkpoints/pi05_handson_lora/ui_correct_1792434450272918886/999 \
  --steps 2000 --batch-size 4 --learning-rate 1e-5 \
  --experiment NEW_UNIQUE_NAME
```

命令用于复现，不要覆盖已存在实验。完整实际启动记录见actual-launch-config.json及上层ui_correct_1792480483802126335-launch.json；完整日志training.log。最终1999异步保存完成、训练进程退出均已核查。

## 固定评测与全部结果

训练前冻结plan.json；相同物理初始状态、每条连接固定推理seed0/1；每次只执行新预测的第一步；最多800控制步。连续10步满足On、开爪命令、真实夹爪间隙≥6cm、末端撤离碗≥10cm，之后继续策略50步。严格通过还要求50步全部门限通过、碗漂移≤5mm。父模型评测在新模型训练完成后执行，但父权重哈希未变、评测矩阵事先固定。

| 位置 | 推理seed | 旧模型 | 新模型 |
|---|---|---|---|
| negative_edge_target | 0 | 直到800步仍无完整松爪撤手（800步） | 通过（403步） |
| negative_edge_target | 1 | 通过（350步） | 通过（486步） |
| first_regrasp_control | 0 | 第800步才满足10步门限，后续观察不足（800步） | 通过（364步） |
| first_regrasp_control | 1 | 通过（328步） | 直到800步仍无完整松爪撤手（800步） |
| near_negative_control | 0 | 通过（234步） | 通过（311步） |
| near_negative_control | 1 | 通过（229步） | 通过（475步） |
| positive_control | 0 | 直到800步仍无完整松爪撤手（800步） | 通过（471步） |
| positive_control | 1 | 通过（240步） | 通过（353步） |
| positive_edge_control | 0 | 通过（303步） | 通过（398步） |
| positive_edge_control | 1 | 通过（259步） | 后续闭爪命令一次，碗仍稳定（387步） |

### 两个未通过案例的实际含义

1. first_regrasp_control/seed1：第212步已On，到800步仍持续闭爪；最后爪间隙约8.8mm、末端距碗约68.7mm。第400/600/800步闭爪命令约0.99，说明策略确实继续输出闭爪和小动作；不是推理进程停掉。联系表已查看。800是最终预算，但碗到盘后仍有588步，不能将不松手归因于运行只差几步。
2. positive_edge_control/seed1：已放稳、开爪、撤手，后50步中第386步输出了一次闭爪命令，第387步重新开爪。全程碗仍在盘上，漂移约0.00034mm，实际爪间隙约79mm、末端距碗213mm。**这是预定持续开爪门限未通过，不是碗掉落或放置失败**。不事后修改门限掩盖，也不把它说成实质性掉碗。

动作模式说明局部松爪/撤手决策和速度仍需改进；本次没有权重比例消融，不能证明根因一定是数据质量、比例、步数或过拟合。所有新人工示范来自同一个负向边缘区域，仅提高抽样频率不会自动补齐其它状态覆盖。

复现评测（已存在输出会拒绝覆盖；中断恢复用--resume）：

```bash
.venv/bin/python evidence/edge-recovery-20261020/evaluate.py --arm parent
.venv/bin/python evidence/edge-recovery-20261020/evaluate.py --arm candidate
.venv/bin/python evidence/edge-recovery-20261020/summarize.py
```

index.html含全部20条仿真视频，review/index.html含4条人工双相机片段。rollouts/归档全部20条CSV/结果/状态，可用 `.venv/bin/python evidence/edge-recovery-20261020/audit_artifacts.py` 无模型重算并核对视频、状态与指标哈希。视频按控制帧保存，不能用视频播放秒数替代包含模型推理与渲染的实际墙钟耗时。部分父模型评测期间有CPU导出活动，故仅比较配对控制步，不宣称纯推理速度差。

## 备份范围

实验模型以公开prerelease `edge-recovery-20261020` 保存，正式权重需要精确匹配的官方π0.5基座恢复；adapter仅含20个LoRA张量、元数据与归一化，无优化器。

另附4条新纠错的805帧完整原始记录及5个初始场景，共835个文件，raw-backup-manifest.json逐文件列出哈希。旧47条训练数据仍本机保留，不在本次新增数据压缩包中。训练checkpoint/train_state仍在本机，不能把推理adapter称为完整可续训checkpoint。

远端下载已核查完成：model-restore-verification.json确认71张量及归一化与原模型逐项相同；raw-restore-verification.json确认四个远端资产的SHA256相同，并实际恢复835个原始文件逐个比对通过；release-verification.json确认公开prerelease及4个资产状态。代码、配置、视频与证据通过experiment/correction-learning分支备份，并以独立干净工作树应用远端补丁、比较Git树和逐文件SHA256核查。代码恢复证明写入本机 `.local/edge-recovery-remote-20261020/code-restore-proof.json`，该证明不加入自身校验的Git树以避免自引用。
