# 2026-10-19 实际重训：原速与快搬运数据对照

用户请求：重训练。当前两组CPU预检和原生loader检查均通过，实际训练已启动（先控制组，再候选组）；本文件后续追加实际训练、评估结果，不将准备完成写成模型修好。

## 本轮干预

同一父模型 `ui_correct_1792311653585294069/1999`，两臂分别从相同权重重新初始化优化器，2000次新更新、batch4、LR1e-5、seed42、预测窗口10、state条件仍False。保留原37公开+17个人工原件整理出的19段，共13521帧，以及10条排除的公开留出。

仅新增两条来源一致、重新执行物理仿真采集的teacher。控制使用原速1x（新增1531帧，总15052帧），候选只在已审普通搬运区间内部用2x参考进度（新增1394帧，总14915帧）；两者抓/放/恢复段逐步使用原动作，起终物理场景相同。双RGB、pre-state、actualaction、post-state均重新采集，非复用旧画面乘动作标签。生成数据明确 `offline_tracking_teacher`、`human_actions_only=False`，不伪装人工批准。

原速回放先修复了控制器缓存与目标恢复。初版生成器在全程加入追踪反馈，改变抓取接触导致一条滑移，按原门拒绝；v2只在搬运段追踪，匹配控制和候选四条均通过原质量门。两条实际搬运控制步分别160→90、154→87，约1.77x；**这是生成示范的提速，不是模型已学会的提速**，整条仅小幅提速。失败原件仍保留，未进入训练。

两组采用相同60%公开、20%全部批准人工、15%teacher明确搬运、5%teacher其他的起点采样。每个搬运起点的10个真实future labels必须全部位于保护后的原搬运区间，episode末尾由原生loader填充，不拼接不同物理状态。100000虚拟起点保留每个真实起点；本次有限更新不声称实际看过全部起点。配比是本轮受控实验的选择，不是通用最优参数。

## 证据

- `experiment-plan.json`：共同参数、源/归一化哈希及预声明评估。
- `controller-restoration.json`、`rejected-v1-*.json`：恢复修正和被拒初版。
- `teacher-rate1.json`、`teacher-rate2.json`、`teacher-artifact-check.json`：实际物理轨迹及RGB/动作/视频核验。
- `loader-check.json`：真实OpenPI/LeRobot读取首、中、尾及边界10动作窗口，状态与双RGB相同，模型输入shape正确，不加载权重。
- 原始新teacher与MP4位于本机 `.local/retrain-20261019/teacher-pilot/carry_only_v2/`，原人工数据未修改。

## 训练入口

```bash
# 在项目根目录，先CPU预检；删除 --dry-run 才会实际训练。两臂的输出目录须不存在。
JAX_PLATFORMS=cpu .venv/bin/python scripts/train_tracking_teacher.py \
  --manifest evidence/tracking-teacher-retrain_rate2_mix_v2-manifest.json \
  --parent-checkpoint checkpoints/pi05_handson_lora/ui_correct_1792311653585294069/1999 \
  --experiment retrain_teacher_2x_20261019 \
  --steps 2000 --batch-size 4 --learning-rate 1e-5 --dry-run
```

实际训练需去掉CPU限制，使用现有JAX/CUDA环境；控制臂对应rate1数据和 `retrain_teacher_1x_20261019`。不得覆盖现有实验或把fresh optimizer称精确resume。checkpoint索引1000表示已经执行1001次更新，最终1999对应2000次。

## 评估边界

预声明零位移、两个合法远负位置、一个正位置，策略seed0/1/2，各臂共12次；每次最多800步、每执行1步重新看图。连续10步On/实际开爪/开爪命令/撤手后，再有界追加50个策略动作检查碗漂移、重抓与On保持。跟进步数跑完不等于稳定性通过，需从真实CSV单独回算。已看过的布局属于开发验证，不能称未见测试。

本轮仅补搬运标签与保留既有恢复数据，没有自动补出从home到所有远端抓取的缺失路径，不保证修复远端自恢复。状态输入A/B已审查可行但本轮暂不改，避免同时改变数据和模型输入。
