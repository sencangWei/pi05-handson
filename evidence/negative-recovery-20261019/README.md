# 负位置放置与恢复：数据覆盖实验

状态：新数据质量门通过；模型重训练及配对评测尚未完成。

## 已确认

已批准17份人工原件/实际保留19段全部保留，10462原始帧→9121保留帧，public37额外4400帧，10条public留出不进入训练。已有负偏移示范存在；本轮审计的人工帧里worldX<0且worldY>=.35m共0帧，当前失败Y约.373m。这是人工覆盖差异，不能据此断言public37也无覆盖或唯一根因。多数原人工动作并非停顿，不能再以训练数据多数停顿作解释。

## 实际新增

6条从home开始的真实离线teacher，共2055帧；2条从用户失败末状态开始的持碗放置恢复，共458帧。每条都连续真实env.step，fresh RGB/状态→实际action→fresh post记录；没有把失败策略动作当作专家标签。GT物体位姿只用于离线teacher，不注入运行时VLA。原人工批准制度不变，新teacher明确非人工。

8条均通过末60步On+开爪命令+实际gap>=.06m+末端距碗>=.10m，首次10步确认之后50步稳定随访；同一份物理轨迹、原生20Hz、实际MP4双视角，不是视频加速。新的完整搬运21–33控制步。终点成功不等于模型学会；还需真正训练与闭环评测。

新增位置检查曾拒绝重叠4.4mm、1.1mm的候选，不放宽碰撞门；最终冻结两个相邻holdout位置，只做评测，未生成它们的teacher。

## 重现与存储边界

源码在scripts/generate_placement_recovery_teacher.py、generate_retargeted_task_teacher.py、prepare_placement_teacher.py、train_placement_teacher.py。原始NPZ仍在本地.local/negative-recovery-20261019，原人工原件在corrections；本证据包保存生成metadata/每帧哈希、源初末状态、双摄视频、阶段记录与审计，不把未上传的原始数据称为远端数据备份。源场景快照另见source-scenes；完整teacher均使用同一固定脚本版本。

experiment-plan.json在训练前冻结训练预算、采样比例和两arm各20次评测。训练位置、历史开发对照与两个未用于teacher的相邻位置分别报告，单训练seed42、2个策略seed，不是LIBERO官方benchmark或真机结果。
