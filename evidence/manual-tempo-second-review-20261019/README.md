# 第二条新速度人工示范复核

- 来源：`corrections/human_1792404996662425932`，280控制步，metadata SHA256：`dbdff3dff26502de44ae77d3da06e27a1b512ebf2d54e3e951415677c4841b03`。
- 草稿 `human_1792404881458601498`（560步）排除，原件不改。
- 全280帧通过 correction_data.load_frames 校验，末10步 approval_checks 全通过。
- 人工检查47个双视角关键帧，包含每个动作段起终点。一次闭爪抓取后搬运、放置、松爪、撤手；起初存在小幅横向往返。保留为完整任务候选，不自动选入训练。
- 使用 scripts/review_corrections.py 的 render_pair，逐帧生成20FPS回放；全部解码，7个样本与原图比较，MAE<8。操作等待省略；非模型评测或物理重放。
- 移动统计使用相邻新鲜 pre-state；post_On/末尾观测门不能替代额外50步稳定性测试。
- Git备份范围：复查结果、视频、图集、原元数据清单。原始NPZ保留本地，不含于本次远端备份。
