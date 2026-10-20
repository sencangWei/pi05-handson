# Demo 录制按钮（2026-10-20）

用户要求：在中文面板添加录制 Demo、删除上一个 Demo；同时录制中文仿真面板和 MuJoCo 三维窗口。

## 使用

1. 加载布局，点击“录制 Demo”，再点击“运行”执行策略。录制不自动触发机器人运动。
2. 同一按钮变为“停止并保存 Demo”，点击保存 MP4。结束场景、切换布局或退出时也会收尾保存。
3. 视频位于项目 demos/demo-时间戳/demo.mp4，附 demo.json 和 encoder.log。面板显示保存位置。
4. “删除上一个 Demo”只处理最近一个本功能保存成功的 Demo，移到 .local/demo-trash/，可找回。录制时按钮禁用。原评测和人工示范不删除。

这是按实际时间、15FPS录制的屏幕区域，包含暂停时间；没有音频，不用来生成训练动作标签。两个窗口重叠时会并排摆放，使用桌面工作区排除系统左侧 Dock。开始后区域固定，保持窗口位置，避免其它窗口遮挡或最小化。依赖本机已有 ffmpeg、ffprobe、X11/python-xlib，没有安装新依赖。

## 修改与验证

- 新增 scripts/demo_recording.py；接入 lab_workbench.py / lab_panel.py；demos/ 加入忽略列表。
- simulation_lab.py、OpenPI、模型权重和训练动作逻辑未修改。
- 120 tests passed、5 subtests passed；Ruff PASS。目标：Demo 生命周期和回收边界、工作台、Tk面板、响应循环、人工接管。
- 合成双窗口实录生成 H264 MP4；实际软件也通过按钮完成录制、保存、删除，录制视频2466×1046、2.867秒。删除后的文件SHA256一致。对应 capture-smoke.json / live-ui-smoke.json。
- 测试时仿真保持暂停、0个控制动作；不是模型任务效果 Demo。测试短片在回收目录保留。
- 普通关闭并重开面板，精确恢复用户原布局0及X/Y偏移，原模型服务PID4048231保持运行；ui-verification.json提供记录。
- 截图 live-panel-and-simulation.png 为实际录制帧，buttons-ready.png 为最终面板。

备份范围：上述代码、测试和本证据目录。用户 Demo 默认不进 Git；本次仅将已检查的应用窗口截图纳入代码仓库。
