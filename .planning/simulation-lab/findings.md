# 接口与决策

MuJoCo launch_passive + LivePreview 保留原生相机、SPACE/N/Q；现有 before_step 在暂停时阻塞，不适合 Tk 主循环，面板用 after 定时检查 paused/single_step 并 sync。

LIBERO env.reset 可能替换 MjSim，需要先 close viewer 再重建环境/viewer。位置修改只允许新场景且开爪，保留目标 Z 与朝向、清零速度、刷新观测。reset/apply 自动封存上一轮已有动作。

原 eval 在首个 On 后继续40步，最大220步；新面板同一默认，success仍只是On，不自动宣称完整收尾。

新增独立 simulation_lab.py/lab_panel.py。后台线程仅做 WebSocket 推理，主 Tk 线程持 viewer lock 修改状态与执行动作；一次仅一个在途请求，暂停/重置使代次失效，重置前先保存。原生 SPACE/N 旧单步意图遗留已通过专门序列回归封住。

只读 LiberoTask 元数据读取成功：50 布局，BDDL/初始状态包哈希已核对。Tk9 仅暴露核心 X 字体，强行选 Noto 会退化 fixed；选择实际可用 song ti 并加行间距，已视检 560x510 控制窗口截图。
