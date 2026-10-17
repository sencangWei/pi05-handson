from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from openpi.shared import normalize

root = Path(__file__).resolve().parents[1]
dataset = root / "data/lerobot/course/libero_black_bowl"
horizon = 10

stats = {
    "state": normalize.RunningStats(),
    "actions": normalize.RunningStats(),
}
frames = 0

for path in sorted(dataset.glob("data/**/*.parquet")):
    table = pq.read_table(path, columns=["state", "actions"])
    states = np.asarray(table["state"].to_pylist(), dtype=np.float32)
    actions = np.asarray(table["actions"].to_pylist(), dtype=np.float32)

    indices = np.minimum(
        np.arange(len(actions))[:, None] + np.arange(horizon),
        len(actions) - 1,
    )

    stats["state"].update(states)
    stats["actions"].update(actions[indices])
    frames += len(states)

assert frames == 1971, f"训练帧数不符合预期：{frames}"

output = root / "assets/pi05_handson_lora/course/libero_black_bowl"
normalize.save(output, {
    key: value.get_statistics() for key, value in stats.items()
})

print("统计的训练帧数:", frames)
print("动作窗口长度:", horizon)
print("统计文件:", output / "norm_stats.json")
