# Golden Episode 当前失败项

Golden Episode 验收项 GE-01 到 GE-15 当前全部通过。

## 尚未完成的更高层目标

1. `MODEL_READY_FOR_NEXT_STAGE` 仍为 `NO`：目前只有固定 Golden 场景封存结果，没有正式多场景 success-rate evaluation。
2. 当前 imitation policy 的训练集包含固定 Golden 场景和少量随机场景，不能把单条成功轨迹解释为整体泛化能力。
3. 浏览器页面的 replay 专用注册视图仍需将 `artifacts/golden_episode/replay.jsonl` 接入 viewer registry 后做人工 UI 截图验收；原始 replay 和机器校验已完成。

## 当前 Gate

`GOLDEN_EPISODE_READY = YES`

`MODEL_READY_FOR_NEXT_STAGE = NO`
