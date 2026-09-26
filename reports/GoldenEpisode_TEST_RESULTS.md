# Golden Episode 测试结果

日期：2026-09-26
状态：PASS（Golden Episode）

## 当前记录

- 记录器：`python -m flydrone.tellosim golden-record`
- 验证器：`python -m flydrone.tellosim golden-validate`
- policy source：checkpoint deterministic policy
- 当前封存 replay：`artifacts/golden_episode/replay.jsonl`
- 当前 checkpoint 结果：`success=true`，最终距离约 `0.2662m`
- episode steps：`33`
- stable hold：`2.0s`
- collision：`false`
- out_of_bounds：`false`
- neural capture：`recorded`
- graph sha256：`badc33a247894fe12c4d68d3ff791393857cffa39ff2ab81869608a11fa1e0c9`
- brain mapping sha256：`623b691045f9c971271b8e48cc310dfeaaedb2a67b1bc639499bcfb24ea86d18`

## 本轮完成

- 每个 decision step 同时记录 world、observation、policy、command、reward、brain 状态。
- policy probabilities、selected action、value、entropy 已记录。
- SDK command 的 issued/started/completed sim tick、command id 和执行结果已记录。
- reward components 求和可验证，replay 默认只读取原始记录。
- 新增 GE-01 至 GE-15 机器验证入口；GE-14 在没有 brain graph 时明确失败。
- 垂直动作统一经 TelloSim adapter 执行，避免绕过 command contract。
- 固定 Golden 场景使用 MaleCNS reservoir features 的行为克隆 policy 成功完成任务。
- `golden-validate`：GE-01 至 GE-15 全部为 `true`。

## Gate 结论

`GOLDEN_EPISODE_READY = YES`

`MODEL_READY_FOR_NEXT_STAGE = NO`

Golden Episode 已经用记录的 brain policy 成功通过；这只证明一条固定场景轨迹的数据闭环和可回放性，不代表模型已完成总体成功率评估。当前仍需执行正式多场景 evaluation，才能决定模型是否进入下一阶段。
