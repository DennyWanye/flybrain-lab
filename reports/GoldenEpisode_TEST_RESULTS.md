# Golden Episode 测试结果

日期：2026-09-26
状态：BLOCKED（实现已补齐，正式运行待依赖环境）

## 当前记录

- 记录器：`python -m flydrone.tellosim golden-record`
- 验证器：`python -m flydrone.tellosim golden-validate`
- policy source：checkpoint deterministic policy
- 当前封存 replay：`artifacts/golden_episode/replay.jsonl`
- 当前 checkpoint 结果：`success=false`，最终距离约 `1.2458m`
- collision：`false`
- out_of_bounds：`false`
- neural capture：`not_recorded`

## 本轮完成

- 每个 decision step 同时记录 world、observation、policy、command、reward、brain 状态。
- policy probabilities、selected action、value、entropy 已记录。
- SDK command 的 issued/started/completed sim tick、command id 和执行结果已记录。
- reward components 求和可验证，replay 默认只读取原始记录。
- 新增 GE-01 至 GE-15 机器验证入口；GE-14 在没有 brain graph 时明确失败。
- 垂直动作统一经 TelloSim adapter 执行，避免绕过 command contract。

## Gate 结论

`GOLDEN_EPISODE_READY = NO`

原因仍是当前 smoke checkpoint 没有产生成功 episode，且它不是 MaleCNS/Connectome brain policy checkpoint。根据 Handoff 合同，不使用脚本动作替代，也不伪造成功结果。当前 WSL 的 `/usr/bin/python3` 没有项目运行依赖（torch、gymnasium、mujoco），因此本轮只完成语法和静态一致性检查，正式 recorder/validator 运行需要在已安装项目依赖的环境执行。
