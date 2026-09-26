# Golden Episode 测试结果

日期：2026-09-26
状态：BLOCKED

## 当前记录

- 记录器：
- checkpoint：
- policy source：checkpoint deterministic policy
- episode：
- replay：
- steps：120
- success：false
- final distance：约 1.2458m
- collision：false
- out_of_bounds：false
- neural capture：

## Gate 结论



原因：当前 smoke checkpoint 没有产生成功 episode；同时该 checkpoint 不是 MaleCNS/Connectome brain policy checkpoint，神经活动字段尚未记录。根据 Handoff 合同，不使用脚本动作替代，也不伪造成功结果。

已验证：策略 checkpoint 产生动作，动作经过 TelloSim SDK adapter，仿真连续推进，replay 与 checkpoint SHA256 已保存。
