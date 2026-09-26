# Golden Episode 当前失败项

1. GE-01 未通过：当前 checkpoint policy episode 未进入目标半径并稳定保持。
2. GE-14 未通过：当前 checkpoint 没有有效 MaleCNS/Connectome graph 与 mapping contract；神经活动保持 `not_recorded`。
3. 正式 recorder/validator 尚未在当前 WSL 解释器执行：系统 Python 缺少 torch、gymnasium、mujoco 依赖。

已补齐的验收基础：GE-02/03/04/05/06/08/09/10/11/12/13/15 的记录字段和机器检查已实现。GE-07 的 replay seek 一致性仍需在浏览器验收环境中执行，不能由静态文件检查替代。

最终 Gate 仍保持：`GOLDEN_EPISODE_READY = NO`，`MODEL_READY_FOR_NEXT_STAGE = NO`。
