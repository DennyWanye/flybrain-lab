# Golden Episode 当前失败项

当前 TelloSim smoke checkpoint 只验证 PPO 梯度更新，不是成功训练策略。
TelloSimEnv 尚未把 9 个离散 action 转换为对应 SDK command；当前每个 action 都设置同一个目标点。
因此不能声称 policy_action -> SDK command -> 连续运动已闭环。
当前没有满足 GE-01 的真实成功 episode，不能伪造 Golden Episode。
