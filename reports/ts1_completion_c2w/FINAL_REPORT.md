# R2 + R8 当前模拟阶段完成报告（2026-09-28）

软件验收、三维联合模型、当前模拟总目标、FULL_TS1_READY均为YES；REAL_FLIGHT_READY为NO。本轮未发送任何真实无人机命令。

| seed | 验证 | 封存 | 边界 | 指令 | 零特征 | 碰撞/越界 |
|---|---|---|---|---|---|---|
|11|100/100|290/300|12/12|8/8|0/12|0|
|22|99/100|293/300|12/12|8/8|0/12|0|
|33|99/100|294/300|11/12|8/8|0/12|0|

规则100/100，随机0/300；四扰动组均满足原门槛。模型不是100%无失败控制器，全部失败结果保留。

本轮修复保持目标漂移与120Hz连续保持计时；C2W使用正式训练C2Q的三组权重，逐张量不变，新训练动作0，源训练16896动作。完整MaleCNS仍冻结。C2V的236/240 tick小跑失败和部分开发结果保留。

主体实现、针对性检查、完整神经小跑先通过，再完成开发与三组正式评估，最后才运行211项软件回归；全部通过。最新真实浏览器回放、时间轴、帧率、视角与最终就绪状态已核验。60项矩阵按实际影响保留历史适用证据，不宣称重跑所有旧学习或压力实验。

交付包：FlyBrain_TS1_V1R_C2W_20260928.zip
SHA256：151026687f6aad962931730d19d6f06b37035cea9aebb2254c3363998e9a97d7

包内15328文件；全新目录解压、逐文件hash、九读出完整图推理、六回放、三HTTP路由及探针运行后再次hash通过。使用现有WSL依赖，不是全新机器安装测试。包旁PACKAGE_AUDIT.json记录ZIP摘要；evidence/保留发布后截图与最终进程核验。这些发布后证据不反写已冻结ZIP。

实际源码：/home/denny/projects/flybrain_lab_4spark_v0_1，当前工作区文件清单和哈希定义交付版本；未把旧Git HEAD说成包含本轮未提交代码。

R2：reports/ts1_spatial_continuous/README.md、summary.json、TRAINING_AND_EVALUATION_AUDIT.json、FORMAL_REPLAY_AUDIT.json。
R8：reports/ts1_completion_c2w/summary.json、TEST_MATRIX.json、IMPACT_AND_RETENTION.json、PACKAGE_AUDIT.json。
当前全局就绪状态以R8 summary为准；R2 summary中的FULL_TS1_READY=false是R8打包前的阶段快照，未篡改原始评估报告。

本轮记录的18个工作进程已确认退出；采样私有内存跨阶段合计约17.04GiB（不是同一时刻释放量）。仅保留Viewer PID33503。访问 http://127.0.0.1:8765/tellosim 。无遗留训练、评估或打包GPU计算进程。

全部主体由当前主代理完成，未使用子代理编码或plan-test，未重装环境或重建房间。旧源码、权重、失败结果与交付包保留。

后续真实定位、标定、真实适配器与shadow、受控真机飞行属于另外阶段；本次完成不构成真实飞行授权。
