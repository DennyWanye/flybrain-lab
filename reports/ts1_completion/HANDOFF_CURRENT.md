# 2026-09-28 当前交付状态与继续入口

实际工作区：Ubuntu-24.04:/home/denny/projects/flybrain_lab_4spark_v0_1，Windows入口D:\projects\果蝇训练。Python:/home/denny/projects/flybrain-env/bin/python。起点提交4668d90698c277af24191f12f90e3ab0c6cd1632；本轮没有提交Git。不要重装环境、重建房间或重跑全部历史实验。

R1已完成：V1R种子11/22/33验证均100/100，封存298/300、300/300、300/300，边界均12/12，指令均4/4，零特征均0/12，碰撞0，原J2R12例逐场不变。报告reports/ts1_altitude_refined，权重runs/tellosim-sdk9/altitude-refined-s*/checkpoint.pt。

R2仍未通过。最新完成正式训练与评估的是C2Q（spatial_orientation）。每种子5632动作，三种读出确实更新，完整MaleCNS仍冻结。种子11/22/33验证97/100、97/100、97/100；封存287/300、290/300、292/300；边界10/12、12/12、12/12；指令均8/8；零特征均0/12；碰撞均0。规则98/100低于99，11号边界10/12低于11。MODEL_READY_FOR_NEXT_STAGE和JOINT_3D_TASK_VERIFIED均NO。权重runs/tellosim-sdk9/spatial-orientation-s*/checkpoint.pt；报告reports/ts1_spatial_orientation。不能重用这批封存调参或选权重。

后续开发候选全部保留且未正式评估：C2S(spatial_settled)规则开发119/128；C2T(spatial_handoff)小开发12/16；C2U(spatial_stationary)小开发16/16，但全开发进度88/90已有2次失败，已不可能满足预注册127/128，因此终止准确进程。S/U仅把C2Q权重逐张量不变迁移到新合同，没有新增训练；T未生成checkpoint。U神经开发尚未启动。不要将这些候选说成已通过模型。

开发证据：reports/ts1_rule_development（独立420000000..127，原规则122/128）；reports/ts1_phase_development、2、3（各候选独立保存的开发对照与轨迹）；各候选DEVELOPMENT_FAILURE.json、日志和process-cleanup.json。开发问题是噪声边界与扰动响应的权衡：长窗口可改善静止高度交接，但会延迟真实漂移恢复；原始测量又会反复打断近10cm边界的阶段保持。只改滤波未得到完整开发通过。下一步应基于开发数据重新查明控制与估计误差，不预设继续增加平滑窗口；新训练/正式评估必须另开版本、目录和划分。全部主体代码仍由当前主代理编写，不授权子代理代码或plan-test。

R3—R7软件证据已具备：取消/关闭观察不等于设备STOP；27项动力学参数/饱和/外力检查；普通MLP与V1R11同60场景均60/60（单种子pilot、训练数据不同、不能证明连接组优越）；真实MaleCNS轨迹与梯度更新对0/30/60FPS及断开消费等价；浏览器播放/暂停/seek/平移/断开重连；实测256MiB配额和forward500长命令分块。证据集中reports/ts1_completion与reports/ts1_mlp_pilot。

软件完整回归146项通过是在C2Q主体和真实640动作8/8小验证之后执行。候选S/T/U之后仅有各自12/13/12项针对性检查，未再运行大规模回归，亦未把开发门槛失败当通过。最终交付包携带候选作为失败研究记录，默认Viewer展示最新正式C2Q。

R8：60项原TS1软件矩阵保留原55项适用证据、补齐5项，交付包T60须以PACKAGE_AUDIT.json实际完成为准。即使软件包验收通过，FULL_TS1_READY/SIMULATION_GOAL_READY仍NO，因为R2尚未通过。

所有训练/评估/诊断子进程已回收；FINAL_PROCESS_AUDIT.json列明51个已消失PID，跨多阶段累计采样私有内存约34.59GiB（不是同时释放量）。保留Viewer PID25983，http://127.0.0.1:8765/tellosim 。交付包验证会另起短期加载/HTTP进程并逐个回收，见PACKAGE_AUDIT。

真实定位、实机标定、适配器与shadow验证、受控真机测试四阶段均未完成；REAL_FLIGHT_READY=NO。没有修改真实无人机适配器开放状态。
