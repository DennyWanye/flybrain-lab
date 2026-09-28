# C2W 三维位置、指定朝向与连续保持

JOINT_3D_TASK_VERIFIED = YES
MODEL_READY_FOR_NEXT_STAGE = YES
REAL_FLIGHT_READY = NO

| seed | validation | sealed | boundary | instruction | zero | collision |
|---|---|---|---|---|---|---|
| 11 | 100/100 | 290/300 | 12/12 | 8/8 | 0/12 | 0 |
| 22 | 99/100 | 293/300 | 12/12 | 8/8 | 0/12 | 0 |
| 33 | 99/100 | 294/300 | 11/12 | 8/8 | 0/12 | 0 |

规则100/100；随机0/300。三组四类扰动均达到冻结原门槛。12份固定索引回放、全部正式结果的时钟/物理连续性/动作mask、源训练数据与正式划分零重叠均已审计。

本轮新增训练动作0；三组九个读出逐张量等于正式C2Q权重，源训练动作合计16896。完整166700神经元、25582938边的MaleCNS仍冻结；没有用规则替代神经模型、没有选择最好种子或按封存调参。

控制修复：新版本SDK执行器保留未被命令改变的保持目标轴；空闲STOP和yaw不累积位置扰动。运动中的STOP仍在当前位置制动。旧VisualSession及旧模型保留原语义，新执行器只用于C2W。

阶段管理只对静止高度测量取均值，神经策略仍使用原始26通道输入。最终成功逐120Hz物理tick判断连续2秒，10Hz观测和每次4神经子步未改变。

C2V保存为失败：采样smoke8/8中一场实际只有236/240个合格物理tick，故独立审计7/8并中断开发。C2W14项前置检查通过，新完整图smoke8/8且240/240 tick，旧失败完整神经复现修复，新规则开发128/128，神经开发31/32。

C2W不是100%无失败模型，所有正式失败结果保留。整体模拟交付状态见../ts1_completion_c2w/summary.json，不能仅用R2通过代替R8。

划分与来源见cases.json、ACTUAL_SEED_AUDIT.json及TRAINING_AND_EVALUATION_AUDIT.json。冻结protocol中development结束数字存在复制笔误，实际新范围550000000..550000127以独立manifest和PROTOCOL_NOTES.json为准，未覆盖原协议。
