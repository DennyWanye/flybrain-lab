# Shadow V1：定位输入、记录回放与无发送推理

本版本是 TS1 C2W 之后的独立软件增量。原 C2W 源码、权重、模拟结果、交付包不变。它提供真实定位的输入边界，尚未实现某一种相机/动捕/定位标记的定位算法，也没有连接无人机。

## 当前能力与边界

- 接收规范化 JSONL：三维位置、三维速度、yaw、SDK 高度与电量、时间戳、任务上下文。
- 使用既有 SensorObservation 构造 26 通道，阶段编码 34 通道，完整冻结 MaleCNS 产生 128 特征，C2W 三个读出给出建议。没有原始误差绕过神经模型的动作规则。
- 三个阶段由**上游记录的 context.phase**提供。goal_m 是该阶段的目标/锚点，必须由上游任务管理器提供。本版本不声称已有独立真实任务管理器，也不根据建议动作假设移动完成。
- previous_action、previous_duration_s、operation_status 必须记录实际受监督任务的状态。Shadow 建议不会写回它们，busy/unknown/error 时不提供可执行建议。
- 没有网络 transport、设备对象、发送回调或 enable-flight 开关。STOP 也仅能成为建议文本。RealDeviceDisabled 继续拒绝实例化。
- 此时 REAL_FLIGHT_READY 恒为 false；hardware_validated 恒为 false。接口检查和合成输入神经小跑不能替代真实定位与飞行验收。

## 输入合同

使用 `python -m flydrone.shadow example --output 新目录` 生成 24 条明确标记 synthetic_fixture 的接口示例。它不是物理轨迹，也不是闭环任务成功数据。

每行 schema 为 flybrain.shadow.sample/1，字段必须精确匹配：

| 对象 | 字段 | 约束 |
|---|---|---|
| 顶层 | schema, seq, time_ns, pose, state, context | seq 连续；time_ns 连续相差 100000000 ns；不补帧、不加速神经时间 |
| pose | captured_ns, received_ns, valid, frame_id, position_m, velocity_mps, yaw_rad | 米、米/秒、弧度；速度在源地图坐标系；valid=true 时全部必要测量存在 |
| state | captured_ns, received_ns, valid, height_m, battery_fraction, airborne | 高度米，电量 0..1；airborne 为上游运行状态判定，不能冒充 SDK 原生字段 |
| context | phase, goal_m, target_yaw_rad, previous_action, previous_duration_s, remaining_s, operation_status | phase 为 altitude/navigation/heading；goal_m 和目标 yaw 在 room_map；阶段剩余 0..60 秒 |

所有时间必须来自同一已同步单调时钟域，满足 captured <= received <= time_ns。采集器应把设备时间换算到该域，并在 10 Hz 调度时显式输出 invalid 帧表示丢失；缺失 seq 或时钟跳变会停止本次会话。重复设备捕获时间允许表达暂时无新数据，但年龄持续增加并触发失效。重复消费相同 seq 不推进神经时钟。

Profile 要求 source_id、provenance、source_frame。provenance 必须为 synthetic_fixture、simulated_observation 或 real_recording；这些是来源声明，real_recording 标签本身不证明硬件达标。

仅支持两个右手、Z 向上地图之间的 yaw 旋转和 XYZ 平移：`position_room = Rz(yaw_to_room_rad) * position_source + translation_m`；速度只旋转，yaw 加上旋转角。默认单位是 SI。NED、反射坐标、毫米和角度数据必须先由采集器转换，不能仅改 frame_id。SDK height_m 与定位 Z 可能有不同零点，需独立标定。不能拿 h/tof 填充 XY，不能把未标定的 SDK 速度当全局速度。

本版开发默认新鲜度预算 max_age_s=0.2，时钟不确定性 clock_uncertainty_s=0.02。年龄加不确定性超预算即失效；不确定性不得超过 20 ms，max_age 不得超过既有模型 0.5 秒边界。它们是接口开发预算，**尚非实机验证门槛**。无效/过期、未飞行、动作忙/未知/错误、阶段超时均抑制所有建议。格式错误、时钟不连续、神经步数错误会中止并保留失败证据。

## 使用现有 WSL 环境

以下命令在 `/home/denny/projects/flybrain_lab_4spark_v0_1` 执行，使用 `/home/denny/projects/flybrain-env/bin/python`。输出必须是未存在的新目录。

```bash
python -m flydrone.shadow example --output reports/shadow-demo-input
python -m flydrone.shadow record --input reports/shadow-demo-input/samples.jsonl --profile reports/shadow-demo-input/profile.json --output reports/shadow-demo-record
python -m flydrone.shadow verify --recording reports/shadow-demo-record
python -m flydrone.shadow replay --root . --seed 11 --recording reports/shadow-demo-record --output reports/shadow-demo-replay
python -m flydrone.shadow verify-shadow --session reports/shadow-demo-replay/shadow
```

`shadow --input 文件 --profile 文件 --root . --output 新目录` 同时记录和推理，按文件时间离线运行。`--seed` 仅允许 11/22/33；不选择最好模型、不训练。

`shadow --input - ...` 接受上游进程提供的规范化实时 stdin。等待 stderr 的 SHADOW_INPUT_READY 后再开始提供样本。此模式要求 time_ns 使用本机 time.monotonic_ns() 时钟域；额外校验真实接收时间，积压和未来时间会停止。默认断流 1 秒退出，可用 --idle-timeout-s 在 (0,10] 内配置。其线程仅读取 stdin，不打开无人机端口。过慢推理造成过期也会拒绝，不悄悄延长模型时钟。实际 10 Hz 端到端性能仍需使用目标设备实测。

## 证据与故障保留

录制目录有 header.json、samples.jsonl、footer.json；样本按 SHA256 链接。完整校验通过才开始回放。中断、截断、内容变更、序号/时间不连续都不能通过校验。

推理输出有 input/ 原始规范化输入副本、shadow/session.json、decisions.jsonl、result.json；失败时保留 partial 文件及 failure.json。每个建议记录输入哈希、26 通道、mask、四步神经时钟、128 特征哈希、概率、checkpoint/graph 哈希、transmitted=false。verify-shadow 可在不加载神经图时检查哈希链、时钟、动作及结果一致性。哈希是完整性检查，不是外部签名或数据真实性认证。

历史 C2W 正式成绩继续只适用于原模拟合同。本版本没有重新统计成功率，不得把建议动作重放误称闭环完成任务，也不得把观测到的两秒稳定当 120 Hz 物理连续稳定的证明。

## 后续实机前工作

见 `reports/shadow_v1/HARDWARE_ACCEPTANCE.md`。需要先确定机型、定位源、设备坐标/时间同步方式和高度基准，再实现该设备的只读采集器并记录真实样本。通过定位与标定审查后才设计受监督硬件 Shadow；任何实际飞行仍需单独授权。
