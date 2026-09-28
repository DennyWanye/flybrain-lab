# SDK 子集兼容性

本表是本地模拟器能力，不是实机兼容性认证。

| 能力 | 本轮实现 | 证据/边界 |
|---|---|---|
| command/takeoff/land/stop | 共享连续物理执行器 | 与训练同一 VisualSession；起降由任务管理器负责 |
| forward/back/left/right/up/down/cw/ccw | 共享执行器 | SDK9 C0 只开放 STOP 和四个水平动作 |
| go x y z speed | 已补充 | 本机 FLU 解释；>20 cm 保守边界；速度在命令结束后恢复 |
| speed 与只读查询 | 已补充 | 数值和模拟身份显式标明；不伪造设备序列号 |
| request/reply drop/delay | 独立有界通道 | 不自动重发相对动作；未知操作阻塞新任务 |
| loopback UDP | 127.0.0.1 literal only | 单一客户端、无设备转发；默认端口 18889 |
| status UDP | 可选本机状态端口 | h/tof cm，vg* 按本项目 dm/s profile；不提供可靠 world XYZ |
| pitch/roll/温度/加速度 | 未提供 | 当前保持水平的控制替身不假造传感器精度 |
| 15s idle watchdog | loopback fixture | busy 长命令不被误算为空闲；只代表仿真定义 |
| mid/mission pads、视频、rc、emergency 等 | 不支持 | parser 拒绝 |
| 身体姿态力矩控制 | 未完成原 TS1 规格 | 平移用 MuJoCo 力积分；yaw 是受限积分状态，不能称完整刚体飞控 |
| 实机 SDK / 定位 / 标定 | NO | RealDeviceDisabled 保持禁用 |

新通道已接入训练设备接口，通过 case.channel_profile 显式启用。nominal C0 保持既有执行语义，避免偷偷改变已经训练的模型。定位噪声、延迟和丢失来自 ExternalPoseMock；观测包含有效位，不给模型隐藏位置真值。

协议测试入口：
```bash
python -m flydrone.tellosim.sdk.loopback_gateway --project-root . --host 127.0.0.1 --port 18889
```
退出 Ctrl-C 会关闭该测试实例。它不启动真实无人机连接。
