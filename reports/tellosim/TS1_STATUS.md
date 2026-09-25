# TS1 TelloSim 状态

## 已完成

- SDK 命令解析、范围校验与 operation ACK 状态机
- MuJoCo headless surrogate world
- 起飞、平移、旋转、停止、落地脚本路线
- 有界速度、房间范围、有限值检查
- PoseProvider 统一位姿快照接口
- 26 通道 observation 接口与 schema manifest
- 9 动作 SDK 子集与变时长动作环境 TelloSimEnv
- 独立 PPO smoke runner，已完成 16 env steps
- Viewer 回放注册：tellosim-demo

## 当前边界

- 这是脚本驱动的 MuJoCo surrogate，不是真实 Tello 数字孪生。
- 未连接真实 UDP/Tello。
- smoke runner 只验证训练接口和梯度更新，不代表果蝇训练成功。
- 26 通道中的保留位需要在后续神经控制器阶段填充真实传感器语义。

## TS1 完成标准

SDK、物理 surrogate、统一 observation/action contract、变时长环境、PPO smoke、记录回放和限制文档均已具备。后续可进入正式训练配置与神经控制器集成，不应将当前结果解释为真实飞行或生物学成功证据。
