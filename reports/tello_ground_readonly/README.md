# RoboMaster TT 地面只读与任务卡 Shadow 交接

更新时间：2026-10-09（Asia/Shanghai）

## 已完成

- `tello_ground_readonly/`：TT/Tello UDP 地面采集器。
- 命令白名单：`command`、`sdk?`、`battery?`、`hardware?`、`time?`、`speed?`。
- 任务卡识别配置只允许显式的 `mon`、`mdirection 0`；不包含移动、转向、`rc`、`stop` 或 `emergency`。
- 原始状态包保留 ASCII、十六进制和时间戳。
- 任务卡 `mid/x/y/z` 保留为任务卡相对坐标，不冒充房间全局坐标。
- Shadow V1 只记录建议，不发送建议动作。

## 真实证据

`reports/tello_ground_readonly/20261009-pad-no-led/`：拆除 LED 点阵后的地面采集，`mid=-1`。

`reports/tello_ground_readonly/20261009-controlled-hover/`：用户授权的一次受控测试。仅发送 `takeoff`、悬停读取、`land`，无水平移动、转向或 `rc`。最高高度约 60 cm，随后识别到 `mid=3` 持续 65 个状态包；额外发送 `land` 3 次后连续 `h=0` 确认落地。

`reports/tello_ground_readonly/20261008-shadow/`：50 条真实地面状态的 C2W Shadow 审计输入、决策日志和结果。50/50 被阻塞，`transmitted_commands=0`。

## 另一台电脑准备

1. 克隆本仓库并运行：

   `python -m unittest discover -s tests -p "test_tello_ground_readonly.py"`

2. 运行地面只读采集：

   `python -m tello_ground_readonly collect --output <目录> --duration-s 5`

3. C2W Shadow 仍需要本地准备交接中已有的 MaleCNS 图和 `spatial-continuous-s11/checkpoint.pt`；这些大文件不放入 Git。没有它们时只能运行采集器和协议测试，不能声称完成 C2W 推理。

4. `REAL_FLIGHT_READY` 保持 `false`。任务卡识别成功只证明 TT 能报告任务卡 ID，不等于已获得房间全局位姿，也不等于模型动作已执行。
