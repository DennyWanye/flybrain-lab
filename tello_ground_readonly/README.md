# TT 地面只读采集器

这个包只连接 RoboMaster TT/Tello 的地面观测边界。命令端口的硬白名单是
`command`（仅进入 SDK 模式）、`sdk?`、`battery?`、`hardware?`、`time?` 和
`speed?`；不包含运动、降落、停止、紧急、视频流或任务卡模式切换命令。

8890 状态包按原始十六进制、ASCII 和接收时间写入 `udp_raw.jsonl`。`mid/x/y/z`
仅被标为 `mission_pad_relative`，不会被转换为房间全局位姿。没有独立校准的
房间定位源时，Shadow V1 样本的 pose 永远无效，因此模型建议被阻塞，且
`transmitted=false`。

## 采集

```powershell
python -m tello_ground_readonly collect `
  --output .\tello-ground-readonly-YYYYMMDD `
  --duration-s 5 `
  --timeout-s 2
```

该命令不发送起飞、降落、移动、转向、`rc`、`stop` 或 `emergency`。设备不回某
个查询时仍写入带时间戳的 timeout 记录。不要为了打开任务卡模式追加 `mon` 或
`mdirection`，因为本阶段只读取设备已经提供的状态。

## Shadow 输入

```powershell
python -m tello_ground_readonly shadow-jsonl `
  --session .\tello-ground-readonly-YYYYMMDD `
  --output .\tello-ground-readonly-YYYYMMDD\shadow-samples.jsonl
```

同时生成 `shadow-profile.json`。它是 `real_recording` 但使用
`tello_unmapped` frame；在加入独立室内定位和标定前，Shadow 只能记录被阻塞的
建议，不能代表执行或真实飞行就绪。
