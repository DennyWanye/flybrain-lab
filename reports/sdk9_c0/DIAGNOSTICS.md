# 三模型诊断与失败清单

部署模式为 argmax；采样结果仅作探索诊断。故障表独立于 nominal 封存成功率。

| 种子 | 近距离固定动作 | 近距离概率采样 |
|---|---:|---:|
| 11 | 40/40 | 40/40 |
| 22 | 40/40 | 40/40 |
| 33 | 40/40 | 40/40 |

## 故障诊断

定位丢失或通信状态不明时，保护终止是可接受的安全行为，但不能计作导航成功。噪声和延迟下的低成功率也不会隐藏。

| 种子 | 故障条件 | 成功 | 终止原因计数 |
|---|---|---:|---|
| 11 | pose-noise | 8/8 | success: 8 |
| 11 | pose-delay | 7/8 | success: 7, task_deadline: 1 |
| 11 | pose-mild | 5/8 | localization_lost: 1, success: 5, task_deadline: 2 |
| 11 | pose-lost | 0/8 | localization_lost: 8 |
| 11 | reply-lost | 0/8 | command_unknown: 8 |
| 11 | request-lost | 0/8 | command_unknown: 8 |
| 11 | channel-delay | 8/8 | success: 8 |
| 22 | pose-noise | 6/8 | success: 6, task_deadline: 2 |
| 22 | pose-delay | 7/8 | success: 7, task_deadline: 1 |
| 22 | pose-mild | 7/8 | localization_lost: 1, success: 7 |
| 22 | pose-lost | 0/8 | localization_lost: 8 |
| 22 | reply-lost | 0/8 | command_unknown: 8 |
| 22 | request-lost | 0/8 | command_unknown: 8 |
| 22 | channel-delay | 6/8 | success: 6, task_deadline: 2 |
| 33 | pose-noise | 6/8 | success: 6, task_deadline: 2 |
| 33 | pose-delay | 8/8 | success: 8 |
| 33 | pose-mild | 6/8 | localization_lost: 1, success: 6, task_deadline: 1 |
| 33 | pose-lost | 0/8 | localization_lost: 8 |
| 33 | reply-lost | 0/8 | command_unknown: 8 |
| 33 | request-lost | 0/8 | command_unknown: 8 |
| 33 | channel-delay | 8/8 | success: 8 |

## 全部开发与封存失败

| 种子 | 数据集 | case | 原因 | 终止距离 m |
|---|---|---|---|---:|
| 11 | validation | c0-validation-009 | task_deadline | 0.0203 |
| 11 | validation | c0-validation-048 | task_deadline | 0.0511 |
| 11 | sealed | c0-sealed-074 | task_deadline | 0.0921 |
| 11 | sealed | c0-sealed-212 | task_deadline | 0.1483 |
| 22 | validation | c0-validation-098 | task_deadline | 0.3933 |
| 22 | sealed | c0-sealed-037 | task_deadline | 0.3156 |
| 22 | sealed | c0-sealed-056 | task_deadline | 0.0485 |
| 22 | sealed | c0-sealed-074 | task_deadline | 0.1247 |
| 22 | sealed | c0-sealed-163 | task_deadline | 0.1444 |
| 22 | sealed | c0-sealed-214 | task_deadline | 0.1424 |
| 22 | sealed | c0-sealed-223 | task_deadline | 0.0953 |
| 22 | sealed | c0-sealed-251 | task_deadline | 0.4581 |
| 22 | sealed | c0-sealed-260 | task_deadline | 0.2000 |
| 33 | validation | c0-validation-001 | task_deadline | 0.0832 |
| 33 | validation | c0-validation-015 | task_deadline | 0.3041 |
| 33 | validation | c0-validation-058 | task_deadline | 0.1793 |
| 33 | sealed | c0-sealed-020 | task_deadline | 0.1366 |
| 33 | sealed | c0-sealed-031 | task_deadline | 0.3149 |
| 33 | sealed | c0-sealed-040 | task_deadline | 0.2889 |
| 33 | sealed | c0-sealed-076 | task_deadline | 0.2880 |
| 33 | sealed | c0-sealed-109 | task_deadline | 0.2494 |
| 33 | sealed | c0-sealed-129 | task_deadline | 0.1994 |
| 33 | sealed | c0-sealed-141 | task_deadline | 0.4079 |
| 33 | sealed | c0-sealed-179 | task_deadline | 0.1514 |

失败后未对这些最终权重继续训练；所有正式结果留存。模型 11 的开发失败复现细节另见 FAILURES.md。
