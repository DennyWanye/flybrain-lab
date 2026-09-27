# 封存失败案例与复现

最终候选冻结后不再针对这些案例训练或调参。以下均保留在分母内。

| 种子 | 失败原因 | 数量 | 首个失败 case |
|---|---|---:|---|
| 11 | task_deadline | 16 | rigid-sealed-004 |
| 22 | task_deadline | 24 | rigid-sealed-035 |
| 33 | task_deadline | 21 | rigid-sealed-007 |

task_deadline 表示 60 秒内未达到目标稳定停留条件，不是撞墙。每个失败的动作序列、最终距离、奖励和时长见 FAILED_CASES.json。当前 C0 达标不等于所有目标都可靠完成。

示例复现（保持源码/设备/依赖一致，导出实际模型回放）：

```bash
python -m reports.ts1_rigid_v2.replay_case --seed 11 --case-id rigid-sealed-004
```

输出带时间戳，不覆盖原封存结果；复现输出不得回写正式分母或挑选成绩。
