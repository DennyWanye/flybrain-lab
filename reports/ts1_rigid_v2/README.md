# 六自由度模拟与 C0 交付

生成时间：2026-09-27T03:25:52.887833+00:00

新六自由度环境的三种子 C0 封存门槛已通过。本轮每个模型新增 672 个训练动作；只训练读出，MaleCNS 连接图保持冻结。完整 TS1 仍有未完成验收项，C1/C2 与真机未开放。

| 模型种子 | 旧权重在新环境 | 适配后开发验证 | 新封存测试 | 碰撞/越界 |
|---|---:|---:|---:|---:|
| 11 | 93/100 | 94/100 | 284/300 | 0 |
| 22 | 89/100 | 92/100 | 276/300 | 0 |
| 33 | 85/100 | 96/100 | 279/300 | 0 |

随机基线：5/300。每个模型按同一 300 场景验收；成功率至少 90%，碰撞/越界至多 1%，比随机基线高至少 20 个百分点。

`MODEL_READY_FOR_NEXT_STAGE = YES`（仅 nominal 模拟 C0）；`FULL_TS1_READY = NO`；`REAL_FLIGHT_READY = NO`。

本轮运行的 80 项软件回归全部通过，物理规则基线 100/100；完整图的三次保存/恢复重复对照通过。原 TS1 的 60 条验收要求另见 TEST_MATRIX.md，不能用 80 个 pytest 用例数代替它。

## 怎么看

打开 http://127.0.0.1:8765/tellosim ，点“新环境模型 11”；22、33 是同一冻结果蝇图上分别训练的读出。点物理脚本验证可看起飞、平移、真实转向与降落。标“旧环境”的记录保留原来的历史含义。

## 保存与继续

最终模型与精确训练边界在 `runs/tellosim-sdk9/rigid-final-s11/`、`rigid-final-s22/`、`rigid-final-s33/` 对应目录。已完成本轮预算的 resume.pt 不会偷偷增加训练预算；新课程应显式新建热启动运行。精确恢复测试用 fp64-boundary.pt（16/32 动作）与连续完成的产物逐项比较。

```bash
python -m flydrone.tellosim.training.continuation --project-root . --graph data/male-v1.npz --out runs/tellosim-sdk9/new-pilot --init-from runs/tellosim-sdk9/rigid-final-s11/checkpoint.pt --batch 4 --options 128 --rollout 32 --seed 11 --wall-seconds 300
```

运行中断后，用同一个 out，移除 --init-from，改为 --resume <out>/resume.pt，并保持 options/rollout/batch/seed/wall-seconds 不变。仅适用于相同源码、依赖、配置和设备合同；不支持旧 CSR/COO 诊断文件静默迁移。

## 仍未完成的原计划部分

C1/C2、系统化动力学扰动、每环境独立的策略随机数流、四 lane 实时切换、长时性能与多浏览器压力等，详见 TEST_MATRIX.md。下一阶段先处理这些门槛中影响训练正确性的条目，再扩展课程。

本包不包含虚拟环境和大型 MaleCNS 图；外部依赖 `data/male-v1.npz` 的 SHA256 为 badc33a247894fe12c4d68d3ff791393857cffa39ff2ab81869608a11fa1e0c9。纯回放不依赖图文件。见 IMPLEMENTATION.md、FAILURES_AND_FIXES.md、BROWSER_CHECK.md 及逐 case JSON。
