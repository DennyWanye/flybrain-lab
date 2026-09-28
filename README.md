# FlyBrain Lab — TS1 C2W + Shadow V1

当前版本完成三维位置、指定朝向和连续保持的模拟验收，并实现无发送的定位输入/记录回放/Shadow 推理。真实飞行未开放。

- [正式模拟验收](reports/ts1_completion_c2w/summary.json)：三组 C2W 模型通过；封存集分别 290/300、293/300、294/300。
- [Shadow V1 报告](reports/shadow_v1/FINAL_REPORT.md)：43 项针对性检查、三组完整神经图重复回放、断流/积压检查通过；输入为合成观测，真实硬件未验证。
- [训练权重、完整记录与下载说明](docs/TRAINED_ARTIFACTS.md)：包括历史与失败版本；代码和摘要在 Git，完整数据在同版本 GitHub Release。
- [Shadow 接口和运行命令](docs/shadow_v1.md)；[后续硬件验收清单](reports/shadow_v1/HARDWARE_ACCEPTANCE.md)。

C2Q 在冻结 MaleCNS 图上实际训练读出，每 seed 5632 个决策动作；C2W 权重逐张量不变迁移，新训练动作数为 0。三阶段读出不等于单一联合策略。模型成功只适用于已记录的模拟合同。

## 初始部署文档与历史范围

先阅读同目录《4DGX_Spark_FlyBrainLab_逐步部署与训练指南.md》。该文档包含全部安装命令、四机配置、科学参考与身体仿真分支，以及与本目录一致的源码附录。

训练主线：MaleCNS/FlyWire稀疏连接图 → 冻结工程LIF reservoir → PPO训练小型策略/价值读出。不是全脑突触反向传播，不是生物大脑完整数字孪生。

主要入口为 `flylab.py --help`。运行容器命令用 `bash scripts/run_core.sh ...`；本机软件自测用 `bash scripts/validate_local.sh`。

执行顺序：宿主检查 → ARM64镜像/GPU稀疏算子 → 合成图软件自测 → 校验真实数据 → 单机全量测试/训练 → 对照实验 → 四机Gloo。不要先改动现有Ring或卸载GPU驱动。

已完成与未完成的验证见 `reports/VALIDATION.md`。初始部署验证使用 x86_64 CPU 和合成图；后续 TS1 和 Shadow 使用完整 MaleCNS 图，实际证据见上方报告。未在用户 Spark 上运行的项目仍不得宣称通过。

资源和授权来源见 `sources.lock.json`、`NOTICE.md`、`LICENSE`。Git 源码不包含大图和权重二进制；GitHub Release 包含训练读出与精确处理后的图及其来源说明。Docker 镜像和运行环境不在交付中。
