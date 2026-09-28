# FlyBrain Lab v0.1 — 4×DGX Spark

先阅读同目录《4DGX_Spark_FlyBrainLab_逐步部署与训练指南.md》。该文档包含全部安装命令、四机配置、科学参考与身体仿真分支，以及与本目录一致的源码附录。

训练主线：MaleCNS/FlyWire稀疏连接图 → 冻结工程LIF reservoir → PPO训练小型策略/价值读出。不是全脑突触反向传播，不是生物大脑完整数字孪生。

主要入口为 `flylab.py --help`。运行容器命令用 `bash scripts/run_core.sh ...`；本机软件自测用 `bash scripts/validate_local.sh`。

执行顺序：宿主检查 → ARM64镜像/GPU稀疏算子 → 合成图软件自测 → 校验真实数据 → 单机全量测试/训练 → 对照实验 → 四机Gloo。不要先改动现有Ring或卸载GPU驱动。

已完成与未完成的验证见 `reports/VALIDATION.md`。本次交付的测试使用x86_64 CPU和合成图，并未在用户Spark上运行；不能用随附CPU小图基准推算全量GPU表现。

资源和授权来源见 `sources.lock.json`、`NOTICE.md`、`LICENSE`。包内不包含上游大脑数据、Docker大镜像或训练好的真实果蝇策略。
