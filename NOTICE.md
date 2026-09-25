# 来源与授权说明

本工具包的自定义代码用于工程实验，不是原论文代码的数值复现，也不声称拥有第三方连接组数据。自定义代码按随附 Apache License 2.0 提供，不附带性能或适用性保证。

`scripts/body_demo.py` 与 `scripts/body_gpu_smoke.py` 是基于 FlyGym 官方公开示例 API 编写、增加 CLI/检查/结果保存后的适配脚本。FlyGym/NeuroMechFly 由 EPFL Neuroengineering Laboratory 及其贡献者开发，项目采用 Apache-2.0；引用时请使用项目指定的学术文献。原代码与本版本不是同一文件。

- 项目：https://github.com/NeLy-EPFL/flygym
- 文档：https://neuromechfly.org/
- 上游授权：https://github.com/NeLy-EPFL/flygym/blob/main/LICENSE

`scripts/download_male.py` 的数据文件名与期望 SHA256 来源于 alex titonis 的 fly.ai 固定版本；本工具包没有包含其连接组二进制文件。使用下载的数据时同时标明 MaleCNS 原始研究数据和 fly.ai 预处理来源。

- MaleCNS 数据：https://male-cns.janelia.org/download/
- 社区预处理：https://github.com/alextitonis/fly.ai
- 固定代码与数据哈希：`sources.lock.json`

`scripts/reference.py` 调用 Philip Shiu 等人的公开原始模型，而不是替换该模型；原始项目另行下载、保留自己的授权与引用要求。

- 原模型：https://github.com/philshiu/Drosophila_brain_model
- 原论文：https://www.nature.com/articles/s41586-024-07763-9

发布论文、产品或衍生数据前，请另行核对各数据集、模型、网格资源与依赖的当前许可。本文件不将第三方资源重新授权为本工具包的许可。
