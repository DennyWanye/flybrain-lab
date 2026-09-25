# FlyBrain Lab

本项目是一个用于果蝇连接组驱动控制实验的本地研究工具包。主线将 MaleCNS 稀疏连接图接入冻结的工程 LIF reservoir，再用 PPO 训练小型读出策略，在离线 `FlyToTarget2D-v1` 环境中观察和评估行为。

## 项目边界

- 这是工程实验代码，不是果蝇全脑数字孪生，也不是全突触反向传播。
- 当前 P1 是二维离线仿真；不包含 Tello、UDP、起飞、降落或真实飞行控制。
- 浏览器 Viewer 是只读观察器，不会从网页控制训练或硬件。
- 连接组数据、模型 checkpoint 和运行结果不随源码发布，需要按各自许可单独获取或本地生成。

## 快速开始

1. 阅读 [`4DGX_Spark_FlyBrainLab_逐步部署与训练指南.md`](4DGX_Spark_FlyBrainLab_逐步部署与训练指南.md)。
2. 准备 Python/PyTorch/CUDA 环境并放置经过许可的数据文件。
3. 使用 `configs/` 中的配置运行 smoke 训练和评估。
4. 启动本地只读观察器：

```bash
PYTHONPATH=. python3 -m flyview serve --project-root . --host 127.0.0.1 --port 8765
```

然后打开 <http://127.0.0.1:8765>。

## 当前状态

- P0 软件环境：已完成。
- P1 软件链路：已完成，支持环境、reservoir、PPO、checkpoint、回放和浏览器观察器。
- P1 学习门槛：尚未通过三个独立 seed 的正式门槛，详见 [`reports/p1/P1_ACCEPTANCE.md`](reports/p1/P1_ACCEPTANCE.md)。
- 真实飞行：明确未实现。

## 许可与数据

自定义代码遵循仓库中的 [`LICENSE`](LICENSE)。第三方连接组数据、上游代码和模型资源遵循各自许可，详见 [`NOTICE.md`](NOTICE.md)。不要将受限数据或本地 checkpoint 提交到公开仓库。
