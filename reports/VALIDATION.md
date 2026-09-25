# 本次交付的验证记录

验证日期：2026-09-18。运行环境为本次会话的 x86_64 CPU 容器，不是用户的 DGX Spark。

## 实际环境

- Python：3.13.5
- PyTorch：2.10.0+cpu（CPU）
- NumPy：2.3.5
- SciPy：1.17.0
- GPU：无；未连接用户的任何机器。

## 已执行并通过

1. Python 编译检查和全部 Shell 脚本 `bash -n`。
2. 13 项 pytest 单元测试：稀疏矩阵方向、超过 2^53 的神经元 ID 保真、MaleCNS 格式转换、行归一化、覆盖保护、状态重置、CSR/COO 一致性、对照组 I/O 映射、终止状态 GAE、迷宫连通性、梯度与 CPU 回退保护、Notebook 常量解析。
3. 512 神经元**合成连接图**的 CPU 稀疏乘法、reservoir 基准、3 次 PPO 更新、checkpoint 保存和重新加载评估。
4. 同一容器内 `torchrun --nproc-per-node=2` 的 Gloo all-reduce：两进程均返回求和 3。
5. 同机两进程同步 PPO：2 次更新；每次更新后检查所有进程参数 SHA256 相等。

最终完整入口脚本 `scripts/validate_local.sh` 也已在本机执行成功，完整输出在 `reports/full-software-validation.txt`。

分项原始日志在 `reports/pytest.txt`、`cpu-spmm.json`、`cpu-benchmark.json`、`smoke-train.jsonl`、`smoke-eval.json`、`local-gloo-check.txt`、`local-gloo-train.txt`。

## 本次测试发现并修改的问题

- 增益 3 在这个合成测试图上产生约 94% 的末步放电比例。因此默认工程增益降为 1，并加入活动率诊断。这个调整不代表对 MaleCNS 或真实果蝇生理的校准。
- 修正一个 Shell 必填参数提示中的引号语法问题，并重新通过全部 `bash -n`。
- 检查 PyTorch 设备选择接口后，将无索引的 `cuda` 显式解析为 `cuda:0`；新增 mock 单元测试验证选择逻辑。该 mock 不替代真实 GPU 运行。

## 没有执行的验证，不应宣称已通过

- ARM64 Docker 镜像构建与实际 GB10/SM121 CUDA 内核执行。
- fly.ai 两个完整数据文件的实际下载及全量 MaleCNS/FlyWire GPU 转换/训练。容器内的通用网络下载不可用；上游代码与链接通过网页和 GitHub 读取核验。
- 原论文 Brian2 环境安装与糖刺激/MN9 全量实验。参考脚本已对照上游函数签名编写，但未运行原始模型。
- FlyGym 2.1.0 安装、MuJoCo-Warp 执行、EGL 渲染和影片生成。
- 四台物理 Spark 的跨机路由、并行吞吐量、通信可靠性。
- 真实连接图优于随机图或直接策略的统计结论；代码没有预设这一结论。

## 如何解释日志

`smoke-eval.json` 只有 5 个 episode，且使用合成数据和很少的训练步数，**只能说明程序能完成评估，不证明学会任务**。

`cpu-benchmark.json` 使用 512 个合成神经元，其 steps/s 不可当作 16 万神经元全量连接图在 Spark 上的预测值，也不可当作 FlyGym 的物理吞吐量。

Shiu parquet 转换单元测试模拟了 `pandas.read_parquet` 返回值，验证了后续逻辑；它不代表已读取真实 parquet，也不验证 pyarrow 的 ARM64 二进制兼容性。MaleCNS 转换测试同样使用极小的本地 fixture，而非全量数据。

每台 Spark 应按主文档 G0—G5 阶段验收，再进行真正的四机训练。
