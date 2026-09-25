# 4×DGX Spark 果蝇连接组实验平台：逐步部署、训练与验收指南

**版本：FlyBrain Lab v0.1｜核验日期：2026-09-18｜交付形式：可独立使用的 Markdown + 同内容代码包**

适用目标：你已经有 4 台 NVIDIA DGX Spark，希望在本地跑真实果蝇连接组，并训练它与小游戏环境交互。主线使用 **MaleCNS 连接图 + 冻结的稀疏 LIF reservoir + 可训练 PPO 输出层**；同时提供 **FlyWire/原论文 Brian2 参考路线** 和 **FlyGym 2.1.0 身体仿真路线**。

> **先说明本次完成到哪一层。** 文中全部自定义 Python/Shell 文件都在附录，可以从本 Markdown 一次性提取，不需要寻找一个并不存在的“FlyBrainOS”仓库。已在当前 x86_64 CPU 环境执行 13 项单元测试、单进程训练/评估、同机双进程 Gloo 同步训练。**没有访问你的 Spark，也没有完成 ARM64 容器构建、全量生物数据 GPU 运行或四台物理机器实测。** 这些环节安排了明确的验收命令；不能将源代码核验当作硬件实测。

---

## 0. 先把目标定清楚

### 0.1 这份方案真正训练什么

```text
迷宫的工程化传感器（目标相对位置、附近墙壁、剩余时间）
                    ↓ 固定输入映射
        真实连接图上的简化脉冲网络
           W 冻结；每个环境独立状态
                    ↓ 读取膜电位和脉冲迹线
           小型策略网络 + 价值网络
                    ↓ PPO 更新这两个输出头
            上 / 右 / 下 / 左
                    ↓
                  迷宫
```

**不是对 16 万个生物神经元的全部突触进行反向传播。** 学习发生在输出网络，连接图本身不变。准确名称是“使用果蝇连接组作为 reservoir 的强化学习实验”。它是否比随机图或直接 MLP 更有价值，要靠对照实验决定。

神经元连接图也不等于一只活果蝇的完整动态大脑。神经递质符号、感受野、细胞动力学、神经调质、身体反馈等需要额外建模。特别是将游戏数值直接投射到某些神经元，是**工程输入映射**，不是复现果蝇真实视觉系统。[1][2][3]

### 0.2 对上一轮构想的必要校正

| 上一轮容易造成误解的说法 | 本次实际采用的做法 |
|---|---|
| 一个统一 `Brain.load()` API 已经能切换全部生态 | 本文提供自己的明确数据格式和完整转换代码；不是冒充上游现成 API。 |
| 4 台机器能直接跑数百/数千个全脑个体 | 不承诺并发数字。从单机 batch=1/4/8 测量，再扩展。身体仿真并发不等于全脑仿真并发。 |
| 原论文默认就是 FlyWire v783 | 原始仓库默认论文实验使用 v630；v783 是另一套数据配置，不混用神经元 ID。[1] |
| FlyWire 某个公开总体数量就是加载后的节点数 | 以实际转换输出的 N 和 nnz 为准。不同筛选版本、脑/CNS范围、神经元对边/突触数量不可混算。 |
| 连接组一接身体就形成完整生物闭环 | 本版分别验收脑、学习、身体；身体示例由手写高层信号驱动，不虚称由已训练大脑控制。 |
| Brian2 不适合 GPU，所以一定要抛弃 | 本版用 Brian2 做原模型参考；另写简化 GPU 主线。没有对各种 Brian2 后端做普遍优劣结论。 |

### 0.3 三条路线与先后顺序

**建议执行顺序：A0 软件自测 → A1 MaleCNS 主线 → A2 单机训练 → A3 四机；B 原论文参考、C 身体仿真单独执行。** B、C 不阻塞 A，避免一次性调试三个不同环境。

| 路线 | 数据/框架 | 目的 | 本文是否提供代码 |
|---|---|---|---|
| A：训练主线 | MaleCNS 预处理图或 FlyWire 图、PyTorch、PPO、Gloo | 真实连接图参与游戏决策；冻结图、训练输出层 | 是：数据下载、转换、稀疏脑、迷宫、训练、评估、四机启动 |
| B：科学参考 | Shiu 原仓库、FlyWire v630、Brian2 2.5.1 | 运行糖刺激→MN9 读出，与作者工作对齐 | 是：固定版本下载、隔离环境、实验包装脚本 |
| C：身体仿真 | FlyGym 2.1.0、MuJoCo、MuJoCo-Warp | 先确认果蝇身体、转向、GPU物理可用 | 是：独立安装、转向影片、GPU物理 smoke |

PPO 在本版中由 `flylab.py` 直接实现，不额外安装 Ray、TorchRL、Stable-Baselines3、Redis 或 Kubernetes。这是减少第一版依赖，不是要求你安装后再自己补上核心训练代码。

---

## 1. 四台机器怎么分工，为什么不用 TP4

### 1.1 推荐的最终部署

```text
Spark 0                         Spark 1 / 2 / 3
┌────────────────────┐          ┌────────────────────┐
│ 本地完整连接组副本    │          │ 本地完整连接组副本    │
│ GPU：稀疏神经状态更新 │          │ GPU：稀疏神经状态更新 │
│ CPU：迷宫 + PPO读出  │          │ CPU：迷宫 + PPO读出  │
│ 写日志/保存checkpoint│          │ 各自生成不同轨迹      │
└──────────┬─────────┘          └──────────┬─────────┘
           └──── Gloo/TCP：平均小策略网络梯度 ──┘
```

每台机器一个训练进程；每个进程可以有多个环境。**四机都进行训练更新，不是只有主节点训练。** 每一步优化先平均各机 CPU 梯度，再使用相同优化器更新；代码会检查各 rank 更新后的参数哈希一致。[4]

这样不传输每一个神经元每一时刻的 spike，也不把图按 TP4 切开。每台 128GB 的内存是本机资源，4 台相加不是一个自动共享的 512GB 地址空间。策略只有小型两层 MLP，第一版用 TCP/Gloo 比从 RDMA、NCCL 和环形路由开始更容易排错。

### 1.2 你现有的直连 Ring 可以保留，但别把物理连通当成 IP 全互通

四条线连接成环，并不自动保证任意两台的 IP 都能互相到达。Gloo 需要各 rank 之间建立连接；**只让所有节点 ping 通主节点还不够**。[4]

第一版优先使用已经能让四台相互 SSH/访问的私有管理网。已有高速环网若已完成路由并通过全互通测试，也能使用。本文**不会覆盖 netplan、重配 RDMA、修改 MTU、清理你现有大模型容器**。

尚未有可用全互通网络时，先在四台各跑独立实验/种子；这仍能使用全部机器，不用等同步通信调通。同步路线见第 11 节。

### 1.3 容量和性能的正确预期

本版 CSR 用每条边一个 FP32 权重和一个 int64 列索引。仅这部分理论容量近似：

```text
edge_storage ≈ nnz × (4 + 8) bytes + (N + 1) × 8 bytes
state_storage ≈ N × batch × 4 bytes × 状态数组数量
```

以社区 MaleCNS 预处理声明的约 2,558 万条神经元对连接估算，前者约 0.31GB（十进制），**不包含转换、元数据、临时张量、CUDA上下文等开销**。[2] 容量够不代表仿真快；此实现的 `sparse.mm` 仍遍历稀疏矩阵存储的连接，不是只计算发放事件的专用 CUDA 引擎。

第一轮使用 **FP32 + CSR + batch=1/4/8**。不启用 FP4/FP8、AMP 或 `torch.compile`；这些不是把大模型量化参数搬过来就能可靠加速神经仿真。[5]

---

## 2. 文件拿到后怎么放：ZIP 和纯 Markdown 两种方式

下文默认项目位于：

```bash
$HOME/projects/flybrain_lab_4spark_v0_1
```

所有未注明的命令都在 **Spark 宿主机终端**执行。`bash scripts/run_core.sh ...` 会把后面的命令放进 GPU 容器执行，路径按项目根目录解析。无需进入容器后再手动 `cd`。

### 2.1 方式一：解压随附 ZIP

把 ZIP 传到第一台 Spark，例如放在 `~/Downloads/`。在这台执行：

```bash
mkdir -p "$HOME/projects"
python3 -m zipfile -e \
  "$HOME/Downloads/FlyBrainLab_4Spark_代码与测试包.zip" \
  "$HOME/projects"
cd "$HOME/projects/flybrain_lab_4spark_v0_1"
chmod +x scripts/*.sh
```

如果下载器改了文件名，第一条路径改为实际名称。压缩包内已经有顶层目录，**不需要再套一层同名目录**。

### 2.2 方式二：仅凭本 Markdown 提取全部源码

本文件末尾每个文件都有 `FILE` 标记。把本 MD 保存到本机后，执行下面的 Python；脚本只写指定新目录，拒绝路径穿越和覆盖既有文件：

```bash
python3 - \
  "$HOME/Downloads/4DGX_Spark_FlyBrainLab_逐步部署与训练指南.md" \
  "$HOME/projects/flybrain_lab_4spark_v0_1" <<'PY'
from pathlib import Path
import re
import sys

source = Path(sys.argv[1]).read_text(encoding="utf-8")
root = Path(sys.argv[2]).expanduser().resolve()
pattern = r'^<!-- FILE: ([^\n]+) -->\n```[^\n]*\n(.*?)\n```[ \t]*$'
files = re.findall(pattern, source, flags=re.M | re.S)
if not files:
    raise SystemExit("没有找到文件块：请使用完整文档，而不是聊天预览摘录。")
plans = []
for relative, text in files:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise SystemExit(f"拒绝不安全路径：{relative}")
    target = (root / rel).resolve()
    if not target.is_relative_to(root) or target.exists():
        raise SystemExit(f"拒绝覆盖或越界：{target}")
    plans.append((target, text))
for target, text in plans:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text + "\n", encoding="utf-8")
    if target.suffix == ".sh":
        target.chmod(0o755)
print(f"已写入 {len(plans)} 个文件：{root}")
PY

cd "$HOME/projects/flybrain_lab_4spark_v0_1"
```

### 2.3 提取后应有的关键目录

```text
flybrain_lab_4spark_v0_1/
├── flylab.py                    # 全部核心：转换/脑/迷宫/PPO/评估/分布式
├── docker/
│   ├── Dockerfile.core          # GPU训练主线
│   ├── Dockerfile.reference     # 原论文Brian2隔离环境
│   └── Dockerfile.body          # FlyGym隔离环境
├── configs/node.env.example     # 每台主机的通信配置模板
├── scripts/
│   ├── preflight.sh
│   ├── run_core.sh
│   ├── download_male.py
│   ├── fetch_shiu.sh
│   ├── launch_rank.sh
│   ├── reference.py
│   ├── run_body.sh
│   ├── body_demo.py
│   ├── body_gpu_smoke.py
│   ├── validate_local.sh
│   └── summarize.py
├── tests/test_flylab.py
├── reports/VALIDATION.md        # 本次交付实际测试范围
├── sources.lock.json            # 上游固定commit/数据hash
├── NOTICE.md
└── LICENSE
```

下载的大脑数据、训练结果和容器镜像不在 ZIP 内。这样不把尚未下载验证的几百 MB 数据伪装成已随包交付，也便于你只在一台下载一次后同步。

---

## 3. G0：检查已有环境，不破坏现有部署

### 3.1 四台都执行只读检查

```bash
cd "$HOME/projects/flybrain_lab_4spark_v0_1"
bash scripts/preflight.sh
```

关注：`uname -m` 应为 Spark 的 `aarch64`；`nvidia-smi` 能看到 GPU；Docker 能列出容器；磁盘有空间；内存未被现有 LLM 占满。**此处不应出现 `x86_64` 的宿主机架构**，否则你正在另一台电脑上操作。

为避免镜像、两套数据、缓存和结果挤占空间，可以先按“空闲 100GB”作为保守规划目标；它是本方案的空间预算建议，不是项目官方最小需求。实际下载体积用 `docker system df` 和 `du -sh` 测量。

### 3.2 只补通用命令行工具

```bash
sudo apt-get update
sudo apt-get install -y git curl ca-certificates rsync jq tmux python3
```

不执行 `apt full-upgrade`，不卸载 CUDA/驱动，不清理 Docker volumes。不需要把 Spark 改装成别的 Linux 发行版。

### 3.3 Docker 已经可用：直接跳到第 4 节

```bash
docker version
docker info --format '{{json .Runtimes}}'
```

本指南后续假设你的当前账号已经能运行 Docker。Docker 组权限很高；没有权限时沿用你原来受控的管理方式，不要为了教程把 Docker socket 开放到网络。

### 3.4 只有全新机器缺少 Docker 时，才执行这一分支

下面使用 Docker 官方 Ubuntu 软件源步骤。[6] 如果机器已有 `docker.io`、containerd 或正在服务的 Docker，**不要直接走“删除冲突包、重装”路线**；先保留现有安装并排查。

```bash
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
sudo systemctl status docker --no-pager
sudo docker run --rm hello-world
```

### 3.5 只有缺少 NVIDIA Container Toolkit 时，才执行这一分支

先看是否已安装：

```bash
command -v nvidia-ctk || true
dpkg-query -W nvidia-container-toolkit 2>/dev/null || true
```

已有 GPU 容器可以正常运行时，不执行下面的重配置。需要补装的干净主机可使用官方软件源：[7]

```bash
sudo apt-get install -y gnupg2
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

仅在能够中断现有 Docker 服务的维护窗口执行以下配置；`nvidia-ctk` 会修改 Docker 的 daemon 配置，重启可能影响现有任务。[7]

```bash
if [ -f /etc/docker/daemon.json ]; then
  sudo cp -a /etc/docker/daemon.json "/etc/docker/daemon.json.before-flylab-$(date +%s)"
fi
sudo nvidia-ctk runtime configure --runtime=docker
# 先确认没有需要保持运行的任务，再执行：
sudo systemctl restart docker
```

**不要因容器驱动不兼容就运行 `apt purge nvidia*`。** 优先用 DGX 官方维护方式检查宿主驱动与目标容器兼容性；保留原环境。

---

## 4. G1：安装 GPU 主框架，先验证 GB10 稀疏计算

### 4.1 本版选择的基线，不冒充“最新版”

使用 `nvcr.io/nvidia/pytorch:25.11-py3` 作为**明确可追踪的候选基线**，不是声称它是 2026 年 9 月最新容器。NVIDIA 该版本文档标明 Ubuntu 24.04、Python 3.12、CUDA 13.0.2 和其 PyTorch 2.10 开发构建，并有 pip 约束与已知问题说明。[8]

本版保留容器自带 Torch/NumPy/Numba，不在宿主机 `pip install torch`，不拉 x86 CUDA wheel。必须以实际 `gpu-check` 结果判断你这台 Spark 的驱动与稀疏算子是否可用。

### 4.2 第一台拉取、记录镜像摘要

```bash
cd "$HOME/projects/flybrain_lab_4spark_v0_1"
mkdir -p reports data runs cache tmp upstream

# manifest 中检查 linux/arm64；这里不是在 Mac 上构建 amd64 再拿去运行。
docker manifest inspect nvcr.io/nvidia/pytorch:25.11-py3 > reports/base-manifest.json

docker pull --platform linux/arm64 nvcr.io/nvidia/pytorch:25.11-py3
BASE_IMAGE=$(docker image inspect \
  --format '{{index .RepoDigests 0}}' nvcr.io/nvidia/pytorch:25.11-py3)
printf '%s\n' "$BASE_IMAGE" | tee reports/base-image-digest.txt

docker image inspect --format '{{.Architecture}} {{.Os}}' \
  nvcr.io/nvidia/pytorch:25.11-py3
```

应输出 `arm64 linux`。镜像 digest 必须从你实际拉取的结果记录，本文没有编造一个 `sha256:...`。

### 4.3 构建核心镜像

```bash
docker build --platform linux/arm64 \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  -f docker/Dockerfile.core -t flybrain-lab:0.1 .

docker image inspect --format '{{.Id}}' flybrain-lab:0.1 \
  | tee reports/core-image-id.txt

docker run --rm flybrain-lab:0.1 cat /opt/flylab/pip-freeze.txt \
  > reports/core-pip-freeze.txt
```

若某个版本范围与 NGC 的 `/etc/pip/constraint.txt` 冲突，**先停下来检查，不要删除整个约束文件或强制升级 Torch/Numba**。查看：

```bash
docker run --rm nvcr.io/nvidia/pytorch:25.11-py3 bash -lc \
  'echo "PIP_CONSTRAINT=$PIP_CONSTRAINT"; grep -Ei "^(numpy|scipy|pandas|pyarrow|pytest|numba|torch)" /etc/pip/constraint.txt || true'
```

核心 Dockerfile 对 pandas/pyarrow/pytest 给出了兼容范围，但完整 ARM64 构建不在本次实测范围；本地构建成功后以 freeze 和 image ID 作为你的真正安装锁定记录。后面分发这个已经构建的镜像，不让其余三台各自重新解析依赖。

### 4.4 GPU 验证：不能只看 nvidia-smi

```bash
bash scripts/run_core.sh python flylab.py gpu-check --device cuda \
  | tee reports/spark-gpu-check.json
```

输出包括 Torch、CUDA runtime、GPU 名称和计算能力，最后必须有：

```json
"csr_spmm": "PASS"
```

这个命令真的运行了 GPU CSR 稀疏矩阵乘法，并和明确的期望值比较。`torch.cuda.is_available() == True` 但 CSR 报错，仍然不算通过。[5]

以下情况均停止全量训练：`no kernel image`、`CUDA driver insufficient`、`illegal instruction`、`cusparse` 算子失败、被迫装 `amd64`。不要让程序悄悄回落 CPU 后再误判 Spark 性能差；代码显式拒绝 CUDA 请求的静默回退。

---

## 5. G2：不下载大脑也能先验证安装和程序

### 5.1 执行完整软件自测

```bash
bash scripts/run_core.sh bash scripts/validate_local.sh
```

它会在新的 `runs/local-validation-时间戳/` 中生成合成图、执行 13 项单元测试、训练 3 次更新、重新加载评估、运行同机两个 CPU 进程的通信和同步训练。

这里特意使用 **512 个合成神经元**。目的是提前找到语法、Torch、状态重置、PPO、checkpoint 和 Gloo 软件问题，避免把下载大图后的错误混在一起。**这个结果不能叫“真实果蝇已跑通”。**

### 5.2 只测最小训练的手动命令

```bash
bash scripts/run_core.sh python flylab.py prepare \
  --source synthetic --neurons 512 --out data/synthetic.npz

bash scripts/run_core.sh python flylab.py train \
  --graph data/synthetic.npz --device cuda \
  --envs 4 --steps 16 --minibatch 32 --epochs 2 --updates 3 \
  --out runs/synthetic-gpu-smoke

bash scripts/run_core.sh python flylab.py eval \
  --graph data/synthetic.npz --device cuda \
  --checkpoint runs/synthetic-gpu-smoke/policy.pt \
  --episodes 10 --out runs/synthetic-gpu-eval.json
```

第二组会在真正 GPU 上执行小图；它与上一节 CPU 软件自测是不同验收。再次执行时改 `--out`，不要覆盖旧运行目录。

---

## 6. G3：主线数据下载——MaleCNS，不必装 CuPy

### 6.1 本版使用哪个数据

主线默认使用 `fly.ai` 对 MaleCNS v1.0 的预处理文件，**不是把社区包与原始官方数据混为一谈**。该固定版本数据说明写明 166,700 个神经元和 25,582,938 条连接；实际加载数量仍由程序输出。[2]

两个文件来自社区的 `brain-v1` release。文件 SHA256 已从固定 commit 的 `flybrain/data.py` 核对，下载器会对新下载和既有文件都重新验证。[2]

| 文件 | 用途 | SHA256 |
|---|---|---|
| `brain.npz` | 神经元 ID、类型等元信息 | `cc9bd1ecd00bd703a6fa648bc6ad145c93c7c1ee53debdcc9ce0d1f4305e6aca` |
| `weights.npz` | 已签名、归一化的稀疏连接矩阵 | `c29919aa44069a271b1ee978abe05fa9bf6e45e4ba3e436e92b624ef1b5be40c` |

此处无需安装 `flybrain[gpu]`。上游 CUDA 后端用 CuPy；本版直接读取数据，然后走已验证的 PyTorch 稀疏接口，避免 CuPy 的 CUDA 版本与 NGC 容器产生额外组合问题。[2]

### 6.2 下载：只需在第一台执行

```bash
python3 scripts/download_male.py --out data/raw/male
(cd data/raw/male && sha256sum -c SHA256SUMS)
du -sh data/raw/male
```

默认会访问：

```text
https://github.com/alextitonis/fly.ai/releases/download/brain-v1/brain.npz
https://github.com/alextitonis/fly.ai/releases/download/brain-v1/weights.npz
```

GitHub release 资产访问失败时，先修复网络或从可访问网络下载这两个确切文件后传入同目录，再重跑校验。**不要绕过哈希检查或把第三方同名文件直接替换。** 官方 MaleCNS 原始数据下载入口见 [3]；从原始 feather 构建是另一条数据处理路线，不是把文件改名就能替代这两个资产。

### 6.3 转换为本工具包格式

```bash
bash scripts/run_core.sh python flylab.py prepare \
  --source male \
  --weights data/raw/male/weights.npz \
  --brain data/raw/male/brain.npz \
  --out data/male-v1.npz

bash scripts/run_core.sh python flylab.py inspect --graph data/male-v1.npz \
  | tee reports/male-inspect.json
```

结果文件：

```text
data/male-v1.npz        # 神经元ID + CSR图
data/male-v1.npz.json   # 来源、方向、数量、hash、归一化记录
```

转换遵守以下规则：

- 始终 `W[post, pre]`，不把连接方向转反；重复连接合并。
- 神经元 ID 按字符串保留，不转成浮点数；大整数 ID 超过浮点精确表示范围时会损坏映射。
- 每个 postsynaptic 行除以 `max(绝对权重和, 1)`；这是工程 reservoir 的归一化，不是原论文的 mV 电导权重。
- 记录输入文件和输出文件 SHA256；四台机器直接分发同一个输出文件，不要各自用不同依赖重新生成。

### 6.4 输入输出没有“假装成真实感觉器官”

主线从图中固定抽样输入神经元，再从它们的一跳下游选取输出神经元；输入和输出集合不重叠，减少直接读出输入造成的旁路。默认采样种子 `--graph-seed 64`，默认输出 128 个神经元，使用膜电位和脉冲迹线两组特征，合计 256 维。

**没有声称这些输入一定是视觉神经元，也没有声称这 128 个输出一定是下降神经元。** 要研究特定生物感觉/运动通路，需要基于注释定义新的映射并单独验证。当前任务优先保证训练链路清楚、可测试。

---

## 7. 可选数据路线：FlyWire v630 / v783

需要论文参考或希望比较两套连接组时，再下载原始仓库。仓库已经固定到可核对的 commit，不跟随浮动的 `main`。[1]

```bash
bash scripts/fetch_shiu.sh
```

脚本固定版本：

```text
91bdd1e7dcf193f3e7ca5a8933497fcef63b7960
```

### 7.1 v630 转换

```bash
bash scripts/run_core.sh python flylab.py prepare \
  --source shiu \
  --complete upstream/Drosophila_brain_model/2023_03_23_completeness_630_final.csv \
  --connectivity upstream/Drosophila_brain_model/2023_03_23_connectivity_630_final.parquet \
  --out data/flywire-v630.npz
```

### 7.2 v783 转换

```bash
bash scripts/run_core.sh python flylab.py prepare \
  --source shiu \
  --complete upstream/Drosophila_brain_model/Completeness_783.csv \
  --connectivity upstream/Drosophila_brain_model/Connectivity_783.parquet \
  --out data/flywire-v783.npz
```

**v783 的 neuron ID 不应直接使用 v630 的糖神经元清单。** 本版训练用的抽样映射不依赖那个生物清单，但后面的原论文糖刺激包装脚本明确只用 v630。[1]

转换后可以把所有训练命令中的 `--graph data/male-v1.npz` 换成对应图路径。策略的输入输出映射和图规模会变，**不能拿在另一张图上训练的 checkpoint 强行加载**；代码以图 SHA256 拒绝这种混用。

### 7.3 parquet/CSV 看起来不对时

```bash
ls -lh upstream/Drosophila_brain_model/*parquet
head -n 2 upstream/Drosophila_brain_model/Completeness_783.csv
```

若文件内容实际是 Git LFS pointer，而不是数据，才补装和拉取 LFS：

```bash
sudo apt-get install -y git-lfs
git -C upstream/Drosophila_brain_model lfs install --local
git -C upstream/Drosophila_brain_model lfs pull
```

不要将错误页、登录页或 pointer 交给 parquet 读取器。转换器严格要求 `Presynaptic_Index`、`Postsynaptic_Index`、`Excitatory x Connectivity` 这三列；名称不一致说明拿错格式，先核对来源。

---

## 8. G4：真实连接图的 GPU 基准与活动率验收

### 8.1 从 batch=1 开始

```bash
bash scripts/run_core.sh python flylab.py bench \
  --graph data/male-v1.npz --device cuda \
  --envs 1 --iterations 20 --gain 1.0 \
  --out reports/male-bench-b1.json
```

再测 4 和 8，不要直接写 256：

```bash
for B in 4 8; do
  bash scripts/run_core.sh python flylab.py bench \
    --graph data/male-v1.npz --device cuda \
    --envs "$B" --iterations 20 --gain 1.0 \
    --out "reports/male-bench-b${B}.json"
done
```

### 8.2 怎么判读结果

| 字段 | 含义 | 判断方式 |
|---|---|---|
| `aggregate_env_steps_per_second` | 全 batch 合计的环境步处理率，只含 reservoir/特征传输 | 不是单只果蝇实时倍率，不是 FlyGym 物理步，不是完整 PPO 吞吐。 |
| `mean_spike_fraction` | 测量窗口中每个内部步末的平均发放比例 | 观察是否长期接近零或高饱和；不是与生物放电率等价的 Hz。 |
| `saturation_warning` | 平均比例超过 0.5 的工程提醒 | 应暂停训练并调整；0.5 只是排错阈值，不是生物学标准。 |
| `reset_input_contrast_mean_abs` | 重置后两种不同输入所产生的读出差异 | brain 模式应有可辨非零差异；零值说明输入可能没有有效传到读出。 |
| `cuda_peak_allocated_bytes` | PyTorch 统计的 CUDA 峰值分配 | 不是系统总统一内存占用；还要看 `free -h`、`nvidia-smi`。 |

`gain=3` 是容易被从社区模型直接复制过来的数值，但它不对所有图和归一化都合适。本工具包在合成图测试中观察到这个值导致高饱和，默认改成 `gain=1`；**真实数据仍必须重新检查**。

### 8.3 输入不传播或高饱和：先做小网格，不要盲训

```bash
for GAIN in 0.25 0.5 1.0 1.5; do
  bash scripts/run_core.sh python flylab.py bench \
    --graph data/male-v1.npz --device cuda --envs 4 \
    --iterations 30 --gain "$GAIN" \
    --out "reports/male-gain-${GAIN}.json"
done
```

挑选不饱和、有限数值、对输入有响应的候选，再用独立验证种子比较训练效果。`--tonic` 也可调整，但每次只动少数变量并记录。调整参数不等于“纠正果蝇生理”；它是在调工程 reservoir。

读出差异只是必要检查，不能证明网络有益。噪声、时钟信号和编码器都可能让策略得分，因此后续仍做对照。

### 8.4 CSR 后端异常时的定位

先重跑 `gpu-check`。仅 CSR 失败、确认该 Torch/CUDA 组合的 COO 路径可用时，可尝试：

```bash
bash scripts/run_core.sh python flylab.py bench \
  --graph data/male-v1.npz --device cuda --layout coo \
  --envs 1 --iterations 5 --out reports/male-coo-test.json
```

COO 会改变存储和性能，不是保证更快的开关。CPU 单元测试验证了本代码 CSR/COO 数值一致性；**没有在你的 GPU 验证这两条路径**。改变 layout 后重新记录基准。

---

## 9. 第一轮真实训练：先完成一个可重复的迷宫实验

### 9.1 任务究竟是什么

地图是 7×7 网格，边界是墙，中间有一道留缺口的墙；每个 episode 从可通行格子随机抽取起点与目标。输入 9 维：目标的相对 x/y 正负分量 4 维、相邻四方向墙壁 4 维、剩余时间 1 维。

动作顺序：`0=上，1=右，2=下，3=左`。每步 `-0.01`，撞墙额外 `-0.02`，达到目标 `+1`。默认 64 步后本任务结束；剩余时间已作为状态输入，所以这里按**有限时域任务终止**处理，不把普通 Gym 的 time-limit truncation 随意当终止。

这是固定地图、不同起终点的工程测试，不是复杂视觉迷宫，也不是新地图泛化。目标位置已知，等于使用特权状态传感器；后续改成相机观察是新的任务。

### 9.2 单机真实数据 smoke

先取短运行验证数据流：

```bash
bash scripts/run_core.sh python flylab.py train \
  --graph data/male-v1.npz --device cuda \
  --envs 4 --steps 16 --minibatch 32 --epochs 2 --updates 3 \
  --gain 1.0 --seed 1 \
  --out runs/male-real-smoke
```

你应看到每轮 JSON 指标，并得到：

```text
runs/male-real-smoke/config.json
runs/male-real-smoke/metrics.jsonl
runs/male-real-smoke/policy.pt
```

终止 episode 尚未出现时，`success_rate` 和 `mean_return` 会是 `null`，不是训练崩了。不用把第一轮两个成功 episode 当成 100% 能力。

### 9.3 正式初始配置

```bash
bash scripts/run_core.sh python flylab.py train \
  --graph data/male-v1.npz --device cuda \
  --mode brain --graph-seed 64 --readout 128 \
  --envs 8 --steps 64 --minibatch 128 \
  --epochs 4 --updates 200 \
  --internal-steps 4 --gain 1.0 --tonic 0.14 \
  --lr 0.0003 --seed 1 \
  --out runs/male-brain-seed1
```

若基准选了其他 gain，这里必须保持一致。单机这组配置的环境交互预算为：

```text
8 envs × 64 rollout steps × 200 updates = 102,400 environment steps
```

内部 LIF 步数还要乘 `--internal-steps 4`，不要把两个步数概念混算。训练不保证在这个预算收敛；先比较学习曲线与对照，再决定增加预算。

### 9.4 监控、长期终端与停止

```bash
# 在另一个宿主机终端：
tail -f runs/male-brain-seed1/metrics.jsonl
nvidia-smi
free -h
```

可用 tmux 保持终端会话：

```bash
tmux new -s flylab
# 在 tmux 内执行训练命令；Ctrl-b 然后 d 分离。
# 重新连接：
tmux attach -t flylab
```

这是你本地命令保持运行，不是聊天助手在后台执行。普通单机训练可用 `Ctrl-C` 停止，保留最近一次完成的 checkpoint；不要 `docker kill $(docker ps -q)`，它会误伤其他模型服务。

### 9.5 评估与轨迹查看

```bash
bash scripts/run_core.sh python flylab.py eval \
  --graph data/male-v1.npz --device cuda \
  --checkpoint runs/male-brain-seed1/policy.pt \
  --episodes 200 --seed 900001 \
  --out runs/eval-brain-seed1.json

cat runs/eval-brain-seed1.json
head -n 80 runs/eval-brain-seed1.json.trace.txt
```

文本轨迹中 `#=墙，.=空地，F=代理，G=目标`。默认使用策略分布抽样；加入 `--greedy` 才变成取最大概率动作。比较各模型时保持动作选择方法一致。

评估会读取 checkpoint 中的图映射、gain、内部步数和 horizon，避免忘记手动同步参数。图文件仍必须由 `--graph` 提供并通过哈希校验。

结果中的 Wilson 区间是描述性的二项比例区间，不能替代多个独立训练种子的比较。相同固定地图上的 200 次采样，也不是 200 个独立训练实验。

### 9.6 继续训练：本版是 warm start，不伪装成精确断点续训

```bash
bash scripts/run_core.sh python flylab.py train \
  --graph data/male-v1.npz --device cuda \
  --mode brain --envs 8 --steps 64 --minibatch 128 \
  --updates 100 --gain 1.0 --seed 2 \
  --warm-start runs/male-brain-seed1/policy.pt \
  --out runs/male-brain-warmstart1
```

它恢复策略/价值网络权重，但重新初始化优化器、环境和随机轨迹；**不是严格恢复中断瞬间**。默认不保存全脑运行状态，避免不必要的复杂度。精确续训需要额外保存优化器、各 rank RNG、环境及 reservoir 状态，此版本不提供该承诺。

---

## 10. 必须做的对照：验证到底是谁在“学”

### 10.1 四个可训练模型和一个随机策略

| 模式 | 改动 | 能回答的问题 |
|---|---|---|
| `brain` | 真实图、固定 I/O、冻结 reservoir | 实际试验组。 |
| `shuffled` | 对图的 presynaptic 列做同一随机置换，I/O不变 | 解剖连接的具体身份是否比被打乱的连接更有用。不是严格逐节点度数守恒重连。 |
| `no_recurrence` | 不计算连接传播，其余时间更新与读出保持 | 能否仅靠时钟、初始态或环境规律得分；检查旁路。 |
| `direct` | 原始 9 维输入直接送入 MLP，无 reservoir | 一个普通小策略是否已经解决任务。输入维度与总参数量并非严格匹配。 |
| `--random` 评估 | 均匀随机动作，不训练 | 地图本身是否太容易，随机走是否也有较高成功率。 |

`shuffled` 会保持每个接收行的权重集合，但改变源神经元标签与解剖位置对应。它不是每个节点入度/出度和细胞类型都严格守恒的全面神经科学 null model。科研论文还需要更细的匹配对照。

### 10.2 顺序运行全部模式 × 3 个训练种子

先在较小数据或低预算试通，再执行完整预算：

```bash
for MODE in brain shuffled no_recurrence direct; do
  for SEED in 1 2 3; do
    bash scripts/run_core.sh python flylab.py train \
      --graph data/male-v1.npz --device cuda \
      --mode "$MODE" --graph-seed 64 --gain 1.0 \
      --envs 8 --steps 64 --minibatch 128 \
      --epochs 4 --updates 200 --seed "$SEED" \
      --out "runs/control-${MODE}-${SEED}"

    bash scripts/run_core.sh python flylab.py eval \
      --graph data/male-v1.npz --device cuda \
      --checkpoint "runs/control-${MODE}-${SEED}/policy.pt" \
      --episodes 200 --seed 900001 \
      --out "runs/eval-${MODE}-${SEED}.json"
  done
done

bash scripts/run_core.sh python flylab.py eval \
  --graph data/male-v1.npz --device cpu --random \
  --episodes 200 --seed 900001 --out runs/eval-random.json

python3 scripts/summarize.py 'runs/eval-*.json' \
  | tee reports/eval-summary.jsonl
```

所有评估使用相同起终点抽样种子，减少任务抽样差异；训练种子不同。按模式比较各训练种子的分数，而不是把 3×200 次 episode 当成 600 次独立模型训练。

**可能出现且必须如实接受的结果：直接 MLP 最好、真实图和随机图无差异、真实图根本没有学会。** 这些结果是实验发现，不应为了“果蝇大脑”叙事隐藏。此时先改善输入编码、动态工作区间和任务设计，而不是宣称只需要更多 DGX。

### 10.3 更稳妥地用满四台：各跑一个实验组

初次使用四台时，可以安排 Spark0=brain、Spark1=shuffled、Spark2=no_recurrence、Spark3=direct；各自循环三个 seed。此方法不依赖跨机 Gloo，只要数据/镜像一致即可。

这四个模式的计算开销不同，direct 很可能先结束。没有必要为了让 GPU 利用率看起来平均而人为增加它的训练步数；保持公平的环境交互预算更重要。

---

## 11. G5：四机同步训练，完整操作顺序

### 11.1 数据和镜像只下载/构建一次

先在你平常使用的 SSH 配置中定义 `fly1`、`fly2`、`fly3` 对应其余三台真实主机。它们只是本文的 SSH 别名，不是假定机器本来就叫这些名字。可编辑：

```bash
mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"
nano "$HOME/.ssh/config"
```

配置结构如下，把每台实际 IP 和登录用户名填进去：

```sshconfig
Host fly1
  HostName REPLACE_WITH_SPARK1_IP
  User REPLACE_WITH_USERNAME

Host fly2
  HostName REPLACE_WITH_SPARK2_IP
  User REPLACE_WITH_USERNAME

Host fly3
  HostName REPLACE_WITH_SPARK3_IP
  User REPLACE_WITH_USERNAME
```

若已有主机别名，直接用已有的，不覆盖文件。先验证：

```bash
ssh fly1 hostname
ssh fly2 hostname
ssh fly3 hostname
```

从第一台分发项目和图，排除结果及大缓存：

```bash
cd "$HOME/projects/flybrain_lab_4spark_v0_1"
for HOST in fly1 fly2 fly3; do
  ssh "$HOST" 'mkdir -p "$HOME/projects/flybrain_lab_4spark_v0_1"'
  rsync -a --info=progress2 \
    --exclude runs/ --exclude cache/ --exclude tmp/ --exclude upstream/ \
    --exclude __pycache__/ --exclude .pytest_cache/ \
    ./ "$HOST:projects/flybrain_lab_4spark_v0_1/"
done
```

不要加 `--delete`，以免删掉目标机已有本地配置和日志。如果不需要在三台运行原论文参考，`upstream/` 不用复制；训练主线只需要转换后的图。

导出已成功构建的核心镜像：

```bash
docker save flybrain-lab:0.1 | gzip -1 > "$HOME/flybrain-core-0.1.tar.gz"
(cd "$HOME" && sha256sum flybrain-core-0.1.tar.gz > flybrain-core-0.1.tar.gz.sha256)

for HOST in fly1 fly2 fly3; do
  rsync -av --progress "$HOME/flybrain-core-0.1.tar.gz" \
    "$HOME/flybrain-core-0.1.tar.gz.sha256" "$HOST:"
  ssh "$HOST" '(cd "$HOME" && sha256sum -c flybrain-core-0.1.tar.gz.sha256)'
  ssh "$HOST" 'gzip -dc "$HOME/flybrain-core-0.1.tar.gz" | docker load'
done
```

每台确认：

```bash
cd "$HOME/projects/flybrain_lab_4spark_v0_1"
docker image inspect --format '{{.Id}}' flybrain-lab:0.1
sha256sum flylab.py data/male-v1.npz
bash scripts/run_core.sh python flylab.py gpu-check --device cuda
```

四台 image ID、代码 hash、图 hash 应一致。Gloo 训练开始时还会校验配置和 Torch 版本；不把不一致的四个环境拼在一起。

### 11.2 每台填写自己的节点配置

```bash
cp configs/node.env.example configs/node.env
nano configs/node.env
```

四台必须一致的字段：

```text
FLY_NNODES=4
FLY_MASTER_ADDR=第0台实际私有IP
FLY_MASTER_PORT=29610
```

四台分别为 `FLY_NODE_RANK=0/1/2/3`。每台 `GLOO_SOCKET_IFNAME` 填**本机实际承担私有通信的接口名**，不同主机可能不同，不能照抄 `eth0`。

检查接口和去主机的路由：

```bash
source configs/node.env
ip -br address
ip route get "$FLY_MASTER_ADDR"
```

`node.env.example` 中的 `REPLACE_...` 必须替换。网络接口上有多段地址或策略路由时，尤其要确认对外宣布的地址可被所有其他节点到达。

### 11.3 必须验证四台之间互相可到达

在每台机器上对其他三台的实际 IP 执行：

```bash
ping -c 3 OTHER_NODE_PRIVATE_IP
ip route get OTHER_NODE_PRIVATE_IP
```

`OTHER_NODE_PRIVATE_IP` 是需替换的文字，不是可直接执行的合法地址。要验证 TCP，不要只看 ICMP。可在每台开一个只共享临时标记文件的测试服务：

```bash
mkdir -p tmp/probe
hostname > tmp/probe/node.txt
# NODE_LOCAL_IP 换成本台选定接口上的实际私有IP。
python3 -m http.server 29611 --bind NODE_LOCAL_IP --directory tmp/probe
```

从其他三台测试：

```bash
curl --fail --max-time 3 http://OTHER_NODE_PRIVATE_IP:29611/node.txt
```

每台都要能够访问另外三台，然后 `Ctrl-C` 停止测试服务。不要在公网绑定服务，不要把整个项目目录作为 HTTP 根目录公开。

Gloo 除 rendezvous 端口外还会建立 peer 连接，**仅放行 29610 不一定足够**。[4] 防火墙启用时按你现有安全规则，仅允许选定私有接口上来自其余三台明确 IP 的 TCP 通信。例如确认 `PEER_IP` 是其中一台后：

```bash
# 只在确实启用 UFW 且需要放行时使用，逐个指定三台 peer。
sudo ufw allow in on "$GLOO_SOCKET_IFNAME" proto tcp from PEER_IP
```

不要关闭整机防火墙、不要开放全网来源。如果直连环网缺失三层路由，不在训练脚本里临时“自动修复”；先走既有全互通管理网或独立实验路线，保留正在运行的大模型网络。

### 11.4 四台先分别启动通信测试

四台各开一个终端，分别执行相同命令，配置文件内 rank 各不相同：

```bash
cd "$HOME/projects/flybrain_lab_4spark_v0_1"
source configs/node.env
bash scripts/run_core.sh bash scripts/launch_rank.sh dist-check
```

四台都应输出：

```json
{"world_size": 4, "sum": 10.0, "status": "PASS"}
```

rank 字段分别为 0/1/2/3。这里只有 all-reduce，不加载大脑。卡住时先排 rank、IP、接口、防火墙，而不是修改 PPO 或神经网络参数。

### 11.5 四机真实图短训练

通信通过后，在四台终端执行：

```bash
source configs/node.env
bash scripts/run_core.sh bash scripts/launch_rank.sh train \
  --graph data/male-v1.npz --device cuda \
  --envs 2 --steps 16 --minibatch 16 --epochs 1 --updates 2 \
  --mode brain --gain 1.0 --seed 1 \
  --out runs/male-4node-smoke
```

只有 rank0 写 `metrics.jsonl` 和 `policy.pt`。每轮会校验所有 rank 的参数 SHA256 一致；不一致直接报错。这个检查成本对于小型输出层可接受。

### 11.6 四机正式训练

在每台以相同配置启动：

```bash
source configs/node.env
bash scripts/run_core.sh bash scripts/launch_rank.sh train \
  --graph data/male-v1.npz --device cuda \
  --mode brain --graph-seed 64 --readout 128 \
  --envs 8 --steps 64 --minibatch 128 \
  --epochs 4 --updates 200 --gain 1.0 --tonic 0.14 \
  --internal-steps 4 --lr 0.0003 --seed 1 \
  --out runs/male-4node-seed1
```

环境预算变成：

```text
4 ranks × 8 envs × 64 rollout steps × 200 updates = 409,600 environment steps
```

同样的 200 updates，四机比单机多收集四倍数据；**比较训练效率时必须报告全局 env steps、wall time 和成功率**，不能将多用四倍数据的结果直接解释成并行加速。比较固定环境预算时，可先用单机 200 updates 对四机 50 updates；但全局 minibatch 和优化统计仍不同，需要在结果中说明。

`minibatch` 是每 rank 的大小，所以四机一次同步优化使用的全局样本数是 4×128，而不是 128。第一版不承诺学习轨迹与单机逐位一致。

### 11.7 如何退出和恢复

四个 rank 是一个同步作业；任一 rank 退出，其他 rank 最终也会失败或超时。本版 `--max-restarts=0`，不做隐藏自动重启。停止时在四台对应终端结束本任务，避免留下进程。

恢复用第 9.6 节的 warm start，在四台分发相同 checkpoint 后重新启动新目录。不要把单个 rank 的环境状态视为整个集群可恢复状态。

---

## 12. B 路线：原论文糖刺激 → MN9 参考实验

这条路线的意义是把“工程训练代码能跑”与“能运行作者原始模型”分开验证。使用 `Drosophila_brain_model` 原版 `model.py`，不把前面归一化后的矩阵送进它。[1]

### 12.1 拉取固定原仓库

尚未做第 7 节时执行：

```bash
bash scripts/fetch_shiu.sh
```

### 12.2 构建单独的 CPU 参考镜像

```bash
docker build --platform linux/arm64 \
  -f docker/Dockerfile.reference -t flybrain-reference:0.1 .

docker run --rm flybrain-reference:0.1 \
  python -c 'import brian2,numpy,pandas,pyarrow; print(brian2.__version__,numpy.__version__)'
```

这套候选环境沿用原作者 Python 3.10/Brian2 2.5.1 的方向，使用独立 NumPy 1.24，**不接触核心 GPU 容器的 NumPy 2.x/NGC环境**。旧科学栈在 ARM64 上的安装仍需这一步实测，不能仅凭版本号就断言可用。

### 12.3 先执行 100ms、每个刺激频率 1 次的 smoke

```bash
mkdir -p runs/reference tmp/reference-home
docker run --rm --init \
  --user "$(id -u):$(id -g)" \
  -e HOME=/work/tmp/reference-home \
  -v "$PWD:/work" -w /work \
  flybrain-reference:0.1 \
  python scripts/reference.py \
    --seconds 0.1 --trials 1 --rates 0 100 150 \
    --codegen numpy --out runs/reference/smoke.csv

cat runs/reference/smoke.csv
```

脚本从 `example.ipynb` **仅解析常量赋值**，提取 `neu_sugar` 和 `id_mn9`，不执行整个 Notebook；再按 v630 完整性表的行序映射到模型索引。[1]

原模型公开的关键参数包括 20ms 膜时间常数、5ms 突触时间常数、1.8ms 突触延迟和 0.275mV 的单突触缩放，和 A 路线的简化更新**不是同一个动力学系统**。[1] 本包装脚本不改变原模型方程；结果中记录输入率、trial、MN9 spike 数和 Hz。

### 12.4 扩大为 1s、多随机重复

```bash
docker run --rm --init \
  --user "$(id -u):$(id -g)" \
  -e HOME=/work/tmp/reference-home \
  -v "$PWD:/work" -w /work \
  flybrain-reference:0.1 \
  python scripts/reference.py \
    --seconds 1.0 --trials 30 --rates 0 100 150 \
    --codegen cython --out runs/reference/sugar-30trials.csv
```

Cython 路径第一次会编译。先用 numpy codegen 排除编译问题，然后再比较速度。单次 100ms 没观察到输出不等于整篇研究失败；应按原论文刺激条件、完整时长和多次重复检查。[1]

**验收层次：** 能运行并生成 CSV 是软件验收；与作者特定图表、刺激协议和统计结果对齐才叫科学复现。本文没有预先写入“MN9 应当精确等于某个 Hz”的伪验收值，也没有实测你的数据。

### 12.5 不要这样混用

不要把 `data/male-v1.npz` 传给这个原模型；不要把 v630 的 ID 列表直接用于 v783；不要对原模型做了归一化/改时间步后仍称“原版复现”；不要在参考 CPU 环境里安装 GPU Torch 并试图覆盖 A 路线。

---

## 13. C 路线：安装 FlyGym 身体，先让身体正常工作

FlyGym 2.x 与旧版 API 不同，本版固定到 `v2.1.0` 对应 commit，并用现行官方教程的接口。[9][10] 网络上的 FlyGym 1.x 代码不能直接混进本环境。

### 13.1 独立构建身体镜像

可复用已拉取的 NGC 基础层，但在镜像内创建不继承系统 site-packages 的独立 Python venv：

```bash
BASE_IMAGE=$(cat reports/base-image-digest.txt)
docker build --platform linux/arm64 \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  -f docker/Dockerfile.body -t flybrain-body:2.1 .

docker run --rm flybrain-body:2.1 cat /opt/body-pip-freeze.txt \
  > reports/body-pip-freeze.txt
```

该镜像安装固定代码对应的 `flygym[examples,warp]`。因为此版 FlyGym 需要的新 NumPy/Numba 组合与 NGC 主环境可能不同，所以只在**独立 body venv 的安装命令**中忽略 NGC pip 约束；绝不在 core 环境做同样的强制替换。[8][9]

镜像构建失败时先读缺失包/轮子错误。特别是 ARM64、Python ABI、MuJoCo/Warp/Numba 的组合，必须在本机核验；不能用 `--platform linux/amd64` 绕过去。此可选路线失败不影响已经运行的 A 路线。

### 13.2 检查 Python 框架导入

```bash
bash scripts/run_body.sh python -c \
  'import importlib.metadata as m; import mujoco,flygym,warp; print("flygym",m.version("flygym"),"mujoco",mujoco.__version__); print(warp.get_devices())'
```

FlyGym 首次使用模型资源可能联网取网格等资产，HOME 已持久化到 `cache/body-home/`。先联网完成资源预热，确认第二次运行不需要额外下载，再考虑离线执行。[9]

### 13.3 无视频身体 smoke：先排除 EGL

```bash
bash scripts/run_body.sh python scripts/body_demo.py \
  --seconds 0.2 --no-video --out runs/body-no-video

cat runs/body-no-video/report.json
```

它使用官方混合转向控制器：输入左右两个高层信号，控制器产生关节/附着控制命令。[10]

### 13.4 生成可看的转向影片

```bash
bash scripts/run_body.sh python scripts/body_demo.py \
  --seconds 1.0 --out runs/body-video

find runs/body-video -maxdepth 2 -type f
```

输出位置由 FlyGym 渲染器的 `save_video` 行为决定；脚本请求 `runs/body-video/turning.mp4`，某些多相机保存方式可能再添加相机名。用 `find` 确认实际影片文件，而不是假设路径一定只会有一个固定 basename。

同时有 `thorax_xyz.npy` 与 `report.json`；后者记录位移和控制来源。**这个视频不是训练完成的证据：左右信号在前后半段由脚本手动切换。**

EGL 失败而无视频能跑，说明优先排渲染栈：确认容器有 `graphics` capability、`MUJOCO_GL=egl`、`PYOPENGL_PLATFORM=egl` 和宿主机驱动支持。不要重装整个训练环境。

### 13.5 GPU 物理并行 smoke

```bash
bash scripts/run_body.sh python scripts/body_gpu_smoke.py \
  --worlds 4 --steps 100 --out reports/body-gpu-w4.json

bash scripts/run_body.sh python scripts/body_gpu_smoke.py \
  --worlds 16 --steps 1000 --out reports/body-gpu-w16.json
```

这调用 FlyGym 的 `GPUSimulation`/MuJoCo-Warp，默认保持中性姿态，仅验证并行物理执行；不生成影片，也不训练行走。[11] 首次 JIT/warmup 不计入脚本测量窗口；它仍然只是 smoke 级吞吐统计，不是完整稳定性测评。

官网在特定硬件和设置上的加速数字**不能当成你的 Spark 全脑+身体闭环速度**。请以本机 JSON 和完整任务的 wall time 为准。

### 13.6 下一阶段连接脑与身体的明确接口，而不是虚假的完成声明

本版提供的是两个已分开的工程模块，**未提供训练完成的神经脑→六足全闭环模型**。真正连接时建议先只学习两个高层控制值，而不是一步学习全部关节：

```text
身体姿态/目标相对方向（新环境 observation）
                 ↓
         固定传感器到神经输入映射
                 ↓
        同一套稀疏 reservoir engine
                 ↓
     两个有界高层下降控制量 left / right
                 ↓
     HybridTurningController（低层步态/修正）
                 ↓
                 MuJoCo
```

迷宫策略输出 4 个离散动作；身体控制是另一种动作空间、观察空间和奖励，**不能把同一个 checkpoint 直接套上去**。要另写对应环境，重新训练并验证动作限幅、控制频率、reset、接触稳定性与跌倒终止条件。

后续加入真实嗅觉/视觉投射或神经调质可塑性，也应单独建立基线。STDP、reward-STDP、受体/递质模型不是在配置里加一个字符串就已完成；本版不伪造这些算法已集成。

---

## 14. 网络与断网运行：尽量避免四台重复下载

### 14.1 主线必须保存的资产

```text
代码包和 sources.lock.json
flybrain-lab:0.1 镜像导出包 + 校验文件
核心 image ID、base digest、pip-freeze
原始 MaleCNS 预处理两个 npz + 已核验的 SHA256
转换后的 data/male-v1.npz 与 .json 来源文件
训练 checkpoint、config、metrics、eval 结果
```

FlyWire/Brian2 为可选资产：固定 upstream 仓库、reference 镜像；身体路线为 body 镜像和已预热资源缓存。**镜像已经包含框架依赖，`docker load` 后不需要再 `pip install` 一遍。**

### 14.2 网络完全关闭前，做一次主线离线验收

先确保项目路径中的数据已齐全。单机训练不需要外网，可以用没有网络的容器测试：

```bash
docker run --rm --init --gpus all --network none --shm-size 2g \
  --user "$(id -u):$(id -g)" -e HOME=/work/tmp \
  -v "$PWD:/work" -w /work flybrain-lab:0.1 \
  python flylab.py train \
    --graph data/male-v1.npz --device cuda \
    --envs 2 --steps 8 --minibatch 16 --epochs 1 --updates 1 \
    --out runs/offline-smoke
```

四机仍需私有网络，但不需要互联网。身体第一次用到未预热网格资源时仍可能联网；所以不能仅凭 pip 已安装就断言身体仿真全离线。

### 14.3 保留和清理原则

可以在确认镜像已成功 load、数据已分发且另有备份后，手动移走镜像传输压缩包释放空间。训练成功前先不删原始数据，转换格式或做科学对比时很有价值。

不提供 `docker system prune -a --volumes` 这类全局清理命令。它和果蝇项目范围不匹配，可能清掉你现有大模型缓存和容器资源。

---

## 15. 验收表：每一层应该留下什么证据

| Gate | 必须完成 | 应留下的文件/现象 | 不满足时停在哪 |
|---|---|---|---|
| G0 宿主 | ARM64、GPU可见、容器可用、磁盘/内存足够 | preflight 报告 | 不构建大规模任务，不动驱动或现有网络 |
| G1 GPU | 实际 CUDA CSR 运算通过 | `spark-gpu-check.json` 的 PASS | 不加载全量图训练 |
| G2 软件 | 单元测试、synthetic训练/评估、同机双进程 | pytest、checkpoint、Gloo求和3 | 修代码/依赖，不把问题推给连接组 |
| G3 数据 | 下载hash、方向/ID/schema验证 | 原始SHA、`male-v1.npz.json` | 不绕过校验 |
| G4 单机全量 | 有限状态、输入有响应、短训练无错 | bench、config、metrics、checkpoint | 先查动力学区间/容量/稀疏算子 |
| G5 四机 | 全互通、求和10、参数hash一致 | 四机通信日志、rank0训练日志 | 排网络/rank/配置；可退回独立实验 |
| B 科学参考 | 原模型运行、多重复、协议对齐 | sugar/MN9 CSV和与作者结果的比较 | 不宣称原论文复现成功 |
| C 身体 | 无视频→视频→GPU物理 | 轨迹、影片、Warp日志 | 单独排身体环境，不破坏A路线 |

最终“学会了”的最低证据不是能输出动作，而是：固定训练预算、多个独立 seed、持出的评估采样、随机/打乱/无传播/直接策略对照，以及完整参数和图版本记录。

---

## 16. 常见错误与具体排查顺序

### 16.1 GPU看得到，但 PyTorch 不能运行

```bash
docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.11-py3 nvidia-smi
bash scripts/run_core.sh python -c \
  'import torch; print(torch.__version__,torch.version.cuda,torch.cuda.is_available()); print(torch.cuda.get_arch_list())'
```

区分宿主驱动、容器runtime、Torch构建三层。`get_arch_list()` 列表是构建信息，最终仍以 kernel 实际执行为准。记录报错和版本后按 NVIDIA 兼容说明处理。[8]

### 16.2 图转换报 schema/ID 错误

```bash
bash scripts/run_core.sh python -c \
  'import pandas as pd; p="upstream/Drosophila_brain_model/2023_03_23_connectivity_630_final.parquet"; f=pd.read_parquet(p); print(f.columns.tolist()); print(f.head())'
```

确认下载的是配套版本，不把 Feather 当 Parquet、不把 neuron ID 当 row index、不排序完整性 CSV 后仍使用原索引。不要用浮点格式输出或存储 ID。

### 16.3 特征几乎不变或脑全体都在发放

先 `bench` 比较 gain 0.25/0.5/1/1.5，再看输入对比、放电比例和读出集合。全零差异可能来自输入映射、图方向、静默通路；高饱和可能来自 gain/tonic/归一化。先改一个变量、另存结果，不同时修改拓扑和任务来“让分数变好”。

### 16.4 OOM 或非常慢

先 `--envs 1`、`--iterations 5`，确认没有把 `W` 转成 dense。暂停你自己决定可以中断的其他占 GPU/内存任务后复测；不要批量杀进程。`readout` 变小只会减小输出网络，**不会减少全图 SpMM 成本**。

如果 CPU负载高，确认命令确实使用 `--device cuda` 并通过 gpu-check。完整训练里策略/环境在 CPU 是本版设计，不是全部计算回退；大图状态更新应在 GPU。

### 16.5 多机一直等待

按顺序看：四个唯一rank → 同一 `FLY_NNODES` → 同一master/port → 私有接口 → 任意两台路由 → 防火墙peer连接 → 四台同时启动 → 同一代码与图hash。先 `dist-check`，通过后再加载全脑。

不要用修改 NCCL 环境变量修 Gloo 的错误；本主线没有调用 NCCL。

### 16.6 checkpoint 加载失败

本代码使用 `torch.load(..., weights_only=True)`，并校验图与模型配置。只加载自己生成、来源可信的文件。图hash不同就重新训练；不要删掉保护强行加载。参数改了要明确做新实验。

### 16.7 身体环境 `Numba`/`Warp`/EGL 错误

先框架 import → 无视频 CPU物理 → GPU物理 → EGL视频，分层定位。NVIDIA 25.11 文档列有特定驱动/Numba组合的已知问题，因此本版不升级 core NumPy/Numba；body 的隔离环境仍需要本机验证。[8]

FlyGym新版缺少 ARM64 wheel 或某依赖解析失败时，先保留失败日志：

```bash
docker build --progress=plain --platform linux/arm64 \
  -f docker/Dockerfile.body -t flybrain-body:2.1 . \
  2>&1 | tee reports/body-build.log
```

这一步不具备“强制安装一定成功”的万能参数。不要将 `--ignore-requires-python`、关闭二进制兼容检查或 amd64 模拟当作可验证的 Spark 部署方案。

---

## 17. 已测试内容、仍需本地验证内容与实验边界

本次交付实际执行了 **13 项单元测试**，测试通过。还执行了 CPU稀疏算子、512神经元合成图的PPO更新/保存/重载、同机双进程Gloo通信与同步训练。原始记录见附录 `reports/VALIDATION.md` 及其列出的日志。

没有执行的内容包括：你的四台Spark、ARM64 Docker构建、全量MaleCNS/FlyWire下载和GPU训练、原论文Brian2实验、FlyGym/Warp/EGL真实运行。因此本文使用“安装步骤 + 验收门槛”，而不是声称已经替你完成部署。

本版的 CPU 小图 steps/s、5次episode结果都**不可外推**到你的全量脑；也没有用这些 smoke 数据证明训练收敛。

版本管理上，上游源代码/数据固定了 commit 与 hash；容器里的实际依赖是构建后 freeze，再通过镜像导出分发。首次构建前不冒称整个依赖树已在 Spark 上逐一验证。

---

## 18. 交给你本地 Agent 执行时的指令

下面一段可以作为执行任务说明，但仍需由你授权的本地工具实际执行：

```text
目标：部署并验收 FlyBrain Lab v0.1，而不是改写为另一个框架。

1. 先确认当前主机身份、aarch64架构、GPU、磁盘、Docker和已有任务；保留报告。
2. 不修改宿主机驱动、不覆盖现有LLM环境、不修改现有直连Ring路由。
3. 用提供的文件，不把上文示意图当成现成API；不自行编造下载地址或commit。
4. 按G0→G1→G2→G3→G4推进；每个gate失败就记录命令、stdout/stderr、版本，先定位。
5. 首先使用MaleCNS主线；参考Brian2和FlyGym使用独立容器，不能混合pip依赖。
6. 只在一台下载数据/构建镜像，hash验证后同步；每台分别执行GPU验收。
7. 多机先全互通与dist-check，再短训练，最后长训练；四台配置必须一致。
8. 不把合成图通过叫真实数据通过，不把身体手写控制影片叫大脑训练成功。
9. 输出真实metrics、checkpoint、eval对照、镜像ID、data hash及失败日志。
10. 不运行全局Docker清理、不开放公网服务、不暴露Docker socket。
```

---

## 19. 官方与上游来源

以下为本指南依赖的直接来源。代码路径/数据校验以 `sources.lock.json` 固定版本为准；在线教程可能随后更新，遇到差异先检查版本，不盲目追随最新 main。

[1] Shiu 等人的原始果蝇模型及代码：
- https://github.com/philshiu/Drosophila_brain_model
- https://github.com/philshiu/Drosophila_brain_model/tree/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960
- https://github.com/philshiu/Drosophila_brain_model/blob/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960/model.py
- https://www.nature.com/articles/s41586-024-07763-9

[2] fly.ai / flybrain 社区实现、预处理与资产校验：
- https://github.com/alextitonis/fly.ai
- https://github.com/alextitonis/fly.ai/tree/95a3dbcb05241b0a5c07028ca8ad945b23fbbe6e
- https://github.com/alextitonis/fly.ai/blob/95a3dbcb05241b0a5c07028ca8ad945b23fbbe6e/flybrain/data.py
- https://github.com/alextitonis/fly.ai/blob/95a3dbcb05241b0a5c07028ca8ad945b23fbbe6e/flybrain/build.py
- https://github.com/alextitonis/fly.ai/releases/tag/brain-v1

[3] MaleCNS 官方数据与下载：
- https://male-cns.janelia.org/download/

[4] PyTorch 2.10 分布式通信文档：
- https://docs.pytorch.org/docs/2.10/distributed.html

[5] PyTorch 稀疏矩阵乘法接口：
- https://docs.pytorch.org/docs/2.10/generated/torch.sparse.mm.html

[6] Docker 官方 Ubuntu 安装说明：
- https://docs.docker.com/engine/install/ubuntu/

[7] NVIDIA Container Toolkit 安装/运行时配置：
- https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

[8] NVIDIA PyTorch 容器 25.11 的组件、约束和已知问题：
- https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-11.html

[9] FlyGym 安装与固定源码版本：
- https://neuromechfly.org/installation/
- https://github.com/NeLy-EPFL/flygym/releases/tag/v2.1.0
- https://github.com/NeLy-EPFL/flygym/tree/ca65a510c2afe6ac61c51df4f274c8d190c2f95f

[10] FlyGym 混合转向控制教程：
- https://neuromechfly.org/tutorials/4d_turning_controller/

[11] FlyGym MuJoCo-Warp GPU仿真教程：
- https://neuromechfly.org/tutorials/3_gpu_accelerated_simulation/

第三方数据和代码遵循各自授权，详见 `NOTICE.md`。本工具包没有附带生物数据、第三方模型大文件或已训练好的果蝇策略。

---

# 附录：完整源码、环境定义与验证记录

下面文件块可以由第 2.2 节提取器自动生成。**附录不是伪代码；与 ZIP 中可执行文件保持一致。** `reports/` 内的报告明确标为本次会话 CPU 环境产物，不属于 Spark 实测数据。


## 文件：`README.md`

<!-- FILE: README.md -->
```text
# FlyBrain Lab v0.1 — 4×DGX Spark

先阅读同目录《4DGX_Spark_FlyBrainLab_逐步部署与训练指南.md》。该文档包含全部安装命令、四机配置、科学参考与身体仿真分支，以及与本目录一致的源码附录。

训练主线：MaleCNS/FlyWire稀疏连接图 → 冻结工程LIF reservoir → PPO训练小型策略/价值读出。不是全脑突触反向传播，不是生物大脑完整数字孪生。

主要入口为 `flylab.py --help`。运行容器命令用 `bash scripts/run_core.sh ...`；本机软件自测用 `bash scripts/validate_local.sh`。

执行顺序：宿主检查 → ARM64镜像/GPU稀疏算子 → 合成图软件自测 → 校验真实数据 → 单机全量测试/训练 → 对照实验 → 四机Gloo。不要先改动现有Ring或卸载GPU驱动。

已完成与未完成的验证见 `reports/VALIDATION.md`。本次交付的测试使用x86_64 CPU和合成图，并未在用户Spark上运行；不能用随附CPU小图基准推算全量GPU表现。

资源和授权来源见 `sources.lock.json`、`NOTICE.md`、`LICENSE`。包内不包含上游大脑数据、Docker大镜像或训练好的真实果蝇策略。
```

## 文件：`.dockerignore`

<!-- FILE: .dockerignore -->
```text
.git
upstream
runs
data
cache
reports
__pycache__
.pytest_cache
*.zip
*.tar
```

## 文件：`LICENSE`

<!-- FILE: LICENSE -->
```text

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

## 文件：`NOTICE.md`

<!-- FILE: NOTICE.md -->
```text
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
```

## 文件：`configs/node.env.example`

<!-- FILE: configs/node.env.example -->
```bash
# Copy to configs/node.env on EACH host, edit actual values, then source it.
# All endpoints must be mutually routable; do not assume a physical ring provides L3 routing.
export FLY_NNODES=4
export FLY_NODE_RANK=0
export FLY_MASTER_ADDR=REPLACE_WITH_RANK0_PRIVATE_IP
export FLY_MASTER_PORT=29610
export GLOO_SOCKET_IFNAME=REPLACE_WITH_LOCAL_PRIVATE_INTERFACE
export OMP_NUM_THREADS=4
export FLY_IMAGE=flybrain-lab:0.1
```

## 文件：`docker/Dockerfile.body`

<!-- FILE: docker/Dockerfile.body -->
```dockerfile
# Optional environment: no changes to the core training image or host packages.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3
FROM ${BASE_IMAGE}
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libegl1 libgl1 libglfw3 build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/body-venv
# No system-site-packages: FlyGym can resolve its newer numpy/numba separately.
# Ignore NGC pip constraints ONLY inside this isolated optional environment.
RUN PIP_CONSTRAINT=/dev/null /opt/body-venv/bin/python -m pip install --no-cache-dir \
    'flygym[examples,warp] @ git+https://github.com/NeLy-EPFL/flygym.git@ca65a510c2afe6ac61c51df4f274c8d190c2f95f'
RUN /opt/body-venv/bin/python -m pip freeze > /opt/body-pip-freeze.txt
ENV PATH=/opt/body-venv/bin:$PATH \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
WORKDIR /work
```

## 文件：`docker/Dockerfile.core`

<!-- FILE: docker/Dockerfile.core -->
```dockerfile
# Build natively on ARM64 Spark. No replacement PyTorch wheel and no host driver changes.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3
FROM ${BASE_IMAGE}
USER root
RUN python -c "import torch,numpy,scipy; print(torch.__version__, numpy.__version__, scipy.__version__)"
# Preserve NVIDIA's inherited PIP_CONSTRAINT and installed torch/numba/numpy stack.
RUN python -m pip install --no-cache-dir 'pandas>=2.2,<3' 'pyarrow>=17,<23' 'pytest>=8,<9'
RUN mkdir -p /opt/flylab && python -m pip freeze > /opt/flylab/pip-freeze.txt
WORKDIR /work
ENV PYTHONUNBUFFERED=1
```

## 文件：`docker/Dockerfile.reference`

<!-- FILE: docker/Dockerfile.reference -->
```dockerfile
# Isolated candidate reproduction of the author's Python 3.10 / Brian2 2.5.1 environment.
# CPU only. Must be validated locally on ARM64; do not replace the GPU container's NumPy.
FROM python:3.10-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir 'setuptools==68.2.2' 'wheel==0.41.3' \
    'numpy==1.24.4' 'Cython==0.29.37'
RUN python -m pip install --no-cache-dir --no-build-isolation 'brian2==2.5.1' \
    'pandas==1.5.3' 'scipy==1.10.1' 'pyarrow==12.0.1' 'joblib==1.3.2' 'sympy==1.12'
RUN python -m pip freeze > /opt/reference-pip-freeze.txt
ENV PYTHONUNBUFFERED=1
WORKDIR /work
```

## 文件：`flylab.py`

<!-- FILE: flylab.py -->
```python
#!/usr/bin/env python3
"""FlyBrain Lab v0.1: engineered frozen-connectome reservoir + PPO.

This is NOT a numerical reproduction of the Shiu et al. biological model.
Only the small CPU policy/value readout is trained; the sparse reservoir is frozen.
Run `python flylab.py --help`. See the accompanying Chinese deployment guide.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
import torch.distributed as dist
from torch import nn
from torch.distributions import Categorical

VERSION = "0.1.0"
OBS_DIM = 9


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def checked_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable. No silent CPU fallback.")
        if device.index is None:
            device = torch.device("cuda", 0)
        torch.cuda.set_device(device.index)
    return device


def load_graph(path: str | Path) -> tuple[sp.csr_matrix, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        n = int(z["n"])
        w = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=(n, n))
        ids = z["ids"].astype(str)
    w.check_format(full_check=True)
    if len(ids) != n or len(np.unique(ids)) != n:
        raise ValueError("Neuron IDs must be unique and match matrix dimensions.")
    if not np.isfinite(w.data).all():
        raise ValueError("Nonfinite edge weights.")
    return w.astype(np.float32), ids


def prepare_graph(a: argparse.Namespace) -> None:
    """Convert data without loading any N-by-N dense matrix."""
    out = Path(a.out)
    if out.exists() and not a.overwrite:
        raise FileExistsError(f"Refusing to overwrite {out}; choose a new path.")
    inputs: dict[str, str] = {}
    if a.source == "synthetic":
        if a.neurons < 128:
            raise ValueError("Use at least 128 synthetic neurons.")
        rng = np.random.default_rng(a.seed)
        n = a.neurons
        rows, cols = rng.integers(n, size=(2, n * 16))
        # Dale-like source sign is only for a software fixture, not real fly data.
        signs = rng.choice([-1., 1.], size=n, p=[.2, .8])
        vals = rng.uniform(.2, 1., len(rows)) * signs[cols]
        w = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        ids = np.array([f"SYNTHETIC_{i}" for i in range(n)])
    elif a.source == "shiu":
        import pandas as pd
        if not a.complete or not a.connectivity:
            raise ValueError("shiu requires --complete and --connectivity.")
        # Preserve row order: source parquet endpoints are row indices, not IDs.
        comp = pd.read_csv(a.complete, dtype=str, keep_default_na=False)
        ids = comp.iloc[:, 0].to_numpy(dtype=str)
        if any(not value.isdigit() for value in ids):
            raise ValueError("First completeness column must contain integer ID strings.")
        n = len(ids)
        frame = pd.read_parquet(a.connectivity)
        required = ["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"]
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"Unexpected parquet schema; missing {sorted(missing)}")
        endpoints = []
        for key in required[:2]:
            values = frame[key].to_numpy()
            if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
                raise ValueError(f"Invalid integer endpoint column {key}")
            values = values.astype(np.int64)
            if values.size and (values.min() < 0 or values.max() >= n):
                raise ValueError(f"Endpoint index out of range: {key}")
            endpoints.append(values)
        vals = frame[required[2]].to_numpy(dtype=np.float32)
        w = sp.coo_matrix((vals, (endpoints[1], endpoints[0])), shape=(n, n)).tocsr()
        inputs = {str(Path(p).name): sha256(p) for p in [a.complete, a.connectivity]}
    elif a.source == "male":
        if not a.weights or not a.brain:
            raise ValueError("male requires --weights and --brain.")
        w = sp.load_npz(a.weights).tocsr()
        with np.load(a.brain, allow_pickle=False) as z:
            ids = z["ids"].astype(str)
        n = len(ids)
        if w.shape != (n, n):
            raise ValueError("MaleCNS metadata/weight shape mismatch.")
        inputs = {str(Path(p).name): sha256(p) for p in [a.weights, a.brain]}
    else:
        raise ValueError(a.source)
    if n == 0 or len(np.unique(ids)) != n or not np.isfinite(w.data).all():
        raise ValueError("Empty graph, duplicate IDs, or nonfinite weights.")
    w.sum_duplicates()
    w.eliminate_zeros()
    # Engineering normalization, not paper-exact synaptic conductance.
    sums = np.asarray(abs(w).sum(axis=1)).ravel()
    w = (sp.diags(1. / np.maximum(sums, 1.)) @ w).tocsr().astype(np.float32)
    w.sort_indices()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        np.savez_compressed(f, n=np.int64(n), indptr=w.indptr.astype(np.int64),
                            indices=w.indices.astype(np.int64), data=w.data, ids=ids)
    meta = {"version": VERSION, "source": a.source, "synthetic": a.source == "synthetic",
            "neurons": n, "nonzero_neuron_pair_edges": int(w.nnz),
            "orientation": "W[postsynaptic, presynaptic]",
            "normalization": "divide each row by max(absolute incoming sum, 1)",
            "file_sha256": sha256(out), "input_sha256": inputs,
            "warning": "Engineered reservoir, not a biologically validated whole-brain digital twin."}
    write_json(str(out) + ".json", meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


class Reservoir:
    """Sparse, frozen simplified LIF network. Batch state has shape (neurons, envs).

    Equation per internal step:
      v <- exp(-dt/tau)*v + gain*W*s + tonic + engineered sensory drive
      s <- 1[v >= 1]; v[s] <- 0; trace <- .8*trace + .2*s
    dt=.020, tau=.100; no biological synaptic delay/refractory/receptor model.
    """
    def __init__(self, path: str, batch: int, device: str = "cpu", mode: str = "brain",
                 graph_seed: int = 64, readout: int = 128, internal_steps: int = 4,
                 gain: float = 1., tonic: float = .14, layout: str = "csr"):
        if internal_steps < 1 or batch < 1 or readout < 1:
            raise ValueError("Invalid reservoir dimensions.")
        self.device = checked_device(device)
        self.batch, self.mode = batch, mode
        self.internal_steps, self.gain, self.tonic = internal_steps, gain, tonic
        self.n_features = 2 * readout
        w, _ = load_graph(path)
        self.n = w.shape[0]
        rng = np.random.default_rng(graph_seed)
        per_channel = min(16, max(2, self.n // (OBS_DIM * 8)))
        inputs = rng.choice(self.n, OBS_DIM * per_channel, replace=False)
        # Keep identical I/O mappings across brain/shuffled/no_recurrence controls.
        # Read out one-hop targets, excluding the injected input cells themselves.
        indicator = np.zeros(self.n, dtype=np.float32)
        indicator[inputs] = 1
        targets = np.asarray(abs(w) @ indicator).ravel()
        targets[inputs] = 0
        candidates = np.flatnonzero(targets > 0)
        if len(candidates) < readout:
            raise ValueError(f"Only {len(candidates)} non-input one-hop targets; reduce --readout.")
        outputs = rng.choice(candidates, readout, replace=False)
        if mode == "shuffled":
            # Global source-label permutation. Not per-neuron degree-preserving rewiring.
            permutation = rng.permutation(self.n)
            w = sp.csr_matrix((w.data.copy(), permutation[w.indices], w.indptr.copy()),
                              shape=w.shape)
            w.sort_indices()
        self.mapping_sha256 = hashlib.sha256(inputs.tobytes() + outputs.tobytes()).hexdigest()
        self.inputs = torch.as_tensor(inputs, dtype=torch.long, device=self.device)
        self.channels = torch.as_tensor(np.repeat(np.arange(OBS_DIM), per_channel),
                                        dtype=torch.long, device=self.device)
        self.outputs = torch.as_tensor(outputs, dtype=torch.long, device=self.device)
        self.w = torch.sparse_csr_tensor(
            torch.as_tensor(w.indptr.astype(np.int64), device=self.device),
            torch.as_tensor(w.indices.astype(np.int64), device=self.device),
            torch.as_tensor(w.data, device=self.device),
            size=w.shape, dtype=torch.float32, device=self.device)
        if layout == "coo":
            self.w = self.w.to_sparse_coo().coalesce()
        self.v = torch.zeros((self.n, batch), dtype=torch.float32, device=self.device)
        self.s = torch.zeros_like(self.v)
        self.trace = torch.zeros_like(self.v)
        self.last_activity = 0.

    @torch.no_grad()
    def reset(self, mask: np.ndarray | None = None) -> None:
        if mask is None:
            self.v.zero_(); self.s.zero_(); self.trace.zero_()
        elif np.any(mask):
            m = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            self.v[:, m] = 0; self.s[:, m] = 0; self.trace[:, m] = 0

    @torch.no_grad()
    def features(self, observations: np.ndarray) -> torch.Tensor:
        if observations.shape != (self.batch, OBS_DIM):
            raise ValueError(f"Expected observation shape {(self.batch, OBS_DIM)}")
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        drive = .8 * obs[:, self.channels].T
        for _ in range(self.internal_steps):
            if self.mode != "no_recurrence":
                current = torch.sparse.mm(self.w, self.s)
                self.v.mul_(math.exp(-.020 / .100)).add_(current, alpha=self.gain)
            else:
                self.v.mul_(math.exp(-.020 / .100))
            self.v.add_(self.tonic)
            self.v[self.inputs] += drive
            fired = self.v >= 1.
            self.s.copy_(fired)
            self.v.masked_fill_(fired, 0.)
            self.trace.mul_(.8).add_(self.s, alpha=.2)
        self.last_activity = float(self.s.mean().item())
        features = torch.cat((self.v[self.outputs].T, self.trace[self.outputs].T), dim=1)
        if not torch.isfinite(features).all():
            raise FloatingPointError("Nonfinite reservoir state; inspect parameters.")
        return features.cpu().contiguous()


class Maze:
    """Small fixed map, varying free start/goal, fully observable engineered sensors.

    Reaching the goal OR the visible task deadline is a true finite-horizon terminal.
    This is not Gym's generic TimeLimit truncation. No sensor claims about real flies.
    """
    MOVES = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])

    def __init__(self, batch: int, seed: int, horizon: int = 64):
        self.batch, self.horizon = batch, horizon
        self.rng = np.random.default_rng(seed)
        self.walls = np.zeros((7, 7), dtype=bool)
        self.walls[0, :] = self.walls[-1, :] = True
        self.walls[:, 0] = self.walls[:, -1] = True
        self.walls[1:4, 3] = True
        self.free = np.argwhere(~self.walls)[:, ::-1].copy()  # x,y
        self.pos = np.zeros((batch, 2), dtype=np.int64)
        self.goal = np.zeros_like(self.pos)
        self.age = np.zeros(batch, dtype=np.int64)
        self.returns = np.zeros(batch, dtype=np.float32)
        self.reset(np.ones(batch, dtype=bool))

    def reset(self, mask: np.ndarray) -> None:
        for i in np.flatnonzero(mask):
            choices = self.rng.choice(len(self.free), 2, replace=False)
            self.pos[i], self.goal[i] = self.free[choices]
            self.age[i] = 0
            self.returns[i] = 0

    def observations(self) -> np.ndarray:
        d = (self.goal - self.pos) / 5.
        relative = np.stack([np.maximum(d[:, 0], 0), np.maximum(-d[:, 0], 0),
                             np.maximum(d[:, 1], 0), np.maximum(-d[:, 1], 0)], axis=1)
        neighbor = self.pos[:, None, :] + self.MOVES[None, :, :]
        blocked = self.walls[neighbor[:, :, 1], neighbor[:, :, 0]]
        remain = (1. - self.age / self.horizon)[:, None]
        return np.concatenate((relative, blocked, remain), axis=1).astype(np.float32)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        if actions.shape != (self.batch,) or not np.isin(actions, np.arange(4)).all():
            raise ValueError("Invalid action array.")
        nxt = self.pos + self.MOVES[actions]
        blocked = self.walls[nxt[:, 1], nxt[:, 0]]
        self.pos[~blocked] = nxt[~blocked]
        self.age += 1
        success = np.all(self.pos == self.goal, axis=1)
        rewards = (-.01 - .02 * blocked + success).astype(np.float32)
        self.returns += rewards
        done = success | (self.age >= self.horizon)
        episodes = [{"return": float(self.returns[i]), "success": int(success[i]),
                     "length": int(self.age[i])} for i in np.flatnonzero(done)]
        return rewards, done, episodes

    def text(self, i: int = 0) -> str:
        rows = np.where(self.walls, "#", ".")
        x, y = self.goal[i]; rows[y, x] = "G"
        x, y = self.pos[i]; rows[y, x] = "F"
        return "\n".join("".join(row) for row in rows)


class Policy(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        # Fixed scale + no running statistics; same features reused during PPO update.
        self.net = nn.Sequential(nn.Linear(n_features, 128), nn.Tanh(),
                                 nn.Linear(128, 128), nn.Tanh())
        self.actor, self.critic = nn.Linear(128, 4), nn.Linear(128, 1)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, math.sqrt(2)); nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor.weight, .01)
        nn.init.orthogonal_(self.critic.weight, 1.)

    def forward(self, features: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        hidden = self.net(features)
        return Categorical(logits=self.actor(hidden)), self.critic(hidden).squeeze(-1)


def gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
        last_value: torch.Tensor, gamma: float, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(last_value)
    for t in reversed(range(len(rewards))):
        nxt = last_value if t == len(rewards) - 1 else values[t + 1]
        mask = 1. - dones[t]
        delta = rewards[t] + gamma * nxt * mask - values[t]
        running = delta + gamma * lam * mask * running
        advantages[t] = running
    return advantages, advantages + values


def init_dist() -> tuple[int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist.init_process_group("gloo", timeout=timedelta(minutes=10))
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def agree(value: Any, world: int) -> None:
    if world > 1:
        all_values: list[Any] = [None] * world
        dist.all_gather_object(all_values, value)
        if any(item != all_values[0] for item in all_values):
            raise RuntimeError("Workers disagree about code/data/config: " + repr(all_values))


def mean_gradients(model: nn.Module, world: int) -> None:
    if world > 1:
        params = list(model.parameters())
        flat = torch.cat([p.grad.reshape(-1) for p in params])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world)
        offset = 0
        for p in params:
            p.grad.copy_(flat[offset:offset + p.numel()].view_as(p))
            offset += p.numel()


def standardize(advantages: torch.Tensor, world: int) -> torch.Tensor:
    stats = torch.tensor([advantages.double().sum(), advantages.double().square().sum(),
                          advantages.numel()], dtype=torch.float64)
    if world > 1:
        dist.all_reduce(stats)
    mean = stats[0] / stats[2]
    var = (stats[1] / stats[2] - mean.square()).clamp_min(0.)
    return (advantages - float(mean)) / (math.sqrt(float(var)) + 1e-8)


def reservoir_from(a: argparse.Namespace, batch: int) -> Reservoir | None:
    if a.mode == "direct":
        return None
    return Reservoir(a.graph, batch, a.device, a.mode, a.graph_seed, a.readout,
                     a.internal_steps, a.gain, a.tonic, a.layout)


def get_features(brain: Reservoir | None, env: Maze) -> torch.Tensor:
    obs = env.observations()
    return brain.features(obs) if brain else torch.from_numpy(obs)


def atomic_checkpoint(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def train(a: argparse.Namespace) -> None:
    rank, world = init_dist()
    try:
        if a.steps * a.envs % a.minibatch:
            raise ValueError("--steps * --envs must be divisible by --minibatch.")
        checked_device(a.device)  # cuda means an actual GPU must exist, even for controls.
        signature = sha256(a.graph)
        agreement = {k: v for k, v in vars(a).items()
                     if k not in {"graph", "out", "device", "func", "warm_start"}}
        agree({"graph": signature, "code": sha256(__file__), "config": agreement,
               "torch": str(torch.__version__)}, world)
        out = Path(a.out)
        if rank == 0:
            out.mkdir(parents=True, exist_ok=True)
            if (out / "metrics.jsonl").exists():
                raise FileExistsError("Use a new --out directory; previous run exists.")
        if world > 1:
            dist.barrier()
        set_seed(a.seed)
        brain = reservoir_from(a, a.envs)
        model = Policy(brain.n_features if brain else OBS_DIM)  # explicitly CPU
        if a.warm_start:
            saved = torch.load(a.warm_start, map_location="cpu", weights_only=True)
            for k in ["mode", "graph_seed", "readout", "internal_steps", "gain", "tonic"]:
                if saved["config"][k] != getattr(a, k):
                    raise ValueError(f"Warm-start configuration differs: {k}")
            if saved["graph_sha256"] != signature:
                raise ValueError("Warm-start graph hash mismatch.")
            model.load_state_dict(saved["policy"])
        state_hash = hashlib.sha256(torch.nn.utils.parameters_to_vector(
            model.parameters()).detach().numpy().tobytes()).hexdigest()
        agree(state_hash, world)
        # Rank-specific environment and action sampling, after identical model init.
        set_seed(a.seed + rank * 100003)
        optimizer = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-5)
        env = Maze(a.envs, a.seed + rank * 100003, a.horizon)
        features = get_features(brain, env)
        config = {k: v for k, v in vars(a).items() if k != "func"}
        config.update(version=VERSION, world_size=world)
        if rank == 0:
            write_json(out / "config.json", {**config, "graph_sha256": signature,
                      "mapping_sha256": brain.mapping_sha256 if brain else None,
                      "trainable_parameters": sum(p.numel() for p in model.parameters()),
                      "python": platform.python_version(), "torch": str(torch.__version__)})
        start = time.perf_counter()
        for update in range(1, a.updates + 1):
            xs, acts, logs, vals, rewards, dones, episodes = [], [], [], [], [], [], []
            for _ in range(a.steps):
                with torch.no_grad():
                    distribution, value = model(features)
                    action = distribution.sample()
                    old_log = distribution.log_prob(action)
                reward, done, ended = env.step(action.numpy())
                xs.append(features); acts.append(action); logs.append(old_log); vals.append(value)
                rewards.append(torch.from_numpy(reward)); dones.append(torch.from_numpy(done).float())
                episodes.extend(ended)
                env.reset(done)
                if brain:
                    brain.reset(done)
                features = get_features(brain, env)
            with torch.no_grad():
                _, last_value = model(features)
                advantage, targets = gae(torch.stack(rewards), torch.stack(vals), torch.stack(dones),
                                          last_value, a.gamma, a.gae_lambda)
            x = torch.stack(xs).flatten(0, 1)
            actions = torch.stack(acts).flatten()
            old_logs = torch.stack(logs).flatten()
            advantages = standardize(advantage.flatten(), world)
            targets = targets.flatten()
            loss_total, kl_total, batches = 0., 0., 0
            for _ in range(a.epochs):
                order = torch.randperm(len(x))
                for idx in order.split(a.minibatch):
                    distribution, value = model(x[idx])
                    new_log = distribution.log_prob(actions[idx])
                    ratio = (new_log - old_logs[idx]).exp()
                    plain = advantages[idx] * ratio
                    clipped = advantages[idx] * ratio.clamp(1 - a.clip, 1 + a.clip)
                    loss = -torch.minimum(plain, clipped).mean()
                    loss = loss + .5 * (value - targets[idx]).square().mean()
                    loss = loss - a.entropy * distribution.entropy().mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite PPO loss.")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    mean_gradients(model, world)
                    nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
                    optimizer.step()
                    with torch.no_grad():
                        kl_total += float(((ratio - 1.) - (new_log - old_logs[idx])).mean())
                    loss_total += float(loss.detach()); batches += 1
            totals = torch.tensor([len(episodes), sum(e["success"] for e in episodes),
                                    sum(e["return"] for e in episodes), loss_total / batches,
                                    kl_total / batches], dtype=torch.float64)
            if world > 1:
                dist.all_reduce(totals)
            # Catch parameter divergence instead of silently saving rank0 only.
            param_hash = hashlib.sha256(torch.nn.utils.parameters_to_vector(
                model.parameters()).detach().numpy().tobytes()).hexdigest()
            agree(param_hash, world)
            if rank == 0:
                count = float(totals[0])
                metrics = {"update": update, "env_steps": update * a.steps * a.envs * world,
                           "episodes": int(count), "success_rate": float(totals[1] / count) if count else None,
                           "mean_return": float(totals[2] / count) if count else None,
                           "loss": float(totals[3] / world), "approx_kl": float(totals[4] / world),
                           "last_spike_fraction_rank0": brain.last_activity if brain else None,
                           "wall_seconds": time.perf_counter() - start, "policy_sha256": param_hash}
                with open(out / "metrics.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(metrics) + "\n")
                print(json.dumps(metrics), flush=True)
                if update % a.save_every == 0 or update == a.updates:
                    atomic_checkpoint(out / "policy.pt", {"policy": model.state_dict(),
                        "config": config, "graph_sha256": signature, "updates": update,
                        "note": "Readout warm start only: optimizer and environment are not resumed."})
        if world > 1:
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def evaluate(a: argparse.Namespace) -> None:
    checked_device(a.device)
    set_seed(a.seed)
    checkpoint = None
    if a.checkpoint:
        checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
        if checkpoint["graph_sha256"] != sha256(a.graph):
            raise ValueError("Evaluation graph differs from training graph.")
        for key in ["mode", "graph_seed", "readout", "internal_steps", "gain", "tonic", "horizon"]:
            setattr(a, key, checkpoint["config"][key])
    elif not a.random:
        raise ValueError("Supply --checkpoint or --random.")
    # Exactly one complete episode at a time avoids selecting only faster-finishing envs.
    env = Maze(1, a.seed, a.horizon)
    brain = reservoir_from(a, 1) if not a.random else None
    model = Policy(brain.n_features if brain else OBS_DIM)
    if checkpoint and not a.random:
        model.load_state_dict(checkpoint["policy"])
    results = []
    trajectory = []
    for episode in range(a.episodes):
        if episode:
            env.reset(np.array([True]))
        if brain:
            brain.reset()
        for step in range(a.horizon):
            if episode < a.trace_episodes:
                trajectory.append(f"episode={episode} step={step}\n{env.text()}\n")
            with torch.no_grad():
                if a.random:
                    action = torch.randint(4, (1,))
                else:
                    distribution, _ = model(get_features(brain, env))
                    action = distribution.probs.argmax(-1) if a.greedy else distribution.sample()
            _, done, ended = env.step(action.numpy())
            if done[0]:
                results.extend(ended)
                break
    successes = sum(x["success"] for x in results)
    n = len(results)
    p = successes / n
    z = 1.96
    center = (p + z*z/(2*n)) / (1+z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    report = {"episodes": n, "seed": a.seed, "successes": successes, "success_rate": p,
              "wilson_95_interval_descriptive": [center-half, center+half],
              "mean_return": float(np.mean([x["return"] for x in results])),
              "mean_length": float(np.mean([x["length"] for x in results])),
              "action_selection": "uniform_random" if a.random else "greedy" if a.greedy else "sampled",
              "graph_sha256": sha256(a.graph), "mode": a.mode,
              "warning": "Same fixed map; varying start/goal. Not an unseen-map generalization test."}
    write_json(a.out, report)
    Path(str(a.out) + ".trace.txt").write_text("\n".join(trajectory), encoding="utf-8")
    print(json.dumps(report, indent=2))


def gpu_check(a: argparse.Namespace) -> None:
    device = checked_device(a.device)
    crow = torch.tensor([0, 2, 3], device=device)
    col = torch.tensor([0, 1, 1], device=device)
    val = torch.tensor([2., -1., 3.], device=device)
    w = torch.sparse_csr_tensor(crow, col, val, size=(2, 2), device=device)
    x = torch.tensor([[1., 4.], [2., 5.]], device=device)
    result = torch.sparse.mm(w, x).cpu()
    torch.testing.assert_close(result, torch.tensor([[0., 3.], [6., 15.]]))
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(json.dumps({"device": str(device), "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda, "architecture": platform.machine(),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "capability": torch.cuda.get_device_capability(device) if device.type == "cuda" else None,
        "csr_spmm": "PASS"}, indent=2))


def dist_check(a: argparse.Namespace) -> None:
    rank, world = init_dist()
    try:
        value = torch.tensor([rank + 1.], dtype=torch.float64)
        if world > 1:
            dist.all_reduce(value)
        assert value.item() == world * (world + 1) / 2
        print(json.dumps({"rank": rank, "world_size": world, "backend": "gloo/CPU",
                          "sum": value.item(), "status": "PASS"}), flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def bench(a: argparse.Namespace) -> None:
    brain = reservoir_from(a, a.envs)
    if brain is None:
        raise ValueError("Benchmark requires a reservoir mode.")
    obs = Maze(a.envs, a.seed).observations()
    for _ in range(3):
        brain.features(obs)
    brain.reset()
    first = brain.features(obs)
    brain.reset()
    changed = obs.copy(); changed[:, :4] = np.roll(changed[:, :4], 1, axis=1)
    second = brain.features(changed)
    contrast = float((first - second).abs().mean())
    if brain.device.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    activity = []
    for _ in range(a.iterations):
        brain.features(obs)
        activity.append(brain.last_activity)
    if brain.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    report = {"mode": a.mode, "device": a.device, "batch_envs": a.envs,
              "internal_steps_per_env_step": a.internal_steps,
              "batched_calls": a.iterations, "seconds": elapsed,
              "aggregate_env_steps_per_second": a.iterations*a.envs/elapsed,
              "last_spike_fraction": brain.last_activity,
              "mean_spike_fraction": float(np.mean(activity)),
              "saturation_warning": bool(np.mean(activity) > .5),
              "reset_input_contrast_mean_abs": contrast,
              "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if brain.device.type == "cuda" else None,
              "graph_sha256": sha256(a.graph), "mapping_sha256": brain.mapping_sha256,
              "warning": "Reservoir-only benchmark including feature transfer; not PPO/FlyGym throughput."}
    write_json(a.out, report); print(json.dumps(report, indent=2))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threads", type=int, default=4, help="CPU Torch threads per worker.")
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("--source", choices=["synthetic", "shiu", "male"], required=True)
    q.add_argument("--out", required=True)
    for arg in ["complete", "connectivity", "weights", "brain"]:
        q.add_argument("--"+arg)
    q.add_argument("--neurons", type=int, default=512)
    q.add_argument("--seed", type=int, default=64)
    q.add_argument("--overwrite", action="store_true")
    q.set_defaults(func=prepare_graph)
    q = sub.add_parser("inspect"); q.add_argument("--graph", required=True)
    q.set_defaults(func=lambda a: print(json.dumps({"sha256": sha256(a.graph),
        "shape": load_graph(a.graph)[0].shape, "nnz": load_graph(a.graph)[0].nnz,
        "metadata": json.loads(Path(a.graph+".json").read_text())
            if Path(a.graph+".json").exists() else None}, indent=2)))
    q = sub.add_parser("gpu-check"); q.add_argument("--device", default="cuda"); q.set_defaults(func=gpu_check)
    q = sub.add_parser("dist-check"); q.set_defaults(func=dist_check)
    for command, func in [("train", train), ("eval", evaluate), ("bench", bench)]:
        q = sub.add_parser(command)
        q.add_argument("--graph", required=True)
        q.add_argument("--device", default="cuda")
        q.add_argument("--mode", choices=["brain", "shuffled", "no_recurrence", "direct"], default="brain")
        q.add_argument("--graph-seed", type=int, default=64)
        q.add_argument("--readout", type=int, default=128)
        q.add_argument("--internal-steps", type=int, default=4)
        q.add_argument("--gain", type=float, default=1.)
        q.add_argument("--tonic", type=float, default=.14)
        q.add_argument("--layout", choices=["csr", "coo"], default="csr")
        q.add_argument("--seed", type=int, default=1 if command == "train" else 900001)
        q.add_argument("--out", required=True)
        if command in {"train", "bench"}:
            q.add_argument("--envs", type=int, default=8)
        if command in {"train", "eval"}:
            q.add_argument("--horizon", type=int, default=64)
        if command == "train":
            q.add_argument("--steps", type=int, default=64)
            q.add_argument("--updates", type=int, default=100)
            q.add_argument("--epochs", type=int, default=4)
            q.add_argument("--minibatch", type=int, default=128)
            q.add_argument("--lr", type=float, default=3e-4)
            q.add_argument("--gamma", type=float, default=.99)
            q.add_argument("--gae-lambda", type=float, default=.95)
            q.add_argument("--clip", type=float, default=.2)
            q.add_argument("--entropy", type=float, default=.01)
            q.add_argument("--save-every", type=int, default=10)
            q.add_argument("--warm-start")
        elif command == "eval":
            q.add_argument("--checkpoint")
            q.add_argument("--random", action="store_true")
            q.add_argument("--greedy", action="store_true")
            q.add_argument("--episodes", type=int, default=200)
            q.add_argument("--trace-episodes", type=int, default=2)
        else:
            q.add_argument("--iterations", type=int, default=20)
        q.set_defaults(func=func)
    return p


def main() -> None:
    a = parser().parse_args()
    torch.set_num_threads(a.threads)
    for key in ["threads", "episodes", "horizon", "envs", "updates", "steps", "epochs", "minibatch", "save_every", "iterations"]:
        if hasattr(a, key) and getattr(a, key) <= 0:
            raise ValueError(f"--{key.replace('_', '-')} must be positive.")
    a.func(a)


if __name__ == "__main__":
    main()
```

## 文件：`reports/VALIDATION.md`

<!-- FILE: reports/VALIDATION.md -->
```text
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
```

## 文件：`reports/cpu-benchmark.json`

<!-- FILE: reports/cpu-benchmark.json -->
```json
{
  "mode": "brain",
  "device": "cpu",
  "batch_envs": 4,
  "internal_steps_per_env_step": 4,
  "batched_calls": 20,
  "seconds": 0.005105662000005395,
  "aggregate_env_steps_per_second": 15668.878981788348,
  "last_spike_fraction": 0.19287109375,
  "mean_spike_fraction": 0.1592041015625,
  "saturation_warning": false,
  "reset_input_contrast_mean_abs": 0.013922130689024925,
  "cuda_peak_allocated_bytes": null,
  "graph_sha256": "be43d1e7d581b4079af059a3112212228202e297eca4776078c97d6b74599a66",
  "mapping_sha256": "867109e67e3022892493383c15aa90bc24540ed27d3b39b49f3f9932aee4f4a8",
  "warning": "Reservoir-only benchmark including feature transfer; not PPO/FlyGym throughput."
}
```

## 文件：`reports/cpu-spmm.json`

<!-- FILE: reports/cpu-spmm.json -->
```json
{
  "device": "cpu",
  "torch": "2.10.0+cpu",
  "cuda_runtime": null,
  "architecture": "x86_64",
  "gpu": null,
  "capability": null,
  "csr_spmm": "PASS"
}
```

## 文件：`reports/full-software-validation.txt`

<!-- FILE: reports/full-software-validation.txt -->
```text
.............                                                            [100%]
=============================== warnings summary ===============================
tests/test_flylab.py::test_reservoir_determinism_and_reset
  /mnt/data/flybrain_lab_4spark_v0_1/flylab.py:201: UserWarning: Sparse CSR tensor support is in beta state. If you miss a functionality in the sparse tensor support, please submit a feature request to https://github.com/pytorch/pytorch/issues. (Triggered internally at /pytorch/aten/src/ATen/SparseCsrTensorImpl.cpp:49.)
    self.w = torch.sparse_csr_tensor(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
13 passed, 1 warning in 1.17s
{
  "version": "0.1.0",
  "source": "synthetic",
  "synthetic": true,
  "neurons": 512,
  "nonzero_neuron_pair_edges": 8068,
  "orientation": "W[postsynaptic, presynaptic]",
  "normalization": "divide each row by max(absolute incoming sum, 1)",
  "file_sha256": "be43d1e7d581b4079af059a3112212228202e297eca4776078c97d6b74599a66",
  "input_sha256": {},
  "warning": "Engineered reservoir, not a biologically validated whole-brain digital twin."
}
/mnt/data/flybrain_lab_4spark_v0_1/flylab.py:576: UserWarning: Sparse CSR tensor support is in beta state. If you miss a functionality in the sparse tensor support, please submit a feature request to https://github.com/pytorch/pytorch/issues. (Triggered internally at /pytorch/aten/src/ATen/SparseCsrTensorImpl.cpp:49.)
  w = torch.sparse_csr_tensor(crow, col, val, size=(2, 2), device=device)
{
  "device": "cpu",
  "torch": "2.10.0+cpu",
  "cuda_runtime": null,
  "architecture": "x86_64",
  "gpu": null,
  "capability": null,
  "csr_spmm": "PASS"
}
/mnt/data/flybrain_lab_4spark_v0_1/flylab.py:201: UserWarning: Sparse CSR tensor support is in beta state. If you miss a functionality in the sparse tensor support, please submit a feature request to https://github.com/pytorch/pytorch/issues. (Triggered internally at /pytorch/aten/src/ATen/SparseCsrTensorImpl.cpp:49.)
  self.w = torch.sparse_csr_tensor(
{"update": 1, "env_steps": 64, "episodes": 0, "success_rate": null, "mean_return": null, "loss": 0.042247409699484706, "approx_kl": 0.00021748198196291924, "last_spike_fraction_rank0": 0.11474609375, "wall_seconds": 0.020759698999881948, "policy_sha256": "f570ac4cdd24087137f60e854d23cd7cbbd277341bbfa9545c3afbd6464fcbf4"}
{"update": 2, "env_steps": 128, "episodes": 1, "success_rate": 1.0, "mean_return": 0.5800000429153442, "loss": 0.04642951022833586, "approx_kl": 9.677838534116745e-05, "last_spike_fraction_rank0": 0.15625, "wall_seconds": 0.03833583200002977, "policy_sha256": "b9b1de36244db62156ad62ceed05da4979faed3d86051df93379dba82d0537f3"}
{"update": 3, "env_steps": 192, "episodes": 1, "success_rate": 1.0, "mean_return": 0.2600002884864807, "loss": 0.0262870981823653, "approx_kl": 6.56861811876297e-05, "last_spike_fraction_rank0": 0.12158203125, "wall_seconds": 0.05664445299998988, "policy_sha256": "b8b13a8b6e4c22a231769dfccd451732c5fbf65463da10bbfde3c9eb3df58c56"}
/mnt/data/flybrain_lab_4spark_v0_1/flylab.py:201: UserWarning: Sparse CSR tensor support is in beta state. If you miss a functionality in the sparse tensor support, please submit a feature request to https://github.com/pytorch/pytorch/issues. (Triggered internally at /pytorch/aten/src/ATen/SparseCsrTensorImpl.cpp:49.)
  self.w = torch.sparse_csr_tensor(
{
  "episodes": 5,
  "seed": 900001,
  "successes": 4,
  "success_rate": 0.8,
  "wilson_95_interval_descriptive": [
    0.3755282641185388,
    0.9637768390302125
  ],
  "mean_return": 0.1840002179145813,
  "mean_length": 38.4,
  "action_selection": "sampled",
  "graph_sha256": "be43d1e7d581b4079af059a3112212228202e297eca4776078c97d6b74599a66",
  "mode": "brain",
  "warning": "Same fixed map; varying start/goal. Not an unseen-map generalization test."
}
[Gloo] Rank [Gloo] Rank 1 is connected to 1 peer ranks. Expected number of connected peer ranks is : 10 is connected to 1 peer ranks. Expected number of connected peer ranks is : 1

{"rank": 0, "world_size": 2, "backend": "gloo/CPU", "sum": 3.0, "status": "PASS"}
{"rank": 1, "world_size": 2, "backend": "gloo/CPU", "sum": 3.0, "status": "PASS"}
[Gloo] Rank 1 is connected to 1 peer ranks. Expected number of connected peer ranks is : 1
[Gloo] Rank 0 is connected to 1 peer ranks. Expected number of connected peer ranks is : 1
/mnt/data/flybrain_lab_4spark_v0_1/flylab.py:201: UserWarning: Sparse CSR tensor support is in beta state. If you miss a functionality in the sparse tensor support, please submit a feature request to https://github.com/pytorch/pytorch/issues. (Triggered internally at /pytorch/aten/src/ATen/SparseCsrTensorImpl.cpp:49.)
  self.w = torch.sparse_csr_tensor(
/mnt/data/flybrain_lab_4spark_v0_1/flylab.py:201: UserWarning: Sparse CSR tensor support is in beta state. If you miss a functionality in the sparse tensor support, please submit a feature request to https://github.com/pytorch/pytorch/issues. (Triggered internally at /pytorch/aten/src/ATen/SparseCsrTensorImpl.cpp:49.)
  self.w = torch.sparse_csr_tensor(
{"update": 1, "env_steps": 64, "episodes": 2, "success_rate": 1.0, "mean_return": 0.8250000178813934, "loss": 0.05917298700660467, "approx_kl": 4.3526291847229004e-05, "last_spike_fraction_rank0": 0.123046875, "wall_seconds": 0.021975435999820547, "policy_sha256": "175df1af90b1d8e671b2306bf2fc6b807f7a9a3d8f35c7040afbbb374eda6b65"}
{"update": 2, "env_steps": 128, "episodes": 0, "success_rate": null, "mean_return": null, "loss": 0.05235862731933594, "approx_kl": 1.4048069715499878e-05, "last_spike_fraction_rank0": 0.1416015625, "wall_seconds": 0.039353808999976536, "policy_sha256": "42237128003b029f529478cc9970b7f563bd6722483ea110dea1374e0a665ccb"}
Software-only validation completed: /mnt/data/flybrain_work/full-validation
```

## 文件：`reports/local-gloo-check.txt`

<!-- FILE: reports/local-gloo-check.txt -->
```text
[Gloo] Rank [Gloo] Rank 1 is connected to 1 peer ranks. Expected number of connected peer ranks is : 10
 is connected to 1 peer ranks. Expected number of connected peer ranks is : 1
{"rank": 1, "world_size": 2, "backend": "gloo/CPU", "sum": 3.0, "status": "PASS"}
{"rank": 0, "world_size": 2, "backend": "gloo/CPU", "sum": 3.0, "status": "PASS"}
```

## 文件：`reports/local-gloo-train.txt`

<!-- FILE: reports/local-gloo-train.txt -->
```text
[Gloo] Rank [Gloo] Rank 1 is connected to 01 is connected to  peer ranks. Expected number of connected peer ranks is : 11 peer ranks. 
Expected number of connected peer ranks is : 1
/mnt/data/flybrain_lab_4spark_v0_1/flylab.py:201: UserWarning: Sparse CSR tensor support is in beta state. If you miss a functionality in the sparse tensor support, please submit a feature request to https://github.com/pytorch/pytorch/issues. (Triggered internally at /pytorch/aten/src/ATen/SparseCsrTensorImpl.cpp:49.)
  self.w = torch.sparse_csr_tensor(
/mnt/data/flybrain_lab_4spark_v0_1/flylab.py:201: UserWarning: Sparse CSR tensor support is in beta state. If you miss a functionality in the sparse tensor support, please submit a feature request to https://github.com/pytorch/pytorch/issues. (Triggered internally at /pytorch/aten/src/ATen/SparseCsrTensorImpl.cpp:49.)
  self.w = torch.sparse_csr_tensor(
{"update": 1, "env_steps": 64, "episodes": 2, "success_rate": 1.0, "mean_return": 0.8250000178813934, "loss": 0.05917298700660467, "approx_kl": 4.3526291847229004e-05, "last_spike_fraction_rank0": 0.123046875, "wall_seconds": 0.022284894999984317, "policy_sha256": "175df1af90b1d8e671b2306bf2fc6b807f7a9a3d8f35c7040afbbb374eda6b65"}
{"update": 2, "env_steps": 128, "episodes": 0, "success_rate": null, "mean_return": null, "loss": 0.05235862731933594, "approx_kl": 1.4048069715499878e-05, "last_spike_fraction_rank0": 0.1416015625, "wall_seconds": 0.0436745570000312, "policy_sha256": "42237128003b029f529478cc9970b7f563bd6722483ea110dea1374e0a665ccb"}
```

## 文件：`reports/pytest.txt`

<!-- FILE: reports/pytest.txt -->
```text
.............                                                            [100%]
=============================== warnings summary ===============================
tests/test_flylab.py::test_reservoir_determinism_and_reset
  /mnt/data/flybrain_lab_4spark_v0_1/flylab.py:201: UserWarning: Sparse CSR tensor support is in beta state. If you miss a functionality in the sparse tensor support, please submit a feature request to https://github.com/pytorch/pytorch/issues. (Triggered internally at /pytorch/aten/src/ATen/SparseCsrTensorImpl.cpp:49.)
    self.w = torch.sparse_csr_tensor(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
13 passed, 1 warning in 1.14s
```

## 文件：`reports/smoke-eval.json`

<!-- FILE: reports/smoke-eval.json -->
```json
{
  "episodes": 5,
  "seed": 900001,
  "successes": 4,
  "success_rate": 0.8,
  "wilson_95_interval_descriptive": [
    0.3755282641185388,
    0.9637768390302125
  ],
  "mean_return": 0.1840002179145813,
  "mean_length": 38.4,
  "action_selection": "sampled",
  "graph_sha256": "be43d1e7d581b4079af059a3112212228202e297eca4776078c97d6b74599a66",
  "mode": "brain",
  "warning": "Same fixed map; varying start/goal. Not an unseen-map generalization test."
}
```

## 文件：`reports/smoke-eval.json.trace.txt`

<!-- FILE: reports/smoke-eval.json.trace.txt -->
```text
episode=0 step=0
#######
#..#..#
#..#..#
#..#..#
#...G.#
#F....#
#######

episode=0 step=1
#######
#..#..#
#..#..#
#..#..#
#F..G.#
#.....#
#######

episode=0 step=2
#######
#..#..#
#..#..#
#..#..#
#F..G.#
#.....#
#######

episode=0 step=3
#######
#..#..#
#..#..#
#..#..#
#F..G.#
#.....#
#######

episode=0 step=4
#######
#..#..#
#..#..#
#..#..#
#F..G.#
#.....#
#######

episode=0 step=5
#######
#..#..#
#..#..#
#..#..#
#...G.#
#F....#
#######

episode=0 step=6
#######
#..#..#
#..#..#
#..#..#
#F..G.#
#.....#
#######

episode=0 step=7
#######
#..#..#
#..#..#
#..#..#
#F..G.#
#.....#
#######

episode=0 step=8
#######
#..#..#
#..#..#
#..#..#
#F..G.#
#.....#
#######

episode=0 step=9
#######
#..#..#
#..#..#
#..#..#
#.F.G.#
#.....#
#######

episode=0 step=10
#######
#..#..#
#..#..#
#.F#..#
#...G.#
#.....#
#######

episode=0 step=11
#######
#..#..#
#..#..#
#..#..#
#.F.G.#
#.....#
#######

episode=0 step=12
#######
#..#..#
#..#..#
#..#..#
#F..G.#
#.....#
#######

episode=0 step=13
#######
#..#..#
#..#..#
#..#..#
#...G.#
#F....#
#######

episode=0 step=14
#######
#..#..#
#..#..#
#..#..#
#F..G.#
#.....#
#######

episode=0 step=15
#######
#..#..#
#..#..#
#F.#..#
#...G.#
#.....#
#######

episode=0 step=16
#######
#..#..#
#..#..#
#..#..#
#F..G.#
#.....#
#######

episode=0 step=17
#######
#..#..#
#..#..#
#..#..#
#.F.G.#
#.....#
#######

episode=0 step=18
#######
#..#..#
#..#..#
#..#..#
#F..G.#
#.....#
#######

episode=0 step=19
#######
#..#..#
#..#..#
#..#..#
#...G.#
#F....#
#######

episode=0 step=20
#######
#..#..#
#..#..#
#..#..#
#...G.#
#F....#
#######

episode=0 step=21
#######
#..#..#
#..#..#
#..#..#
#...G.#
#F....#
#######

episode=0 step=22
#######
#..#..#
#..#..#
#..#..#
#...G.#
#.F...#
#######

episode=0 step=23
#######
#..#..#
#..#..#
#..#..#
#...G.#
#.F...#
#######

episode=0 step=24
#######
#..#..#
#..#..#
#..#..#
#...G.#
#..F..#
#######

episode=0 step=25
#######
#..#..#
#..#..#
#..#..#
#...G.#
#...F.#
#######

episode=1 step=0
#######
#F.#..#
#..#..#
#..#..#
#.G...#
#.....#
#######

episode=1 step=1
#######
#F.#..#
#..#..#
#..#..#
#.G...#
#.....#
#######

episode=1 step=2
#######
#F.#..#
#..#..#
#..#..#
#.G...#
#.....#
#######

episode=1 step=3
#######
#F.#..#
#..#..#
#..#..#
#.G...#
#.....#
#######

episode=1 step=4
#######
#F.#..#
#..#..#
#..#..#
#.G...#
#.....#
#######

episode=1 step=5
#######
#.F#..#
#..#..#
#..#..#
#.G...#
#.....#
#######

episode=1 step=6
#######
#..#..#
#.F#..#
#..#..#
#.G...#
#.....#
#######

episode=1 step=7
#######
#.F#..#
#..#..#
#..#..#
#.G...#
#.....#
#######

episode=1 step=8
#######
#.F#..#
#..#..#
#..#..#
#.G...#
#.....#
#######

episode=1 step=9
#######
#F.#..#
#..#..#
#..#..#
#.G...#
#.....#
#######

episode=1 step=10
#######
#..#..#
#F.#..#
#..#..#
#.G...#
#.....#
#######

episode=1 step=11
#######
#..#..#
#..#..#
#F.#..#
#.G...#
#.....#
#######

episode=1 step=12
#######
#..#..#
#..#..#
#..#..#
#FG...#
#.....#
#######

episode=1 step=13
#######
#..#..#
#..#..#
#F.#..#
#.G...#
#.....#
#######

episode=1 step=14
#######
#..#..#
#F.#..#
#..#..#
#.G...#
#.....#
#######

episode=1 step=15
#######
#F.#..#
#..#..#
#..#..#
#.G...#
#.....#
#######

episode=1 step=16
#######
#F.#..#
#..#..#
#..#..#
#.G...#
#.....#
#######

episode=1 step=17
#######
#.F#..#
#..#..#
#..#..#
#.G...#
#.....#
#######

episode=1 step=18
#######
#..#..#
#.F#..#
#..#..#
#.G...#
#.....#
#######

episode=1 step=19
#######
#.F#..#
#..#..#
#..#..#
#.G...#
#.....#
#######

episode=1 step=20
#######
#.F#..#
#..#..#
#..#..#
#.G...#
#.....#
#######

episode=1 step=21
#######
#.F#..#
#..#..#
#..#..#
#.G...#
#.....#
#######

episode=1 step=22
#######
#..#..#
#.F#..#
#..#..#
#.G...#
#.....#
#######

episode=1 step=23
#######
#..#..#
#..#..#
#.F#..#
#.G...#
#.....#
#######
```

## 文件：`reports/smoke-train.jsonl`

<!-- FILE: reports/smoke-train.jsonl -->
```json
{"update": 1, "env_steps": 64, "episodes": 0, "success_rate": null, "mean_return": null, "loss": 0.042247409699484706, "approx_kl": 0.00021748198196291924, "last_spike_fraction_rank0": 0.11474609375, "wall_seconds": 0.020621491999918362, "policy_sha256": "f570ac4cdd24087137f60e854d23cd7cbbd277341bbfa9545c3afbd6464fcbf4"}
{"update": 2, "env_steps": 128, "episodes": 1, "success_rate": 1.0, "mean_return": 0.5800000429153442, "loss": 0.04642951022833586, "approx_kl": 9.677838534116745e-05, "last_spike_fraction_rank0": 0.15625, "wall_seconds": 0.038301343999933124, "policy_sha256": "b9b1de36244db62156ad62ceed05da4979faed3d86051df93379dba82d0537f3"}
{"update": 3, "env_steps": 192, "episodes": 1, "success_rate": 1.0, "mean_return": 0.2600002884864807, "loss": 0.0262870981823653, "approx_kl": 6.56861811876297e-05, "last_spike_fraction_rank0": 0.12158203125, "wall_seconds": 0.05608531400002903, "policy_sha256": "b8b13a8b6e4c22a231769dfccd451732c5fbf65463da10bbfde3c9eb3df58c56"}
```

## 文件：`reports/synthetic-prepare.json`

<!-- FILE: reports/synthetic-prepare.json -->
```json
{
  "version": "0.1.0",
  "source": "synthetic",
  "synthetic": true,
  "neurons": 512,
  "nonzero_neuron_pair_edges": 8068,
  "orientation": "W[postsynaptic, presynaptic]",
  "normalization": "divide each row by max(absolute incoming sum, 1)",
  "file_sha256": "be43d1e7d581b4079af059a3112212228202e297eca4776078c97d6b74599a66",
  "input_sha256": {},
  "warning": "Engineered reservoir, not a biologically validated whole-brain digital twin."
}
```

## 文件：`scripts/body_demo.py`

<!-- FILE: scripts/body_demo.py -->
```python
#!/usr/bin/env python3
"""Optional FlyGym 2.1.0 integration smoke: a hand-written turning controller.

Not driven by a learned connectome; not a full brain-body reconstruction.
Uses public FlyGym APIs illustrated in the official turning-controller tutorial.
Reference: https://neuromechfly.org/tutorials/4d_turning_controller/
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seconds", type=float, default=1.)
    p.add_argument("--out", default="runs/body-demo")
    p.add_argument("--no-video", action="store_true")
    a = p.parse_args()
    if a.seconds <= 0:
        raise ValueError("Positive --seconds required.")
    from flygym import Simulation
    from flygym.anatomy import BodySegment, ContactBodiesPreset
    from flygym.compose import FlatGroundWorld
    from flygym_demo.complex_terrain import (
        HybridTurningController, HybridControllerObservation, LocomotionAction,
        PreprogrammedSteps, apply_locomotion_action, make_locomotion_fly)
    from flygym.utils.math import Rotation3D

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fly = make_locomotion_fly(name="body_test", add_adhesion=True, colorize=True)
    camera = None if a.no_video else fly.add_tracking_camera(
        name="body_cam", pos_offset=(-.5, -7.5, 0.),
        rotation=Rotation3D("euler", (1.57, 0., 0.)), fovy=35.)
    world = FlatGroundWorld()
    world.add_fly(fly, [0., 0., .8], Rotation3D("quat", [1, 0, 0, 0]),
                  bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
                  add_ground_contact_sensors=False)
    sim = Simulation(world)
    if camera is not None:
        sim.set_renderer([camera], camera_res=(240, 320), playback_speed=.1, output_fps=25)
    steps = PreprogrammedSteps()
    order = fly.get_actuated_jointdofs_order("position")
    controller = HybridTurningController(timestep=sim.timestep,
        preprogrammed_steps=steps, output_dof_order=order)
    sim.reset(); controller.reset(seed=0)
    apply_locomotion_action(sim, fly.name, LocomotionAction(
        joint_angles=steps.default_pose_by_dof_order(order), adhesion_onoff=np.ones(6, dtype=bool)))
    sim.warmup()
    thorax = fly.get_bodysegs_order().index(BodySegment("c_thorax"))
    positions = []
    ticks = int(a.seconds / sim.timestep)
    if ticks < 1:
        raise ValueError("Duration shorter than one simulation step.")
    for i in range(ticks):
        signal = np.array([1.2, .4]) if i < ticks // 2 else np.array([.4, 1.2])
        observation = HybridControllerObservation.from_sim(sim, fly.name)
        apply_locomotion_action(sim, fly.name, controller.step(signal, observation))
        sim.step_with_profile()
        positions.append(sim.get_body_positions(fly.name)[thorax].copy())
        if camera is not None:
            sim.render_as_needed_with_profile()
    trajectory = np.asarray(positions)
    if not np.isfinite(trajectory).all():
        raise FloatingPointError("Nonfinite body trajectory.")
    np.save(out / "thorax_xyz.npy", trajectory)
    if camera is not None:
        sim.renderer.save_video(out / "turning.mp4")
    report = {"physics_steps": ticks, "timestep": float(sim.timestep),
              "actuated_dofs": len(order), "displacement_mm": (trajectory[-1]-trajectory[0]).tolist(),
              "control": "handwritten left/right drive; NOT trained connectome output"}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2)); sim.print_performance_report()


if __name__ == "__main__":
    main()
```

## 文件：`scripts/body_gpu_smoke.py`

<!-- FILE: scripts/body_gpu_smoke.py -->
```python
#!/usr/bin/env python3
"""Optional FlyGym/MuJoCo-Warp GPU availability and parallel physics smoke.

Neutral posture only. This is not locomotion learning or brain-controlled behavior.
Reference: https://neuromechfly.org/tutorials/3_gpu_accelerated_simulation/
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worlds", type=int, default=4)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--out", default="reports/body-gpu-smoke.json")
    a = p.parse_args()
    if min(a.worlds, a.steps) < 1:
        raise ValueError("Positive worlds/steps required.")
    import warp as wp
    from flygym.warp import GPUSimulation
    from flygym.warp.utils import check_gpu
    from flygym.compose import NeuroMechFly, ActuatorType, FlatGroundWorld, KinematicPosePreset
    from flygym.anatomy import Skeleton, JointPreset, ActuatedDOFPreset, AxisOrder
    from flygym.utils.math import Rotation3D
    check_gpu()
    if not wp.is_cuda_available():
        raise RuntimeError("Warp CUDA unavailable; not falling back to CPU.")
    fly = NeuroMechFly()
    skeleton = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL, joint_preset=JointPreset.LEGS_ONLY)
    fly.add_joints(skeleton, neutral_pose=KinematicPosePreset.NEUTRAL)
    dofs = fly.skeleton.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
    fly.add_actuators(dofs, actuator_type=ActuatorType.POSITION, kp=50.,
                      neutral_input=KinematicPosePreset.NEUTRAL)
    fly.add_leg_adhesion()
    world = FlatGroundWorld()
    world.add_fly(fly, (0., 0., .8), Rotation3D("quat", (1, 0, 0, 0)))
    sim = GPUSimulation(world, a.worlds)
    sim.set_leg_adhesion_states(fly.name, np.ones((a.worlds, 6), dtype=np.float32))
    sim.warmup(); wp.synchronize()
    start = time.perf_counter()
    for _ in range(a.steps):
        sim.step_with_profile()
    wp.synchronize()
    elapsed = time.perf_counter()-start
    result = {"worlds": a.worlds, "batch_physics_steps": a.steps, "seconds": elapsed,
              "aggregate_world_steps_per_second": a.worlds*a.steps/elapsed,
              "scope": "execution smoke only, neutral control, no rendering, no learning"}
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2)); sim.print_performance_report()


if __name__ == "__main__":
    main()
```

## 文件：`scripts/download_male.py`

<!-- FILE: scripts/download_male.py -->
```python
#!/usr/bin/env python3
"""Download fly.ai brain-v1 assets, checking even already-existing files.
Hashes pinned from fly.ai 95a3dbcb05241b0a5c07028ca8ad945b23fbbe6e/flybrain/data.py.
Data are not included in this starter kit. Requires network access on your machine.
"""
import argparse
import hashlib
import time
import urllib.request
from pathlib import Path

URL = "https://github.com/alextitonis/fly.ai/releases/download/brain-v1"
FILES = {
    "brain.npz": "cc9bd1ecd00bd703a6fa648bc6ad145c93c7c1ee53debdcc9ce0d1f4305e6aca",
    "weights.npz": "c29919aa44069a271b1ee978abe05fa9bf6e45e4ba3e436e92b624ef1b5be40c",
}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/raw/male")
    a = p.parse_args()
    root = Path(a.out); root.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        target = root / name
        if target.exists():
            if digest(target) == expected:
                print(f"Verified existing: {target}")
                continue
            raise RuntimeError(f"Checksum mismatch in {target}; quarantine it, then retry. Not overwriting.")
        temporary = root / (name + ".part")
        for attempt in range(1, 4):
            try:
                request = urllib.request.Request(f"{URL}/{name}", headers={"User-Agent": "FlyBrainLab/0.1"})
                with urllib.request.urlopen(request, timeout=120) as response, open(temporary, "wb") as f:
                    while block := response.read(1 << 20):
                        f.write(block)
                if digest(temporary) != expected:
                    raise RuntimeError(f"Downloaded {name} has the wrong SHA256. Not using it.")
                temporary.replace(target)
                print(f"Downloaded and verified: {target}")
                break
            except Exception as exc:
                temporary.unlink(missing_ok=True)
                if attempt == 3:
                    raise
                print(f"Attempt {attempt} failed: {exc}; retrying")
                time.sleep(attempt * 3)
    (root / "SHA256SUMS").write_text("".join(f"{v}  {k}\n" for k, v in FILES.items()))


if __name__ == "__main__":
    main()
```

## 文件：`scripts/fetch_shiu.sh`

<!-- FILE: scripts/fetch_shiu.sh -->
```bash
#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
SHA=91bdd1e7dcf193f3e7ca5a8933497fcef63b7960
DIR=upstream/Drosophila_brain_model
mkdir -p upstream
if [[ ! -d "$DIR/.git" ]]; then
  git init "$DIR"
  git -C "$DIR" remote add origin https://github.com/philshiu/Drosophila_brain_model.git
fi
if [[ -n "$(git -C "$DIR" status --porcelain)" ]]; then
  echo "Upstream directory has local changes. Back them up; refusing checkout." >&2
  exit 1
fi
git -C "$DIR" fetch --depth 1 origin "$SHA"
git -C "$DIR" checkout --detach "$SHA"
test "$(git -C "$DIR" rev-parse HEAD)" = "$SHA"
printf '\nPinned upstream: %s\n' "$SHA"
find "$DIR" -maxdepth 2 -type f \( -name '*.parquet' -o -name '*completeness*.csv' -o -name 'Completeness*.csv' \) -printf '%p\n'
```

## 文件：`scripts/launch_rank.sh`

<!-- FILE: scripts/launch_rank.sh -->
```bash
#!/usr/bin/env bash
set -Eeuo pipefail
: "${FLY_NODE_RANK:?Set node rank 0,1,2,3}"
: "${FLY_NNODES:?Set number of participating nodes}"
: "${FLY_MASTER_ADDR:?Set rank0 private, routable IP}"
: "${FLY_MASTER_PORT:=29610}"
: "${GLOO_SOCKET_IFNAME:?Set local reachable private interface}"
exec torchrun --nnodes="$FLY_NNODES" --nproc-per-node=1 \
  --node-rank="$FLY_NODE_RANK" --master-addr="$FLY_MASTER_ADDR" \
  --master-port="$FLY_MASTER_PORT" --max-restarts=0 flylab.py "$@"
```

## 文件：`scripts/preflight.sh`

<!-- FILE: scripts/preflight.sh -->
```bash
#!/usr/bin/env bash
# Read-only inspection. No drivers, network configuration, firewall, or services are changed.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p reports
OUT="reports/preflight-$(hostname)-$(date +%Y%m%d-%H%M%S).txt"
{
  date -Is
  hostname
  uname -a
  cat /etc/os-release
  echo '--- memory/disk ---'
  free -h
  df -h "$PWD"
  echo '--- GPU ---'
  command -v nvidia-smi && nvidia-smi
  echo '--- Docker ---'
  command -v docker && docker version
  docker info --format '{{json .Runtimes}}'
  docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
  echo '--- network (inspection only) ---'
  ip -br address
  ip route
  echo '--- occupied listening ports ---'
  ss -lnt
} 2>&1 | tee "$OUT"
printf 'Report: %s\n' "$OUT"
```

## 文件：`scripts/reference.py`

<!-- FILE: scripts/reference.py -->
```python
#!/usr/bin/env python3
"""Run unmodified Shiu v630 model with sugar inputs and MN9 readout.

Notebook constants are parsed with ast.literal_eval, never notebook execution.
This wrapper is source-checked, not executed against full biological data in this delivery.
"""
import argparse
import ast
import copy
import csv
import json
import sys
from pathlib import Path


def notebook_constant(path: Path, name: str):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell["source"])
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                    return ast.literal_eval(node.value)
    raise KeyError(f"Notebook constant not found: {name}; inspect the pinned source.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default="upstream/Drosophila_brain_model")
    p.add_argument("--out", default="runs/reference-smoke.csv")
    p.add_argument("--seconds", type=float, default=.1)
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--rates", type=float, nargs="+", default=[0, 100, 150])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--codegen", choices=["numpy", "cython"], default="numpy")
    a = p.parse_args()
    if a.seconds <= 0 or a.trials < 1 or min(a.rates) < 0:
        raise ValueError("Invalid duration/trials/rates.")
    import brian2 as b2
    import pandas as pd
    repo = Path(a.repo).resolve()
    sys.path.insert(0, str(repo))
    import model
    comp = repo / "2023_03_23_completeness_630_final.csv"
    con = repo / "2023_03_23_connectivity_630_final.parquet"
    for path in [comp, con, repo / "example.ipynb"]:
        if not path.is_file():
            raise FileNotFoundError(path)
    sugar = notebook_constant(repo / "example.ipynb", "neu_sugar")
    mn9 = notebook_constant(repo / "example.ipynb", "id_mn9")
    ids = pd.read_csv(comp, dtype=str).iloc[:, 0].tolist()
    index = {int(value): i for i, value in enumerate(ids)}
    missing = [value for value in [*sugar, mn9] if value not in index]
    if missing:
        raise ValueError(f"v630 neuron IDs not in completeness table: {missing}")
    b2.prefs.codegen.target = a.codegen
    b2.defaultclock.dt = .1 * b2.ms
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["input_hz", "trial", "seconds", "mn9_spikes", "mn9_hz"])
        writer.writeheader()
        for rate in a.rates:
            for trial in range(a.trials):
                b2.start_scope()
                b2.seed(a.seed + trial)  # paired seeds across stimulation rates
                params = copy.deepcopy(model.default_params)
                params["t_run"] = a.seconds * b2.second
                params["r_poi"] = rate * b2.Hz
                activity = model.run_trial([index[x] for x in sugar], [], [],
                                           str(comp), str(con), params)
                count = len(activity.get(index[mn9], []))
                result = {"input_hz": rate, "trial": trial, "seconds": a.seconds,
                          "mn9_spikes": count, "mn9_hz": count / a.seconds}
                writer.writerow(result); f.flush(); print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
```

## 文件：`scripts/run_body.sh`

<!-- FILE: scripts/run_body.sh -->
```bash
#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$ROOT"/{runs,reports,cache/body-home}
exec docker run --rm --init --gpus all --shm-size 2g \
  --user "$(id -u):$(id -g)" -e HOME=/work/cache/body-home \
  -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -v "$ROOT:/work" -w /work flybrain-body:2.1 "$@"
```

## 文件：`scripts/run_core.sh`

<!-- FILE: scripts/run_core.sh -->
```bash
#!/usr/bin/env bash
# All paths passed to the contained command should be relative to this project root.
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$ROOT"/{data,runs,reports,cache,tmp,upstream}
ARGS=(--rm --init --gpus all --network host --shm-size 2g
      --user "$(id -u):$(id -g)" -e HOME=/work/tmp -e PYTHONUNBUFFERED=1
      -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
      -v "$ROOT:/work" -w /work)
for NAME in GLOO_SOCKET_IFNAME OMP_NUM_THREADS FLY_NODE_RANK FLY_NNODES \
            FLY_MASTER_ADDR FLY_MASTER_PORT; do
  if [[ -n "${!NAME:-}" ]]; then ARGS+=(-e "$NAME=${!NAME}"); fi
done
exec docker run "${ARGS[@]}" "${FLY_IMAGE:-flybrain-lab:0.1}" "$@"
```

## 文件：`scripts/summarize.py`

<!-- FILE: scripts/summarize.py -->
```python
#!/usr/bin/env python3
"""Aggregate evaluation JSON by mode, reporting seeds separately, not pooled pseudo-replication."""
import argparse
import glob
import json
import statistics
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("pattern", help="Quoted glob, e.g. 'runs/eval-*.json'")
a = p.parse_args()
groups = {}
for path in sorted(glob.glob(a.pattern)):
    item = json.loads(Path(path).read_text())
    key = item["mode"] if item["action_selection"] != "uniform_random" else "uniform_random"
    groups.setdefault(key, []).append((path, item["success_rate"], item["seed"]))
if not groups:
    raise SystemExit("No matching evaluation files.")
for key, rows in groups.items():
    values = [x[1] for x in rows]
    print(json.dumps({"mode": key, "runs": len(rows), "mean_success_rate": statistics.mean(values),
        "between_run_sample_std": statistics.stdev(values) if len(values)>1 else None,
        "files_and_scores": rows, "warning": "Check identical evaluation seeds and matched training budgets."}))
```

## 文件：`scripts/validate_local.sh`

<!-- FILE: scripts/validate_local.sh -->
```bash
#!/usr/bin/env bash
# CPU/software validation inside the core container or an existing compatible CPU environment.
# This is not a full-data/GPU/four-physical-node validation.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-"runs/local-validation-$(date +%Y%m%d-%H%M%S)"}
mkdir -p "$OUT"
python -m compileall -q flylab.py scripts tests
for file in scripts/*.sh; do bash -n "$file"; done
python -m pytest -q tests | tee "$OUT/pytest.txt"
python flylab.py --threads 1 prepare --source synthetic --neurons 512 --out "$OUT/synthetic.npz"
python flylab.py --threads 1 gpu-check --device cpu | tee "$OUT/cpu-spmm.txt"
python flylab.py --threads 1 train --graph "$OUT/synthetic.npz" --device cpu \
  --envs 4 --steps 16 --minibatch 32 --epochs 2 --updates 3 --out "$OUT/train"
python flylab.py --threads 1 eval --graph "$OUT/synthetic.npz" --device cpu \
  --checkpoint "$OUT/train/policy.pt" --episodes 5 --out "$OUT/eval.json"
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  flylab.py --threads 1 dist-check | tee "$OUT/gloo-check.txt"
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  flylab.py --threads 1 train --graph "$OUT/synthetic.npz" --device cpu \
  --envs 2 --steps 16 --minibatch 16 --epochs 1 --updates 2 --out "$OUT/distributed"
echo "Software-only validation completed: $OUT"
```

## 文件：`sources.lock.json`

<!-- FILE: sources.lock.json -->
```json
{
  "checked_date": "2026-09-18",
  "shiu_model": {
    "repository": "https://github.com/philshiu/Drosophila_brain_model",
    "commit": "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960",
    "default_paper_dataset": "FlyWire v630",
    "additional_engineering_dataset": "v783"
  },
  "fly_ai_preprocessing": {
    "repository": "https://github.com/alextitonis/fly.ai",
    "commit": "95a3dbcb05241b0a5c07028ca8ad945b23fbbe6e",
    "asset_release": "brain-v1",
    "brain.npz_sha256": "cc9bd1ecd00bd703a6fa648bc6ad145c93c7c1ee53debdcc9ce0d1f4305e6aca",
    "weights.npz_sha256": "c29919aa44069a271b1ee978abe05fa9bf6e45e4ba3e436e92b624ef1b5be40c"
  },
  "flygym": {
    "repository": "https://github.com/NeLy-EPFL/flygym",
    "tag": "v2.1.0",
    "commit": "ca65a510c2afe6ac61c51df4f274c8d190c2f95f"
  },
  "core_base_image": {
    "tag": "nvcr.io/nvidia/pytorch:25.11-py3",
    "digest": "Resolve on the user's Spark before building; not fabricated here."
  }
}
```

## 文件：`tests/test_flylab.py`

<!-- FILE: tests/test_flylab.py -->
```python
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import flylab as f


@pytest.fixture
def graph(tmp_path):
    path = tmp_path / "synthetic.npz"
    f.prepare_graph(argparse.Namespace(source="synthetic", neurons=256, seed=64,
                                      out=str(path), overwrite=False))
    return path


def test_id_precision_and_edge_direction(tmp_path, monkeypatch):
    import pandas as pd
    comp, con, out = tmp_path / "comp.csv", tmp_path / "con.parquet", tmp_path / "graph.npz"
    ids = ["720575940660219265", "720575940660219266", "720575940660219267"]
    comp.write_text("root_id,other\n" + "\n".join(x + ",x" for x in ids))
    con.write_bytes(b"Mock parquet; real parquet reader is not tested here.")
    monkeypatch.setattr(pd, "read_parquet", lambda _: pd.DataFrame({
        "Presynaptic_Index": [0, 1], "Postsynaptic_Index": [1, 2],
        "Excitatory x Connectivity": [2., -4.]}))
    f.prepare_graph(argparse.Namespace(source="shiu", complete=str(comp), connectivity=str(con),
                                      out=str(out), overwrite=False))
    w, actual = f.load_graph(out)
    assert actual.tolist() == ids
    assert w[1, 0] == 1 and w[2, 1] == -1 and w[0, 1] == 0


def test_male_converter(tmp_path):
    weights, meta, out = tmp_path / "weights.npz", tmp_path / "brain.npz", tmp_path / "converted.npz"
    w = sp.csr_matrix(np.array([[0, 2.], [-1., 0.]], dtype=np.float32))
    sp.save_npz(weights, w)
    np.savez(meta, ids=np.array(["10", "11"]))
    f.prepare_graph(argparse.Namespace(source="male", weights=str(weights), brain=str(meta),
                                      out=str(out), overwrite=False))
    got, ids = f.load_graph(out)
    assert ids.tolist() == ["10", "11"] and got[0, 1] == 1 and got[1, 0] == -1


def test_graph_normalization(graph):
    w, ids = f.load_graph(graph)
    assert max(np.asarray(abs(w).sum(axis=1)).ravel()) <= 1.00001
    assert w.nnz > 0 and len(ids) == 256
    assert json.loads(Path(str(graph)+".json").read_text())["synthetic"] is True


def test_overwrite_guard(graph):
    with pytest.raises(FileExistsError):
        f.prepare_graph(argparse.Namespace(source="synthetic", neurons=256, seed=64,
                                          out=str(graph), overwrite=False))


def test_reservoir_determinism_and_reset(graph):
    torch.set_num_threads(1)
    b = f.Reservoir(str(graph), 2, readout=32)
    obs = f.Maze(2, 10).observations()
    a = b.features(obs)
    b.reset()
    torch.testing.assert_close(a, b.features(obs))
    b.reset(np.array([True, False]))
    assert b.v[:, 0].abs().sum() == 0
    assert not b.w.requires_grad
    assert not a.requires_grad
    assert set(b.inputs.tolist()).isdisjoint(b.outputs.tolist())


def test_csr_matches_coo(graph):
    obs = f.Maze(2, 10).observations()
    csr = f.Reservoir(str(graph), 2, readout=32, layout="csr")
    coo = f.Reservoir(str(graph), 2, readout=32, layout="coo")
    torch.testing.assert_close(csr.features(obs), coo.features(obs), atol=2e-5, rtol=2e-5)


def test_controls_keep_mapping(graph):
    original = f.Reservoir(str(graph), 2, readout=32)
    shuffled = f.Reservoir(str(graph), 2, mode="shuffled", readout=32)
    none = f.Reservoir(str(graph), 2, mode="no_recurrence", readout=32)
    assert original.mapping_sha256 == shuffled.mapping_sha256 == none.mapping_sha256
    obs1, obs2 = f.Maze(2, 1).observations(), f.Maze(2, 2).observations()
    a = none.features(obs1); none.reset(); b = none.features(obs2)
    torch.testing.assert_close(a, b)
    original.reset(); a = original.features(obs1); original.reset(); b = original.features(obs2)
    assert (a - b).abs().max() > 0


def test_terminal_gae():
    r = torch.tensor([[1.], [2.]])
    v = torch.tensor([[.2], [.3]])
    done = torch.ones_like(r)
    advantage, ret = f.gae(r, v, done, torch.tensor([999.]), .99, .95)
    torch.testing.assert_close(ret, r)
    torch.testing.assert_close(advantage, r-v)


def test_maze_connected_and_deadline():
    maze = f.Maze(8, 2, horizon=4)
    seen, pending = set(), [tuple(maze.free[0])]
    while pending:
        xy = pending.pop()
        if xy in seen:
            continue
        seen.add(xy)
        for move in maze.MOVES:
            p = np.array(xy)+move
            if not maze.walls[p[1], p[0]] and tuple(p) not in seen:
                pending.append(tuple(p))
    assert len(seen) == len(maze.free)
    assert maze.observations().shape == (8, 9)
    for _ in range(4):
        _, done, _ = maze.step(np.zeros(8, dtype=int))
    assert done.all()


def test_policy_backward():
    model = f.Policy(64)
    policy, value = model(torch.randn(8, 64))
    loss = -policy.log_prob(torch.zeros(8, dtype=torch.int64)).mean()+value.square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_gpu_unavailable_is_error():
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="No silent CPU fallback"):
            f.checked_device("cuda")


def test_reference_notebook_parser(tmp_path):
    spec = importlib.util.spec_from_file_location("reference", Path(__file__).resolve().parents[1]/"scripts/reference.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    path = tmp_path/"example.ipynb"
    path.write_text(json.dumps({"cells": [{"cell_type": "code", "source": ["x = [1, 2, 3]\n"]}]}))
    assert module.notebook_constant(path, "x") == [1, 2, 3]


def test_cuda_device_without_index_is_resolved(monkeypatch):
    selected = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)
    assert f.checked_device("cuda") == torch.device("cuda:0")
    assert selected == [0]
```
