# Golden Episode 验收报告

日期：2026-09-26。依据：Golden_Episode_WorkAgent_Test_Handoff.md。

GOLDEN_EPISODE_READY = YES
MODEL_READY_FOR_NEXT_STAGE = NO

## 可直接打开的回放

http://127.0.0.1:8765/?view=golden-episode

当前正式验收数据包：`artifacts/golden_episode_verified/`。
Viewer 派生产物：`reports/vis/views/golden-episode/`，已注册到 `reports/vis/views/index.json`。
原始 `artifacts/golden_episode/` 的 replay、checkpoint 和数据文件保留；本次只给其旧报告增加版本说明。

## 真实轨迹与来源

- 基础提交：`cd32d73988cae500476981445f7923f9f03e014b`；本轮源代码改动尚未提交，逐文件 SHA256 见 `reports/GoldenEpisode_PROVENANCE.json`。
- 环境：WSL Ubuntu，Python 3.12.3，PyTorch 2.14.0+cu130，MuJoCo 3.3.7。录制和评估实际使用 CPU，4 个 OpenMP/MKL 线程，没有声称使用 GPU。
- checkpoint：`b657cb32ada93c8cee927e72277440702a53d74d2a16b20f3a5ef1cd77d51614`。沿用已有 imitation checkpoint；本轮没有重训或手改模型动作。
- graph：`badc33a247894fe12c4d68d3ff791393857cffa39ff2ab81869608a11fa1e0c9`。
- mapping：`623b691045f9c971271b8e48cc310dfeaaedb2a67b1bc639499bcfb24ea86d18`，通过记录的输入／输出索引重新计算验证。
- scenario：`room6_start_neg15_neg10_target_pos15_pos10`；seed=11。
- 起点 (-1.5, -1.0, 1.2)m，目标 (1.5, 1.0, 1.2)m，初始 yaw=0。
- 初始位姿由 reset 设置，首个动作进入 airborne 状态；本任务没有模拟起飞流程，不连接真机。
- 成功：True；步数 33；时长 65.166667s；累计回报 3.687714149。
- 最终距离 0.266170460m；连续稳定保持 1.716666667s；碰撞 false，越界 false，超时 false。
- 成功合同：距离 <=0.30m、速度 <=0.15m/s、连续保持 >=1s。逐个 1/120s 物理步重新计算，不再用动作时长计划累加。
- 冻结 LIF reservoir：N=166700，nnz=25582938。保存全脑 spike fraction、按 trace 排序的 Top-32（graph 行索引）、64 个选定读出神经元 v/spike/trace。
- 每个决策进行4个神经子步，只记录最后子步 index=3；其余子步、脑区标签和拓扑明确标为 `not_recorded`。没有伪造全脑实时动画。
- 记录26维遥测、实际8维脑输入、14维编码输入及128维策略输入。脑输入实际来自模拟 pose，不宣称已经实现噪声遥测隔离。
- 9动作合同，33条 SDK 命令，无 safety override；分别保存 proposed/approved action、命令参数、执行结果及起止时间。
- verified replay SHA256：`08bebe5164ae67a8b66577d17c88552a5d6aed39e5ca386977d5d05ea2e04a7e`。
- 原始 replay SHA256：`6ca963dbeb57f8dee7f417deab655e45a1b287750f4e16e85c45aeadf03a7537`，本轮操作后不变。

原报告的2秒保持时间来自旧的预设时长累计。新记录按真实物理步得到1.7167秒，仍满足合同。动作和最终位置未变；时间惩罚改按实际持续时间，因此累计回报也相应修正。

## 已执行验证

- GE-01～GE-15：见 `reports/GoldenEpisode_ACCEPTANCE.json`。本次加入实际浏览器证据；此前仅检查文件存在或哈希的结果不再代表浏览器通过。
- 数据校验：成功与保持时间重算、动作来源、时序、step关联、奖励和、概率范围与归一化、命令参数、物理步连续性、checkpoint/graph/mapping哈希。
- 浏览器全程2倍速播放到第32步 success；支持0.25、0.5、1、2倍速。
- 实际验证播放／暂停、慢放、鼠标拖动时间轴、画布平移／滚轮缩放／复位。暂停前后世界时间和位置一致。
- 第10步→第32步→第10步，观测、命令、策略输入、奖励和神经面板内容完全一致。轨迹仅33步，所以用末步32替代Handoff示例中的80。
- 抽查20个决策步；页面观测、命令、输入特征、神经汇总与原始记录精确相等，概率和奖励显示值与原记录一致。源记录全33步均验证时间/ID及奖励和。
- 在播放中关闭页面，服务 `/healthz` 仍正常，原始及verified replay、导出events哈希均不变。验证范围是录制回放；没有并发训练进程，因此未声称做过Live训练断开测试。
- 浏览器 console error：0。
- Python 回归：`tests/flydrone` 共14项通过（包含5项新增Golden/分页测试）；`git diff --check` 通过。主体代码和最小闭环完成后才运行此回归。
- 浏览器证据绑定 replay/events/Viewer源码哈希；修改页面后旧证据不能直接使验收变绿。

证据：`reports/golden_ui/START.png`、`MID-FLIGHT.png`、`SUCCESS-HOLD.png`、`sampled-20-steps.json`、`browser-observations.json`，以及 `reports/GoldenEpisode_UI_ALIGNMENT.json`。

## 多场景评估

预先固定 `configs/tellosim/golden_evaluation.json`，SHA256 `8a347ba513df85b3d0368c56eaf136609893b04dd2317d4518227b97e20b12be`。
当前checkpoint、固定高度、无障碍／无噪声，32个诊断case，全部保存于 `reports/golden_evaluation_final/episodes.jsonl`，包括逐决策动作、位置、距离、奖励和保持时间。结果没有用于更新权重。

- 训练分布附近：7/8成功。
- 反向目标：0/8成功。
- 横跨房间：0/8成功。
- 短距离目标：1/8成功。
- 总计：8/32，成功率 25%；平均回报 -0.606449677。
- 碰撞 0，越界 24，超时 0。

首轮诊断与补充逐决策证据后的最终运行结果一致，两次均为同一32个case，不能合并当作64个独立case。
诊断预设阈值为成功率>=90%、碰撞/越界率<=1%，本次未通过。
此外，本轮也不满足TS1正式三训练seed、每seed300封存case及随机基线比较的要求，不能替代正式阶段验收。
因此 MODEL_READY_FOR_NEXT_STAGE = NO。单条Golden通过只证明可观察、可回放与数据闭环，不代表任务稳定学会。

## 真实执行命令

在 `/home/denny/projects/flybrain_lab_4spark_v0_1`，使用 `/home/denny/projects/flybrain-env/bin/python`：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m flydrone.tellosim golden-record --checkpoint artifacts/golden_episode/checkpoint.pt --scenario configs/tellosim/golden_episode.json --brain-graph data/male-v1.npz --out artifacts/golden_episode_verified --device cpu
python -m flydrone.tellosim golden-export --episode artifacts/golden_episode_verified
python -m flydrone.tellosim golden-validate --episode artifacts/golden_episode_verified --brain-graph data/male-v1.npz
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m flydrone.tellosim golden-evaluate --checkpoint artifacts/golden_episode/checkpoint.pt --brain-graph data/male-v1.npz --cases configs/tellosim/golden_evaluation.json --out reports/golden_evaluation_final --device cpu
python -m flyview serve --project-root /home/denny/projects/flybrain_lab_4spark_v0_1 --host 127.0.0.1 --port 8765
OMP_NUM_THREADS=4 python -m pytest tests/flydrone -q
```

录制／评估输出目录不允许覆盖封存结果；复跑时请换新输出目录。导出命令可重复执行，保留registry其他视图。
生成的Viewer目录和checkpoint符合仓库既有ignore规则，未强行修改这些规则；可由上述命令重建Viewer。
