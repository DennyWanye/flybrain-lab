# 近距离学习诊断：方法与复现

本轮只研究 C0 近距离控制，不宣称完成正式泛化验收或实机迁移。所有主体代码由当前会话完成；主功能 smoke 通过后才进行对应范围回归。

## 修复的信号问题

旧编码将平面目标误差除以 6。较小输入叠加在阈下 LIF 电流上时，可能没有足够脉冲传到随机选取的下游读出。原探针中 15 cm 四个方向的 40 次观测历史完全相同；30 cm 只得到两种历史。

新配置 balanced_rate_v2 使用 0.12 基础驱动和 0.88 的信号幅度，平面目标缩放至 1 m 尺度；按 34 个输入通道轮流选择连接强度较大的不同下游细胞，明确排除直接注入输入的细胞。仍只有膜电位与迹进入 actor/critic，无原始观测旁路。固定归一化采用 LIF 静息电位和指定尺度，不用验证集拟合统计量。这是一种工程感知接口，不是对果蝇真实感觉神经解剖映射的验证。

旧配置 legacy 保留。新配置改变了映射和编码，检查点会拒绝跨配置直接加载。

## 信号诊断的边界

静态方向探针使用 4 个方向、4 个距离，每次 40 个采样步。

另有三分类线性解码探针：区分目标内、前方目标、后方目标；加入独立速度、电量、剩余时间、上一次动作和持续时间干扰。128 个合成测量上下文拟合诊断解码器，64 个独立上下文评估。这个解码器不会保存为飞行策略，不用于 PPO 初始化。读出可解码不等于控制成功。

## 对照训练

- 课程 C0-near-x：固定高度，随机平移起点，目标位于正/负 X 方向 0.30–0.65 m，开放停止/前进/后退；动作掩码只依赖课程和传感器有效性，不依赖目标方向。
- 课程 C0-near-xy 已实现但本轮未扩大训练；原 C0 不变。
- 三个种子：11、23、37。每个模型 1536 次动作决策，rollout=128；同种子模型的初始权重与场景列表相同。
- 三个输入对照：真实果蝇读出、测量观测直达的小网络、脑特征置零。后两者在产物和界面明确标成诊断对照，不计作果蝇控制成果。
- 直接观测对照使用相同 128 维网络，前 26 项为测量观测（目标恢复为米），其余填零。置零对照全部特征为零。为了保持记录和时钟可核查，这些对照也推进真实脑状态，但 actor/critic 不读取它。
- 训练集与 validation 种子分离。使用同一批 8 个验证场景作探索性评估。300 个封存场景不参与本轮。
- 固定动作评估使用 argmax；概率动作评估使用固定随机种子采样。两者分开报告，不以较好的指标替换较差的指标。
- 多种子 × 相同 8 场景，不等同于 24 个独立场景的正式验收。

## 停止动作时长实验

balanced_rate_v3 保持 v2 神经映射和编码，将 STOP 的最小持续观察时间由 0.5 秒（实际首个回执约 0.6 秒）改为 2 秒，使一次 STOP 足够覆盖稳定保持的观察时间。该参数写入检查点兼容契约。未更改成功半径 20 cm、稳定保持 2 秒、60 秒期限或奖励系数。每个动作仍使用实际 k 和 Gamma=0.995^k。

这是单独的动作时长假设检验，保留 v2 的失败/成功结果，不把时长改变归因于神经特征改善。

## 命令

在 WSL 项目根目录 `/home/denny/projects/flybrain_lab_4spark_v0_1`，使用 Python `/home/denny/projects/flybrain-env/bin/python`。

```bash
python -m flydrone.tellosim.training.diagnostics --graph data/male-v1.npz --profile balanced_rate_v2 --out reports/new-signal-probe
python -m flydrone.tellosim.training.signal_probe --graph data/male-v1.npz --out reports/new-decoder-probe.json
python -m flydrone.tellosim.training.runner --graph data/male-v1.npz --device cuda --profile balanced_rate_v2 --curriculum C0-near-x --options 1536 --rollout 128 --eval-cases 8 --seed 11 --out runs/tellosim-sdk9/new-nearx-run
python -m flydrone.tellosim.training.evaluation --graph data/male-v1.npz --checkpoint runs/tellosim-sdk9/new-nearx-run/checkpoint.pt --out reports/new-nearx-evaluation.json
```

输出目录必须全新。`--init-from` 是显式 warm-start，优化器和环境状态重置，不是精确续训。`experiments` 与 `sampled_comparison` 可重现本轮固定矩阵；为避免单 GPU 多进程争用，后续默认顺序执行。实验日志包含实际参数、场景哈希和 checkpoint 哈希。

## 界面与限制

Viewer 增加了多种子对照表、明确区分两种动作选择方式，并修正“策略输入维数”的显示，避免把 26 维感知观测误写成 actor 实际读取的 128 维神经特征。对照模型回放有独立标注。

本轮没有绕过上一轮浏览器工具的安全拒绝，新增界面交互保持 NOT_VERIFIED；代码语法与数据/API 检查不能代替浏览器真人交互验证。实机适配器仍禁用。


## 明确分开的规则示范启动实验

短程纯 PPO 的最高概率动作评估持续失败，因此补充一个独立的受监督启动实验，不能替换或美化 PPO 结果。三个种子仍为 11、23、37，每个种子使用 32 个专家示范回合，再进行两轮、每轮 16 个模型自行执行的回合。后两轮只由专家给模型遇到的神经特征附加动作标签（DAgger），不替模型执行纠正动作。

专家只读取同一份测量观测与动作掩码：首个动作停止以等待感知/脑状态稳定，之后在测量距离 <=17 cm 时停止，否则按测量目标方向前进/后退。最终验证完全由学习后的模型决策。网络输入始终为冻结果蝇网络的 v/trace，没有将专家目标方向直接接到执行器。

每轮固定 60 个训练 epoch，按类别加权交叉熵训练动作读出及其共享隐藏层；价值头未接受有效训练，不可直接当成已训练的 PPO 价值估计器。PPO 更新次数为 0。数据种子 300000 + seed*1000 起始，与 validation 分离；保存 demonstrations.npz、完整数据哈希、初始/最终检查点、各阶段收集成功数及最终验证和脑特征置零结果。没有依据验证成绩挑选最优 epoch 或最优种子。

该实验复现入口：

```bash
python -m flydrone.tellosim.training.bootstrap --graph data/male-v1.npz --tag NEW_TAG
```

后续可从该 actor 做显式 warm-start 的 PPO；需重新训练 critic，并验证 PPO 更新不会破坏已学行为。不能把“专家能飞”算成“模型能飞”，所以最终成绩仅取独立的学习模型验证回合。

对照采用同一预算和优化配置，未为直接观测网络单独调参；脑特征置零是功能性消融，不是随机连接图对照。这些结果不能证明果蝇连接组普遍优于普通神经网络或其他连接图。
