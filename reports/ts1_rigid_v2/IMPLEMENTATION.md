# 本轮实现与边界

本轮范围：TS-G2 六自由度飞控、TS-G3 安全 rollout 边界恢复、1/4 环境训练、C0 在新物理下的迁移与限额适配。全部主体代码由当前主会话完成，无编码子代理；主体路径跑通后才运行全套回归。

## 物理

新 profile 为 `rigid_body_thrust_v2`。重力、三轴位置和四元数由 MuJoCo 积分；水平移动来自机身倾斜后的推力，转向来自受限力矩。SDK 收令时锁定目标，三角形/梯形速度参考受加速度约束；控制器跟踪参考，不写入 qpos/qvel 移动。reset 和精确状态恢复是明确例外。运动完成需连续稳定 0.25 秒。回放直接显示记录的四元数；SDK 状态包包含实际模拟 pitch/roll 与机体系速度。

旧 `bounded_level_body_surrogate` 保留，用于历史模型和历史证据复现。旧 C0 的 298/292/292 不转记为新环境成绩。飞控仍是未标定的工程控制器，没有螺旋桨气动、风场或真实电机标定。

## 存档与四环境

`continuation.py` 支持 batch=1/4，在 PPO 更新结束的安全边界保存 `resume.pt`。保存 MuJoCo integration state（含 warmstart）、轨迹参考、控制器、操作记录/队列、传感器延迟历史、各 RNG、各脑 v/s/trace、优化器与计数。恢复依赖相同软件、设备、配置和执行合同。`checkpoint.pt` 是推理/显式热启动产物，与 `resume.pt` 区分。

四个世界和传感器分别推进，同一冻结连接图批量计算神经状态；各环境独立 reset。策略采样使用被完整保存的全局 Torch RNG；尚未提供每环境单独的策略随机数流，这一项不冒充完整 T33 通过。训练阶段不写大回放，评估单独导出 lane 0 的记录；目前界面不支持四个训练 lane 的实时切换。

## 数值恢复修复

float32 CSR 和 COO 路径在重复对照中出现少量神经元 1 ULP 左右差异；动作、权重、物理和随机数一致，仍判定严格恢复失败，保留失败 checkpoint。新恢复通道使用显式版本 `csr_fp64_accum`：原 float32 边权精确转存 double 累加，结果转回 float32 LIF 状态；原连接组和边权数值不变。完整 166700 神经元 / 25582938 边，与独立 SciPy float64 参考比较误差为 0。相同保存边界的连续/恢复对照及三次重复恢复全部逐项一致。测试保证限于锁定的执行合同和已测轨迹，不声称任意硬件浮点结果逐位一致。

只有新的 FP64 checkpoint 支持本轮交付的恢复合同。先前诊断 CSR/COO 恢复文件保留用于审计，不应作为可恢复训练交付。旧格式不被静默升级。

## 训练来源

三个候选继承旧 C0 的三个独立读出权重；都是同一个冻结 MaleCNS 图。每种子新增 256 规则示范动作、256 DAgger 动作、128 PPO 动作，并在最终 FP64 后端显式热启动追加 32 个 PPO 动作，共 672 个。中间切换后端会重置优化器，不能称为同一训练中途无缝恢复。示范只用于训练采集，正式评估动作仅由保存后的神经策略产生。

开发验证使用 seed 5100000..5100099，封存测试 5200000..5200299；新数据与旧封存集不同。最终三个 checkpoint 在打开封存结果之前统一冻结，之后不再调参。旧权重直接迁移得到 93/100、89/100、85/100。中间 COO 适配验证被最终候选替代并显式中断，不计作正式成绩。

性能文件记录 1/4 环境的短测、内存峰值与成本。短测受加载、存档和并发任务影响，不能据此宣传稳定加速或推算已通过长时性能门槛；尚未执行 10000 动作的长时吞吐验收。

## 仍属于后续范围

C1 随机朝向、C2 高度变化、风/系统化动力学扰动、四环境实时切换和原 TS1 60 项全矩阵的 NOT_RUN 项。真实设备仍由 RealDeviceDisabled 禁用。下一阶段应按验收结论推进；本轮不宣称完整 TS1 或真实飞行已就绪。

数值实现参考：
- https://mujoco.readthedocs.io/en/latest/programming/simulation.html
- https://github.com/google-deepmind/mujoco/blob/main/doc/overview.rst
- https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/sparse/cuda/SparseBlasImpl.cpp
