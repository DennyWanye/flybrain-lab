# Shadow V1 软件阶段完成报告

新增真实定位输入边界、观测记录与回放、完整 C2W 神经模型的无发送 Shadow 推理。全部主体代码由当前 session 主代理完成；没有子代理编码，没有安装依赖、训练新权重或运行历史大规模回归。

## 实际验收

- 针对性检查 **43/43** 通过：坐标变换与旧观测合同一致；错误单位/字段、非有限值、帧错误、时钟跳变、重复/缺帧、无效/过期测量；断流、积压、日志篡改/截断、空会话、模型时钟/mask 异常均有检查。
- 三组冻结模型 **11/22/33** 各进行两次完整 MaleCNS 小跑，每次 24 条观测、96 个神经子步；合计 144 条、576 子步。图为 **166700 个神经元、25582938 条边**。三组各自重复输出逐字节一致，使用实际神经特征，无训练或规则动作替代。
- CLI 直接 Shadow 与录制后回放，各 24 条新接口样本，其中 8 条异常被抑制；建议日志逐字节相同。重复输入以非零状态退出，部分数据与失败结果保留。
- 实际 stdin 进程验证：3 条按本机单调时间实时输入通过；停止供数后 1 秒看门狗终止；1 秒积压在推进神经前终止。输入是合成样本，尚无真实设备数据。
- 主神经探针开启 socket connect/sendto 审计：尝试数 0。业务模块没有 transport 或发送回调。所有记录 transmitted=false，真实适配器仍禁用。
- 基线 **199 份已有源码/权重/摘要逐文件哈希未变**；8 个记录 PID 均已退出。原 Viewer 保留。进程私有内存采样见 PROCESS_AUDIT.json，不把跨阶段采样当同时释放量。

## 使用与接口

源码：flydrone/shadow/inputs.py、recording.py、runtime.py、__main__.py。

使用说明：docs/shadow_v1.md。入口：`python -m flydrone.shadow --help`。提供 example、record、verify、shadow、replay、verify-shadow；所有输出要求新目录。已有 WSL 环境内可直接运行。

输入统一米/秒/弧度、Z 向上坐标与同步单调时钟。真实定位须由设备采集器提供。本版消费上游记录的任务阶段、阶段目标/锚点和实际动作状态；不声称已有真实飞行任务管理器。建议动作绝不写回“已执行”历史。

## 就绪状态与后续依赖

SHADOW_SOFTWARE_VERIFIED=true。REAL_POSE_VALIDATED=false；HARDWARE_CALIBRATED=false；HARDWARE_SHADOW_VALIDATED=false；REAL_FLIGHT_READY=false。

当前验证只证明软件接口、完整模型推理和无发送边界。没有连接无人机，没有测量定位精度，没有验证 SDK 在实机上保持高度/未操作轴的语义，也没有证明真实闭环任务成功。原 C2W 模拟验收状态维持不变。

下一步需要确认具体机型/固件及定位来源，再实现对应的只读采集器，取得可追溯的真实定位记录，冻结并执行定位/标定验收方案。详见 HARDWARE_ACCEPTANCE.md。涉及实际控制/飞行仍须单独授权。

## 可复核证据

focused-release.xml、fullgraph-smoke/summary.json、cli-probe/summary.json、live-stdin-probe/summary.json、HISTORY_AUDIT.json、PROCESS_AUDIT.json、SOURCE_MANIFEST.json。

本目录保留所有开发检查、合成观测、重复回放、预期失败和进程日志；它们不是新的模型泛化成绩。开发新鲜度预算 200ms（含默认 20ms 时钟误差）尚未获得硬件标定。
