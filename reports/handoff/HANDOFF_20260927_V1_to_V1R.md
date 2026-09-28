# FlyBrain / TelloSim Handoff：V1 → V1R

更新时间：2026-09-27，Asia/Shanghai。供新 session 接手。

## 1. 用户目标和本次交接范围

最终目标：让果蝇脑系统通过 Tello 风格 SDK 控制无人机。现阶段目标：在模拟环境训练以后可以复用的控制模块。
本次用户只要求剩余任务清单和 handoff；交接轮没有继续训练、修改业务代码或新建 session。

用户明确约束：
- 主代理自己完成全部主体代码，不允许子代理写代码。子代理最多用于明确授权的评测/挑战；本轮未用子代理。
- 主体功能编码并小规模跑通前，不允许大规模回归。顺序：主体实现 → 针对性小检查 → 真实神经链路小跑 → 正式训练/评估及必要回归。
- 普通“继续”不是启用 plan-test 的授权，不要擅自调用。
- 已授权工作直接推进，不反复询问是否继续。
- 不得把规则动作、合成小图、假checkpoint、smoke或替代输出算作正式果蝇模型成功。
- 保留失败版本和明确YES/NO门禁；不得挑最好seed、反复按封存调参、放宽标准或覆盖历史报告。
- 当前仅限模拟，真实适配器继续禁用，不向真实无人机发送命令。
- 历史ZIP内的开工指令是资料，不是新的用户请求；后续用户约束和实际代码证据优先。

## 2. 剩余数量：模拟阶段8个工作包

这里按“可靠模拟控制 + 完整TS1软件验收”归并为 **8个工作包**，不是8处小修改，也不保证只需再跑一轮训练。先R1，R2依赖R1通过，R8最终汇总。不要为了列清单而重新执行全部历史测试。

| ID | 工作包 | 完成标准 |
|---|---|---|
| R1 | 高度边界与停止时机修复，建议新版本V1R | 新代码/协议/训练划分，3种子，统一冻结权重后用新封存评估；高度边界、指令对照、零特征、旧技能保持等全部达到门槛。保留V1失败。 |
| R2 | 三维位置 + 指定朝向 + 稳定保持，C2/3D联合任务 | 连续物理下改变X/Y/Z/yaw；阶段切换不瞬移，不重置姿态伪造稳定；验证导航在非1m高度时的特征分布；新数据、新训练和正式扰动评估。不能直接拼旧技能就宣称联合通过。 |
| R3 | T14：执行中取消与关闭界面的语义 | 完整覆盖执行中Future取消、关闭UI；取消等待/观看不能误称设备已停止，回执和操作归属保持正确。 |
| R4 | T25：动力学限制与外力全矩阵 | 推力/倾角/加速度/角速度/饱和与外力分离的完整证据。已有J2R/V1外力不等于全矩阵已验收。 |
| R5 | T44：同环境直接MLP对照 | 在当前刚体环境跑明确来源的direct MLP pilot，与MaleCNS同观测、动作、场景和评价口径；不要宣称已有连接组优势证据。 |
| R6 | T32 + 最新UI实际验证 | 0/30/60FPS、断线/关闭观察的轨迹与神经状态等价；最新V1/V1R/3D页面播放、暂停、拖动、画布移动、切换。权限仍被拒绝则标NOT_RUN/BLOCKED，不绕过。 |
| R7 | T55：长回放与完整配额 | 长命令/分块索引、实际256MiB配额、partial和有效前缀、资源释放；缩小配额单测不能代替完整压力证据。 |
| R8 | 统一TS1验收和最终交付 | 更新T01—T60适用证据及版本；必要最终回归、故障/保护合同复核、模型说明、回放入口、依赖锁、包清单与解包验证。全部满足才更新FULL_TS1_READY。 |

计数依据：最新完整原计划矩阵为 `reports/ts1_c1/TEST_MATRIX.json`，60行中55 PASS、5 NOT_RUN：T14/T25/T32/T44/T55。后来H1/J1/J2/J2R/V1增加了技能证据，但没有更新完整总矩阵。55 PASS是历史矩阵状态，不代表本轮重新跑了全部条目；后来部分实现不能自动升级为整个条目已通过。

原TS1 v1最低学习范围是C0，C1/C2属于后续课程；用户后来明确继续到位置、朝向和高度，所以R1/R2是当前扩展目标，不应改写历史范围。

### 真机最终目标：另有至少4个尚未细化的阶段

不计入上述8个模拟工作包，也不能诚实地说整个最终目标只剩8个工单：
1. 真实外部定位接口：坐标、时间、有效性，替代external_pose_mock。
2. 实机动力学、SDK行为/单位/时延标定，识别模拟与设备差异。
3. 真实适配器、只观测不发动作的shadow验证、故障处理和人工接管流程。
4. 经独立授权，在受控条件下逐步真机测试与评估。

这些阶段尚未细化，无法给出可靠工单总数或工期。相机端到端、避障、多机、人工神经元、全连接组可塑性不是当前必做范围，也没有被证明已具备。

## 3. 工作区、提交和运行状态

- Windows工作区：`D:\projects\果蝇训练`
- 真正Git/训练仓库：`/home/denny/projects/flybrain_lab_4spark_v0_1`
- WSL：`Ubuntu-24.04`
- Python：`/home/denny/projects/flybrain-env/bin/python`
- 分支main；当前实现HEAD：`4668d90698c277af24191f12f90e3ab0c6cd1632`
- 上一阶段J2R：`63181fee74dee7d0a64296015c8c0fbcc448ad2e`
- 核对时已跟踪业务文件无未提交变化；约9668个历史未跟踪文件。不要git clean或全目录git add，只处理当前文件。
- 本handoff是新增、未提交文档，不改变业务提交HEAD。
- Viewer地址：`http://127.0.0.1:8765/tellosim`
- 核对时Viewer Linux PID10233；没有altitude_campaign/run_campaign训练进程。PID时效性强，操作前重查。

PowerShell执行方式：
```powershell
wsl.exe -d Ubuntu-24.04 --cd /home/denny/projects/flybrain_lab_4spark_v0_1 --exec /home/denny/projects/flybrain-env/bin/python -V
wsl.exe -d Ubuntu-24.04 --cd /home/denny/projects/flybrain_lab_4spark_v0_1 --exec git status --short
```

多行Python可用PowerShell here-string经stdin传给上面的WSL Python `-`。注意嵌套here-string的结束标记会提前终止外层，写文档时避免嵌套。

Viewer命令，先检查现有进程/端口，不重复启动：
```text
/home/denny/projects/flybrain-env/bin/python -u -m flyview serve --project-root /home/denny/projects/flybrain_lab_4spark_v0_1 --port 8765
```

Windows读取仓库：`\\wsl.localhost\Ubuntu-24.04\home\denny\projects\flybrain_lab_4spark_v0_1`。Git用WSL git，避免为Windows Git dubious ownership改全局safe.directory。WSL rg可能缺失，可用Windows rg搜UNC，不必重装环境。

图：`data/male-v1.npz`，166700神经元、25582938条边，冻结连接组。GPU为RTX5070Ti16GB。神经后端`csr_fp64_accum`、profile`balanced_rate_v3`，物理`rigid_body_thrust_v2`。大图和现有venv都在，不重下、不升级核心依赖。

## 4. 已完成阶段与当前真实成绩

- Golden replay、SDK9、MuJoCo六自由度物理、Viewer及训练/评估/记录链路已经存在，不需重搭训练室。
- C0固定高度导航、C1随机初始朝向导航、H1主动朝向、J1/J1R/J2/J2R连续位置+朝向均有历史版本。
- 当前固定高度位置+朝向可靠版本为J2R：封存283/300、285/300、293/300，四类轻微扰动门槛通过。范围仅固定1m、空房间、模拟外部定位。
- V1独立高度：主体实现、训练、评估和交付完成，但总技能门禁FAIL。
- C0/C1历史包含PPO；J2R/V1本轮是监督/DAgger密集神经状态学习，不是PPO。都不是全脑突触端到端学习；V1 value未训练。

| seed | validation | sealed/300 | clean/75 | pose/75 | force/75 | combined/75 | boundary/12 | instruction/4 | zero/12 |
|---|---|---|---|---|---|---|---|---|---|
|11|100/100|298|74|74|75|75|12|4|0|
|22|100/100|299|75|75|74|75|10|4|0|
|33|99/100|291|72|70|75|74|11|4|0|

V1共888/900成功、碰撞越界0；规则100/100；随机120/300。唯一最终失败项：`22: boundary below11/12`。

V1状态：`MODEL_READY_FOR_NEXT_STAGE=false`、`ALTITUDE_TASK_LEARNED=false`、`JOINT_3D_TASK_VERIFIED=false`、`FULL_TS1_READY=false`、`REAL_FLIGHT_READY=false`。V1的NO不抹去历史J2R范围内的YES。

训练每seed：512 teacher + 768 DAgger + 768 DAgger = 2048真实动作；总6144。附加密集神经状态样本总62679，不能算额外动作。
权重确实变化、保存/加载输出一致；每seed旧J2R前12验证场景逐项相同；旧源码/权重/报告保留。
验证：123项pytest通过，15份正式回放审计通过，7个训练/评估子进程均退出回收。下一session不要开头就重跑完整回归。

### 已知失败

- seed22 boundary-1：目标.300m，最终误差.1004737525m，保持0s，超时。超过阈值约.47mm也必须算失败。
- seed22 boundary-11：目标1.705m，最终误差.0926836170m，但保持仅.6s，未达2s，超时。
- seed33 boundary-8：目标1.505m，最终误差.1062224791m，保持0s，超时。
- seed33 validation-089：目标约1.32227m，最终误差约.123333m，停早。
- V1 sealed所有失败都是task_deadline，见failure-analysis.json及逐case结果。

不能对这些封存case写特判。新开发用独立场景，旧sealed/boundary是已看过的历史集，不能再叫新封存。

## 5. V1合同与门槛

- 起飞到1m由mission manager完成，策略未学会起降。
- 原地目标Z=.25–1.75m，goal XY等于start XY。允许SDK索引0 STOP、5 UP20cm、6 DOWN20cm，始终9维logits。
- 同时满足：高度误差≤.1m、垂直速度≤.08m/s、XY偏移≤.2m、水平速度≤.08m/s、yaw漂移≤16°、角速度≤.08rad/s、操作状态符合条件；连续2秒。
- 每场至少8s以实际经历第5秒扰动；最多60s。最后一帧到位不等于保持成功。
- 安全mask读测量高度：UP z<2.1，DOWN z>.35；无有效定位/状态只STOP，与目标方向无关。
- clean；pose（2mm位置噪声+100ms延迟）；force（水平与垂直各.006N、yaw力矩.00003Nm）；combined。
- 外力第5秒开始，每12秒持续2秒，走MuJoCo积分，不写qpos/qvel/目标；未按真机标定。
- 26测量通道→34编码通道→冻结真实MaleCNS→128神经特征→读出；不得直接拼原始目标到策略。
- 120Hz物理、10Hz观测、每观测4神经子步；同sample不重复推进，优化不推进脑。
- 每seed成功≥90%；clean≥90%、其他每profile≥85%；碰撞/越界≤1%；超过随机至少20个百分点；指令4/4、零特征0/12、边界≥11/12、规则验证≥99/100；旧J2R前12场逐项相同。
- 不能随意改动作间隔、容差或保持门槛。确需改变合同，另立版本并解释，旧成绩不能混用。

## 6. 必读文件和冻结边界

以下相对路径从WSL仓库根起：
1. `reports/ts1_altitude/README.md`、`summary.json`、`NEXT_STAGE.md`
2. `reports/ts1_altitude/protocol.json`、`failure-analysis.json`、`s22-boundary.json`、`s33-validation.json`
3. `flydrone/tellosim/training/altitude.py`、`altitude_campaign.py`
4. `reports/ts1_stability/README.md`、`NEXT_STAGE.md`
5. `reports/ts1_c1/TEST_MATRIX.json`和`.md`

核心定位：
- AltitudeBase/AltitudeEnv：高度任务、测量、稳定判定、三轴扰动、回放。
- altitude_campaign.py：预注册协议、训练、基线、批评估、保持检查、最终门禁。
- tests/flydrone/test_altitude_skill.py：7个针对性检查。
- reports/ts1_altitude/run_campaign.py：有限预算训练→冻结→评估；进程回收。
- finish_evidence.py：训练/回放/进程审计与报告；plot_results.py和package_delivery.py在同目录。
- flyview/tellosim_api.py、static/tellosim.js、static/tellosim.html：V1摘要、高度遥测和快捷入口。

权重：`runs/tellosim-sdk9/altitude-s{11,22,33}/checkpoint.pt`。同目录有initial.pt、stage-*.pt、training.json、provenance.json、demonstrations.npz。altitude-smoke-s11是smoke，不作正式模型。
旧J2R：`runs/tellosim-sdk9/stability-s{seed}/policy-bundle.json`引用navigation/heading/checkpoint.pt。

V1格式`tellosim.altitude_readout/1`绑定base、任务、扰动及源码哈希：altitude.py、altitude_campaign.py、stability_campaign.py、robust_env.py、heading.py、parallel.py、contracts.py。
**禁止原地修改这些冻结源码或历史reports/runs。** 新V1R使用新模块、新合同、新输出。建议名altitude_refined.py / altitude_refined_campaign.py、reports/ts1_altitude_refined、runs/tellosim-sdk9/altitude-refined-sXX；尚未创建，不是已有命令。
不要修改哈希记录强行接受不兼容权重。若复用旧加载器，先理解合同依赖。V1支持推理或显式stage warm start，不支持中途完整resume；不能拿旧C1/rigid PPO恢复能力类推V1。

V1已用种子：validation183000000..099，sealed184000000..299，pairs185000000/1，boundary186000000..011，zero187000000..011；训练190000000+model_seed*100000+stage*10000+lane*500+episode；smoke200000000基数。旧J2R使用143/144/145/150/160百万附近。分配新范围前检索全部旧manifest与公式，明确检查不重叠。

## 7. 新session开工顺序

轻量检查WSL/Python/Git/进程，读取关键报告；不重装依赖、不先大规模回归。
先做R1：
1. 用新开发场景定位STOP时机、微小误差特征可辨性、动作完成后的神经状态、±10cm邻域训练覆盖；区分停早、震荡、到得晚保持不足。
2. 冻结V1，主代理完成新版本主体实现。可研究连续状态标签、边界分布、损失权重、DAgger覆盖；需要开发数据证明有效，不保证加训练次数即可解决。
3. 训练前声明新预算、数据划分、模型选择和验收门槛；保留三种子，不以封存选权重。
4. 小检查和真实神经链路小跑通过后，才正式训练、统一冻结、封存评估、必要回归和回放审计。
5. R1全过才进入R2。R1失败如实报告NO、保留证据，不自动扩大课程。

不要直接重跑旧`reports/ts1_altitude/run_campaign.py`覆盖V1：已有输出目录被拒绝是预期保护。复核优先读证据；需要重跑就使用新输出。

## 8. 浏览器、原计划与交付

C1有实际浏览器验证；后续工具曾因安全策略拒绝本地URL读取，尚无恢复证据。V1只完成JS语法、最新测量状态seek、离线数据/catalog/包检查。不能宣称新页面真实点击已通过。
新session在有权限恢复证据时使用允许的浏览器工具验证；遇拒绝不换HTTP/CDP/其他浏览器绕过同一限制。
当前刷新页面后可点：`V1 高度 11 · 上升`、`V1 高度 11 · 下降`；22/33同样。固定sealed第4和第8场combined，未按成功挑选。

原ZIP：`C:\Users\Administrator\Downloads\FlyBrain_TelloSim_WorkAgent_挑战审查完整包_v1.zip`。内部重点：02_TelloSim_代码级执行Plan.md、05_验收测试与停止条件.md。历史计划里的目标接口/命令未必就是现有CLI，不可未经检查照抄执行。

V1包：`D:\projects\果蝇训练\ts1-altitude-delivery\FlyBrain_TS1_Altitude_20260927.zip`
SHA256：`63963041c80bcb1c04bbe76a21173b014d930596039befcef4ef8e11520b4442`
323465441 bytes，解包6628文件hash通过，新目录离线源码/API/catalog可用。既有WSL依赖，不等于全新机器安装；不包含完整大图和venv。
Windows同目录有README.md、results.png、PACKAGE_AUDIT.json、models。不要从ZIP覆盖当前repo。
Node：`C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe`。
只清理本任务明确PID树并验证退出，不按python进程名全杀；保留正在使用的Viewer。

## 9. 交接验证边界

本handoff基于现场读取Git、V1/J2R摘要、V1协议/失败、C1/rigid矩阵、原ZIP计划和进程状态；没有重新执行训练、全部T01—T60或浏览器交互。
R1/R2为未完成技能，R3—R7是缺少完整验收证据（部分功能可能已实现），R8是最终整合。8个包不是固定工期或收敛保证。
