# TS1 逐项验收

PASS是该行范围的证据，不等于完整TS1已就绪。历史证据位于上一版报告；本轮新增验收见本目录。null耗时未补造。

| ID | 状态 | 输入/范围 | 实际结果 | 证据 |
|---|---|---|---|---|
| T01 | PASS | 合法/非法 wire 与 typed 往返 | 合法/非法向量含 typed-wire 往返；原 approved_command_vectors 已覆盖，复核代码后纠正上一轮保守 NOT_RUN | pytest.xml / test_approved_command_vectors |
| T02 | PASS | typed 类型与 JSON 语义拒绝 | typed float/bool/NaN/Inf 均拒绝，重复 JSON 键/NaN API 拒绝；原测试已有完整参数，本轮复核通过 | pytest.xml |
| T03 | PASS | 握手/地面/重复起降状态 | 未握手起飞、地面移动、重复起降均拒绝且状态/位置不伪改 | sdk-additional.xml |
| T04 | PASS | 20/500 cm 与 speed 10/100 | 20/500cm × 10/100cm/s 四组合实际连续轨迹完成 | sdk-additional.xml |
| T05 | PASS | go 20 拒绝、21 允许 | 合同向量拒绝 20；physics-edges.json 中 go21 实际执行 | ../ts1_rigid_v2/approved_command_vectors |
| T06 | PASS | yaw0/90 下坐标与锁定目标 | 新物理四元数、目标和转向后移动对照通过 | ../ts1_rigid_v2/rigid_body_route_rotates |
| T07 | PASS | cw/ccw 实际 360° | 两个方向完整转动、非零耗时 | ../ts1_rigid_v2/physics-edges.json |
| T08 | PASS | 运动期间查询/busy | 查询不移动目标；忙请求与回执归属通过 | ../ts1_rigid_v2/go_query_share |
| T09 | PASS | request 丢失 | 不执行，不重试，unknown | ../ts1_rigid_v2/request_and_reply_loss |
| T10 | PASS | reply 丢失 | 执行但客户端 unknown，不重试 | ../ts1_rigid_v2/request_and_reply_loss |
| T11 | PASS | 迟到 ack 与后一请求 | 旧延迟回执不会替代 stop 或后续动作，新的运动仍 sent | test_c1_reliability.py |
| T12 | PASS | request_id 本地幂等 | 同一本地请求只执行一次，不声明原生 wire 幂等 | ../ts1_rigid_v2/lost_ack_idempotency |
| T13 | PASS | 运动中保护 stop | 停止与旧请求状态回归通过；模拟能力 | ../ts1_rigid_v2/lost_ack_idempotency |
| T14 | NOT_RUN | 取消 Future/关闭 UI 不等于停止 | 取消排队请求已测；执行中 Future 和关闭 UI 全矩阵未测 | ../ts1_rigid_v2/delayed_channel |
| T15 | PASS | 15s watchdog 与查询/busy 计时 | 真实 loopback fixture 验证 idle15s、query 重置与超过15s的忙动作不误降落 | sdk-additional.xml |
| T16 | PASS | 不支持命令无副作用 | 不支持命令拒绝，真实设备不开放 | ../ts1_rigid_v2/approved_command_vectors |
| T17 | PASS | loopback 与真实 IP 拒绝 | 地址限制与真实 IP JSON 拒绝通过 | ../ts1_rigid_v2/loopback_address |
| T18 | PASS | inprocess/UDP 同列表逐项等价 | 同一17条wire序列的直接执行与本机UDP回执相同，终点误差在5mm内 | sdk-additional.xml |
| T19 | PASS | 自由落体/推力悬停 | 重力积分和悬停通过，不靠覆盖 qpos 移动 | ../ts1_rigid_v2/preflight.json |
| T20 | PASS | 连续起飞/转向/平移/降落 | 8 命令全 ack、无碰撞；真实轨迹可回放 | ../ts1_rigid_v2/preflight.json |
| T21 | PASS | 5 扰动场景悬停 10s | 位置和高度误差均在 0.1m 内 | ../ts1_rigid_v2/preflight.json |
| T22 | PASS | 六向/go/整圈/大房间 | 大场景范围和稳定窗口检查通过 | ../ts1_rigid_v2/physics-edges.json |
| T23 | PASS | 薄障碍穿越 | 0.02m 薄障碍被发现，命令失败而非穿墙成功 | ../ts1_rigid_v2/physics-edges.json |
| T24 | PASS | 起降地面与碰墙分开 | 起降脚本无失败，障碍碰撞失败 | ../ts1_rigid_v2/scene_collision |
| T25 | NOT_RUN | 推力/倾角/加速度/角速率与外力 | 推力和力矩受限实测；尚无风力分离/饱和全矩阵 | ../ts1_rigid_v2/physics-edges.json |
| T26 | PASS | SDK 单位换算 | h/vgx/baro 合同单位与有限值通过 | ../ts1_rigid_v2/state_units |
| T27 | PASS | 无外部定位不可伪造 XYZ | 缺失通道为 null/invalid，丢定位终止 | ../ts1_rigid_v2/invalid_sensor |
| T28 | PASS | pose 延迟/丢失/不同 frame | 延迟/丢失已有回归；新补错误 frame 拒绝与 C1 身体坐标转换 | pytest.xml / test_c1_reset_yaw_mask_and_sensor_frame |
| T29 | PASS | 26 观测到 34 编码 | 顺序/旋转/符号/有效性和 mask 回归通过 | ../ts1_rigid_v2/observation_rotation |
| T30 | PASS | policy/critic 真值泄漏接口攻击 | 实际actor/critic接口仅接受复制的128维神经特征与9维mask，未传world/reward；输入篡改不改变物理状态 | test_c1_reliability.py / runtime.py |
| T31 | PASS | 整数物理/观测/神经时钟 | 120Hz/10Hz/4 子步共享执行器回归通过 | ../ts1_rigid_v2/shared_executor_clock |
| T32 | NOT_RUN | 0/30/60FPS 与断线确定性 | 仅回放交互，不冒充渲染开关同轨迹对照 | ../ts1_rigid_v2/BROWSER_CHECK.md |
| T33 | PASS | 多环境完整状态/RNG 独立 | 新增独立策略 Generator；跨lane调用顺序、reset、全局随机采样不改变本lane动作；v3保存/恢复全图对照一致 | pytest.xml / full-graph-check.json |
| T34 | PASS | 记录 copy 保持历史帧 | 分块重建与只读导出回归通过 | ../ts1_rigid_v2/route_physics_and_chunk |
| T35 | PASS | SMDP 手算 R/Gamma/delta | 手算 reward/discount/terminal 对照通过 | ../ts1_rigid_v2/smdp_discount |
| T36 | PASS | 变时长折扣 | 实际持续时间改变折扣目标 | ../ts1_rigid_v2/smdp_discount |
| T37 | PASS | terminal/truncation/GAE | terminal 与截断 bootstrap 分离；批量 lane 截断防跨局 | ../ts1_rigid_v2/gae_terminal |
| T38 | PASS | stop/无效动作时间与奖励 | 时间推进、成功奖励一次、deadline 回归通过 | ../ts1_rigid_v2/stop_dwell |
| T39 | PASS | 稳定保持而非高速掠过 | 停留门槛与飞出重置回归通过 | ../ts1_rigid_v2/dwell_success |
| T40 | PASS | PPO 缓存与冻结脑 | PPO 更新读出；冻结连接图与缓存回归通过 | ../ts1_rigid_v2/ppo_changes |
| T41 | PASS | checkpoint 不兼容拒绝 | 动作/编码/图/源码执行合同检查通过 | ../ts1_rigid_v2/checkpoint_contract |
| T42 | PASS | 安全边界恢复对照 | 新源码与v3快照完整真实图恢复：policy/optimizer/brain/lanes/RNG/physics/sensors逐项相同 | full-graph-check.json |
| T43 | PASS | 规则及随机基线 | C1规则100/100；同C1封存集随机3/300（随机碰撞1例） | rule-validation.json / random-sealed.json |
| T44 | NOT_RUN | MLP 与 MaleCNS pilot 来源分开 | 本轮真实 MaleCNS 已运行，新环境独立 MLP pilot 未运行 | ../ts1_rigid_v2/IMPLEMENTATION.md |
| T45 | PASS | 3 seed 同 300 封存场景 | C1同一300场封存：11=278/300,22=286/300,33=278/300；权重先锁定、后验收、未重调 | s*-sealed_test.json / frozen-checkpoints.json |
| T46 | PASS | options/wall/Ctrl-C 完整停止矩阵 | option 边界、真实 SIGINT/SIGTERM 和 wall预算停止；墙钟已耗尽的resume不再执行动作 | stop-matrix.json / full-graph-check.json |
| T47 | PASS | 无 GPU/无图模式 | 不存在的图直接FileNotFoundError，无替代图/模型；无GPU明确报错单测已通过 | missing-graph-check.json / pytest.xml |
| T48 | PASS | 浏览器真实脚本与来源 | 实际浏览器播放真实物理记录；明确脚本非模型 | ../ts1_rigid_v2/BROWSER_CHECK.md |
| T49 | PASS | 世界/机体/画面坐标 | 实际四元数可视化，旋转和平移路线验证 | ../ts1_rigid_v2/BROWSER_CHECK.md |
| T50 | PASS | 噪声 ghost/无效 pose 浏览器 | 实际浏览器拖动同一带噪/延迟记录的有效与无效帧；无效明确标注且ghost隐藏，真值继续显示 | BROWSER_CHECK.md / c1-invalid-pose-browser.png |
| T51 | PASS | reply loss 双状态 UI | 实际浏览器注入起飞reply loss：设备completed而客户端unknown，保护stop成功 | BROWSER_CHECK.md / c1-lost-reply-browser.png |
| T52 | PASS | 切换 lane/episode 实时观察 | 四lane独立快照与跨局替换；浏览器切env2及真实env1封存回放；导出全流身份一致且数值payload不变 | pytest.xml / view-export.xml / VIEW_EXPORT_AUDIT.json / BROWSER_CHECK.md |
| T53 | PASS | 暂停/拖动只读回放 | 本轮C1双回放实际播放/暂停/拖动到tick1440，画布平移不改变tick；独立正式评估仍继续 | BROWSER_CHECK.md / c1-camera-pan-browser.png |
| T54 | PASS | 慢客户端/五观看者/断线 | 五个真实HTTP读取者120s含慢读/新建连接，最大22062bytes、95%延迟约3ms；本实现使用HTTP最新快照，非WebSocket | observer-pressure.json |
| T55 | NOT_RUN | 长回放/256MiB 配额 | 分块与缩小配额有效前缀单测通过，完整 256MiB 压力未跑 | ../ts1_rigid_v2/recording_quota |
| T56 | PASS | 沙盒隔离训练源 | 后端只读来源与 epoch 隔离通过 | ../ts1_rigid_v2/sandbox_epoch |
| T57 | PASS | Host/Origin/path/真实 IP 安全 | 拒绝恶意来源与非授权变更；API 路径白名单 | ../ts1_rigid_v2/post_security |
| T58 | PASS | 旧/新/未知 schema | 未知格式拒绝后依次打开Golden和新物理模型；Golden全部文件哈希不变 | schema-check.json / BROWSER_CHECK.md |
| T59 | PASS | 初始/训练后同 case 双回放 | 预选c1-val-000，两次真实权重评估与浏览器双回放同case；两者均成功，未挑选收益样例 | comparison.json / BROWSER_CHECK.md |
| T60 | PASS | 新目录解包复核 | 新临时目录加载解包源码、API/静态依赖/三个模型回放及 Golden 注册均通过；使用现有 WSL 依赖，未测全新机器 | PACKAGE_AUDIT.json / package-smoke.log |
