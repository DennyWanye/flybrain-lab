# 原 TS1 的 60 项验收映射

PASS 仅指本行所列模拟证据。NOT_RUN 表示原条目完整要求尚未验证，即使部分实现或单测已通过。80 个 pytest 用例不等于 60 项全部通过。PENDING 项在最终评估/解包完成后更新。每行版本见 implementation-hashes.json，耗时字段见 TEST_MATRIX.json；缺少单项计时保留 null。

| ID | 状态 | 输入/要求 | 实际证据与限制 | 产物/测试关键字 |
|---|---|---|---|---|
| T01 | NOT_RUN | 合法/非法 wire 与 typed 往返 | wire 向量已通过；完整 typed 往返矩阵未覆盖 | approved_command_vectors |
| T02 | NOT_RUN | typed 类型与 JSON 语义拒绝 | 重复键、NaN 等已覆盖；typed float/bool/Inf 全组合未覆盖 | ambiguous_or_unknown_json |
| T03 | NOT_RUN | 握手/地面/重复起降状态 | 部分状态机回归通过，完整逐状态矩阵未执行 | lost_ack_idempotency |
| T04 | NOT_RUN | 20/500 cm 与 speed 10/100 | 大房间 500 cm 实际跟踪通过；两种速度边界未全部重测 | physics-edges.json |
| T05 | PASS | go 20 拒绝、21 允许 | 合同向量拒绝 20；physics-edges.json 中 go21 实际执行 | approved_command_vectors |
| T06 | PASS | yaw0/90 下坐标与锁定目标 | 新物理四元数、目标和转向后移动对照通过 | rigid_body_route_rotates |
| T07 | PASS | cw/ccw 实际 360° | 两个方向完整转动、非零耗时 | physics-edges.json |
| T08 | PASS | 运动期间查询/busy | 查询不移动目标；忙请求与回执归属通过 | go_query_share |
| T09 | PASS | request 丢失 | 不执行，不重试，unknown | request_and_reply_loss |
| T10 | PASS | reply 丢失 | 执行但客户端 unknown，不重试 | request_and_reply_loss |
| T11 | NOT_RUN | 迟到 ack 与后一请求 | 延迟与取消回归已过；指定迟到 ack 错配矩阵未完整执行 | delayed_channel |
| T12 | PASS | request_id 本地幂等 | 同一本地请求只执行一次，不声明原生 wire 幂等 | lost_ack_idempotency |
| T13 | PASS | 运动中保护 stop | 停止与旧请求状态回归通过；模拟能力 | lost_ack_idempotency |
| T14 | NOT_RUN | 取消 Future/关闭 UI 不等于停止 | 取消排队请求已测；执行中 Future 和关闭 UI 全矩阵未测 | delayed_channel |
| T15 | NOT_RUN | 15s watchdog 与查询/busy 计时 | loopback 已有实现，本轮未完整执行 15s 矩阵 | SDK_COMPATIBILITY_MATRIX.md |
| T16 | PASS | 不支持命令无副作用 | 不支持命令拒绝，真实设备不开放 | approved_command_vectors |
| T17 | PASS | loopback 与真实 IP 拒绝 | 地址限制与真实 IP JSON 拒绝通过 | loopback_address |
| T18 | NOT_RUN | inprocess/UDP 同列表逐项等价 | 两种通道已有独立测试，完整同列表对照未执行 | training_channel |
| T19 | PASS | 自由落体/推力悬停 | 重力积分和悬停通过，不靠覆盖 qpos 移动 | preflight.json |
| T20 | PASS | 连续起飞/转向/平移/降落 | 8 命令全 ack、无碰撞；真实轨迹可回放 | preflight.json |
| T21 | PASS | 5 扰动场景悬停 10s | 位置和高度误差均在 0.1m 内 | preflight.json |
| T22 | PASS | 六向/go/整圈/大房间 | 大场景范围和稳定窗口检查通过 | physics-edges.json |
| T23 | PASS | 薄障碍穿越 | 0.02m 薄障碍被发现，命令失败而非穿墙成功 | physics-edges.json |
| T24 | PASS | 起降地面与碰墙分开 | 起降脚本无失败，障碍碰撞失败 | scene_collision |
| T25 | NOT_RUN | 推力/倾角/加速度/角速率与外力 | 推力和力矩受限实测；尚无风力分离/饱和全矩阵 | physics-edges.json |
| T26 | PASS | SDK 单位换算 | h/vgx/baro 合同单位与有限值通过 | state_units |
| T27 | PASS | 无外部定位不可伪造 XYZ | 缺失通道为 null/invalid，丢定位终止 | invalid_sensor |
| T28 | NOT_RUN | pose 延迟/丢失/不同 frame | 延迟、丢失和状态恢复已测；不同 frame 拒绝专项未跑 | delayed_inflight |
| T29 | PASS | 26 观测到 34 编码 | 顺序/旋转/符号/有效性和 mask 回归通过 | observation_rotation |
| T30 | NOT_RUN | policy/critic 真值泄漏接口攻击 | 接口设计隔离已审读，完整逆向依赖攻击未跑 | checkpoint_contract |
| T31 | PASS | 整数物理/观测/神经时钟 | 120Hz/10Hz/4 子步共享执行器回归通过 | shared_executor_clock |
| T32 | NOT_RUN | 0/30/60FPS 与断线确定性 | 仅回放交互，不冒充渲染开关同轨迹对照 | BROWSER_CHECK.md |
| T33 | NOT_RUN | 多环境完整状态/RNG 独立 | 脑/世界/传感器/队列独立已测；策略采样仍用全局 Torch RNG | masked_batch |
| T34 | PASS | 记录 copy 保持历史帧 | 分块重建与只读导出回归通过 | route_physics_and_chunk |
| T35 | PASS | SMDP 手算 R/Gamma/delta | 手算 reward/discount/terminal 对照通过 | smdp_discount |
| T36 | PASS | 变时长折扣 | 实际持续时间改变折扣目标 | smdp_discount |
| T37 | PASS | terminal/truncation/GAE | terminal 与截断 bootstrap 分离；批量 lane 截断防跨局 | gae_terminal |
| T38 | PASS | stop/无效动作时间与奖励 | 时间推进、成功奖励一次、deadline 回归通过 | stop_dwell |
| T39 | PASS | 稳定保持而非高速掠过 | 停留门槛与飞出重置回归通过 | dwell_success |
| T40 | PASS | PPO 缓存与冻结脑 | PPO 更新读出；冻结连接图与缓存回归通过 | ppo_changes |
| T41 | PASS | checkpoint 不兼容拒绝 | 动作/编码/图/源码执行合同检查通过 | checkpoint_contract |
| T42 | PASS | 安全边界恢复对照 | 完整图连续/恢复和 3 次重复逐项一致；限定锁定配置 | repeat-resume.json |
| T43 | PASS | 规则及随机基线 | 规则 100/100，随机封存 5/300 | rule-validation.json |
| T44 | NOT_RUN | MLP 与 MaleCNS pilot 来源分开 | 本轮真实 MaleCNS 已运行，新环境独立 MLP pilot 未运行 | IMPLEMENTATION.md |
| T45 | PASS | 3 seed 同 300 封存场景 | 同一封存集每种子 300 场：284/276/279 成功，碰撞越界均 0；冻结哈希一致 | final-s*-sealed_test.json |
| T46 | NOT_RUN | options/wall/Ctrl-C 完整停止矩阵 | option 边界与恢复已过，wall/Ctrl-C 完整矩阵未跑 | full-graph-check.json |
| T47 | NOT_RUN | 无 GPU/无图模式 | GPU 不可用报错已有单测，缺图全流程模式未跑 | gpu_unavailable |
| T48 | PASS | 浏览器真实脚本与来源 | 实际浏览器播放真实物理记录；明确脚本非模型 | BROWSER_CHECK.md |
| T49 | PASS | 世界/机体/画面坐标 | 实际四元数可视化，旋转和平移路线验证 | BROWSER_CHECK.md |
| T50 | NOT_RUN | 噪声 ghost/无效 pose 浏览器 | 真值/观测分层已有实现，本轮未逐状态注入 UI 验证 | BROWSER_CHECK.md |
| T51 | NOT_RUN | reply loss 双状态 UI | 后端回归通过，本轮未浏览器注入复核 | request_and_reply_loss |
| T52 | NOT_RUN | 切换 lane/episode 实时观察 | 只导出 lane 0，没有实时四 lane 切换 | IMPLEMENTATION.md |
| T53 | PASS | 暂停/拖动只读回放 | 实测 tick 跳转与暂停；独立训练仍持续运行 | BROWSER_CHECK.md |
| T54 | NOT_RUN | 慢客户端/五观看者/断线 | 未执行压力矩阵，不声明吞吐门槛达标 | IMPLEMENTATION.md |
| T55 | NOT_RUN | 长回放/256MiB 配额 | 分块与缩小配额有效前缀单测通过，完整 256MiB 压力未跑 | recording_quota |
| T56 | PASS | 沙盒隔离训练源 | 后端只读来源与 epoch 隔离通过 | sandbox_epoch |
| T57 | PASS | Host/Origin/path/真实 IP 安全 | 拒绝恶意来源与非授权变更；API 路径白名单 | post_security |
| T58 | NOT_RUN | 旧/新/未知 schema | 旧与新分入口保留，完整未知版本切换验收未跑 | BROWSER_CHECK.md |
| T59 | NOT_RUN | 初始/训练后同 case 双回放 | 同 case 旧权重与新权重逐场景结果已有；浏览器配对对照未完成 | s11-validation.json |
| T60 | PASS | 新目录解包复核 | 新临时目录加载解包源码、API/静态依赖/三个模型回放及 Golden 注册均通过；使用现有 WSL 依赖，未测全新机器 | PACKAGE_AUDIT.json / package-smoke.log |
