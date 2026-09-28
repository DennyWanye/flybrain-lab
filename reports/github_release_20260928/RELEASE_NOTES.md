完成 TS1 C2W 三维位置、指定朝向与连续保持的模拟验收，新增定位输入、记录回放及无发送 Shadow V1。

- C2W seeds 11/22/33：sealed 290/300、293/300、294/300，三组原正式门槛通过。R8 软件回归 211 项通过。
- Shadow V1：43 项针对性检查；三组完整 MaleCNS 图各两次回放逐字节一致；实时输入、断流、积压验证通过。其输入为合成观测，不代表真实硬件验证。
- C2Q 每 seed 实际训练 5632 个决策动作；C2W 参数不变迁移，Shadow 无新增训练。
- REAL_FLIGHT_READY=false；没有连接或控制真实无人机。

附件包含训练权重、阶段权重、训练来源、案例划分、完整成功/失败记录与回放、精确神经图、源码快照及归属说明。15,803 个 payload 文件逐文件 SHA256 校验通过。源代码、测试和可浏览报告在该 tag 的 Git 历史中。

压缩包：1,008,492,390 bytes。
SHA256：44737eea2b378fb6e1ec94510176b44c5be1bb86dfdd6e5fabf1cf5c701fca70。

下载后先用 SHA256SUMS 验证，再解压到新目录。完整清单见 ARTIFACT_MANIFEST.json。排除运行环境、缓存、实时 Viewer 临时状态、第三方原始下载副本及 256MiB 配额测试填充文件；报告、复现脚本和所有本地原件保留。

数据来源与许可见 docs/DATA_PROVENANCE.md。具体使用说明见 docs/TRAINED_ARTIFACTS.md 与 docs/shadow_v1.md。
