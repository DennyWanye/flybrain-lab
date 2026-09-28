# 训练权重与完整记录

Release: https://github.com/DennyWanye/flybrain-lab/releases/tag/ts1-c2w-shadow-v1-20260928

附件 FlyBrain_C2W_ShadowV1_Training_Records_20260928.zip 包含：

- runs/ 下现存训练权重、阶段权重、训练来源、轨迹及历史/失败版本；当前正式模型为 runs/tellosim-sdk9/spatial-continuous-s11、s22、s33。
- reports/ 下训练/开发/正式评估结果、案例划分、封存清单、回放、测试、Shadow 输入/输出与失败证据；包含 reports/vis/tellosim 的完整回放块。
- artifacts/ 与 logs/ 的已有模型/回放/训练日志。
- 精确的 data/male-v1.npz 和元数据，以及对应源码快照与第三方归属说明。

完整逐文件清单与 SHA256 见 reports/github_release_20260928/ARTIFACT_MANIFEST.json。压缩包 SHA256 见同目录 RELEASE_ASSETS.json，以及 Release 的 SHA256SUMS 附件。

未包含运行环境、缓存、Git 内部目录、实时 Viewer 临时状态、第三方原始下载副本及 256MiB 配额测试填充文件。配额测试报告/脚本和所有本地文件仍保留。此排除不丢弃模型、训练来源、评估场景或失败结论。

## 使用

使用此 Release 对应的源码版本，在新目录解压附件，先按 ARTIFACT_MANIFEST.json 校验文件。源码快照随附件保留，可避免与其他 checkout 的未提交更改混用。使用项目既有依赖环境，不需要重新训练。

```bash
gh release download ts1-c2w-shadow-v1-20260928 --repo DennyWanye/flybrain-lab --dir 新目录
# 在下载目录检查整个包
sha256sum -c SHA256SUMS
# 在单独的新目录解压；不要覆盖已有实验目录
unzip FlyBrain_C2W_ShadowV1_Training_Records_20260928.zip -d 新解压目录
```

模型会严格校验图与绑定源码的哈希。当前三组权重、三阶段读出及历史来源权重都包含在 runs/ 中。完整神经图固定，C2W 新训练动作数 0；不能把包大小、下载成功或 Shadow 小跑当模型新增训练/真实飞行通过。

从模拟转向硬件仍需真实定位、标定和独立授权，REAL_FLIGHT_READY=false。
