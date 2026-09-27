# 复核命令与外部依赖

以下命令在解压根目录或WSL项目根目录运行。Python依赖使用reports/sdk9_c0/requirements.lock.txt；本次实际版本见environment.json。图文件需置于data/male-v1.npz，或在脚本里给出绝对路径。哈希为badc33a247894fe12c4d68d3ff791393857cffa39ff2ab81869608a11fa1e0c9。

## 软件验证

```sh
python -m pytest -q tests/flydrone/test_c1_reliability.py tests/flydrone/test_sdk_acceptance_remaining.py tests/flydrone/test_view_export_identity.py
```

本轮完整回归日志为pytest.xml；SDK补充和身份导出补充各有独立XML。不要把这条定向命令称作完整回归。

## 重新评估最终权重

下列命令不训练、不覆盖原始报告，也不复用已有回放目录。会重新进行900场C1封存评估；先查最终正式报告通常即可。

```python
from pathlib import Path
import json, time
from flydrone.tellosim.training.campaign_v2 import evaluate_batch
root=Path.cwd()
cases=json.loads((root/'reports/ts1_c1/cases.json').read_text())['sealed_test']
out=root/'reports'/('c1-recheck-'+str(time.time_ns()))
out.mkdir()
for seed in (11,22,33):
    evaluate_batch(root,root/'data/male-v1.npz',
        root/f'runs/tellosim-sdk9/c1-s{seed}/ppo/checkpoint.pt',
        cases,out/f's{seed}.json',f'recheck-s{seed}',record_indices=(),batch=64)
```

正式冻结权重文件哈希见frozen-checkpoints.json。训练前后的C0保持场景位于同一cases.json的c0_retention。

## 恢复语义

本轮PPO快照是v3完整边界文件，位置为runs/tellosim-sdk9/c1-sXX/ppo/resume.pt；只能以同一源码执行合同、CUDA/软件版本、环境数量及预算设置恢复。快照已完成128个PPO动作预算，重新启动不会自动开始新训练，也不会重置已耗墙钟。要扩大课程/预算应显式新建训练并热启动，不能篡改这批已冻结验收模型。

本轮监督/DAgger阶段保存stage权重、demonstrations.npz和provenance.json，没有宣称任意中途精确恢复。历史rigid-v2快照不满足当前v3合同，历史归档保留其原源码。

## 导出身份

view_export仅统一单环境物理记录与外层训练lane/case的身份字段；三维坐标、动作、神经活动和奖励不变。VIEW_EXPORT_AUDIT.json与每个回放manifest保留原始/导出分块哈希和数值payload哈希。正式评估数值使用的脚本快照见evaluation-source-before-export.py。
