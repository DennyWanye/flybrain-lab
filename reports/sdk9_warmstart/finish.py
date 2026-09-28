from pathlib import Path
import hashlib
import json
import shutil
from flydrone.tellosim.visual import atomic_json
root=Path('.').resolve();folder=root/'reports/sdk9_warmstart'
report=json.loads((folder/'multiseed-summary.json').read_text())
assert len(report['results'])==3
retained=report['RETENTION_CHECK_PASSED']
all_success=all(r['after']==r['episodes'] for result in report['results'] for r in result['results'])
report['NEXT_CURRICULUM_READY']=bool(retained and all_success)
report['decision']='expand_to_four_directions' if retained and all_success else 'diagnose_remaining_near_axis_failures_first'
for result in report['results']:result['next_step']=report['decision']
atomic_json(folder/'multiseed-summary.json',report)
lines=['# 示范模型继续 PPO 训练：三种子检查','',
    '结论：'+('三个种子均通过本轮保留能力检查。' if retained else '本轮保留能力检查未全部通过，需先处理退化。')+' MODEL_READY_FOR_NEXT_STAGE = false。','',
    '每个种子先采集 16 个独立训练回合，固定动作网络单独拟合价值输出层；随后以学习率 0.00003 执行 512 次动作采样、4 次 PPO 更新。冻结果蝇连接组与共享隐藏层，只更新动作和价值输出层。','',
    '| 种子 | 最高概率动作：续训前 → 后 | 概率采样动作：续训前 → 后 | 保留检查 |',
    '|---|---:|---:|---|']
for result in report['results']:
    def score(mode):
        rows=[r for r in result['results'] if r['mode']==mode]
        return f"{sum(r['before'] for r in rows)}/16 → {sum(r['after'] for r in rows)}/16"
    lines.append(f"| {result['seed']} | {score('argmax')} | {score('sampled')} | {'通过' if result['PILOT_RETENTION_CHECK_PASSED'] else '未通过'} |")
lines+=['','三个模型评估相同 16 个验证场景，其中 8 个来自此前验证、另外 8 个为本轮新增；不同动作模式也使用相同场景。不能将这些数字视为独立场景的正式成功率。采样模式每场景只有一个匹配随机数种子。','',
    '通过标准：每个种子、每种动作模式、每个 8 场景分组的成功数不下降，且没有任何原本成功的场景变为失败。该检查证明本轮未观察到能力退化，不等于证明 PPO 带来泛化提升。','',
    '## 参数与产物核对','',
    '- 三个源检查点均保留；初始化参数与源模型完全一致。',
    '- 价值预训练前后动作参数完全一致；PPO 后共享隐藏层仍完全一致，动作和价值输出层均发生实际更新。',
    '- 三个最终检查点保存、重新加载后，抽取的决策探针逐项一致。',
    '- 13 项 SDK9 相关测试通过；另外进行了 JavaScript 语法检查和 seed 11 浏览器回放检查。',
    '- 没有按验证成绩挑选中间模型；所有成绩来自固定预算的最终检查点。没有打开 300 场景封存测试。','',
    '## 下一阶段决定','',
    ('可以进入四方向课程的实现和小规模训练，届时必须同时保留前后方向回归场景。当前模型尚未学会左右移动或避障。' if retained and all_success else '暂不扩大到四方向课程。先分析仍然失败的近距离场景，补齐当前课程能力；保留原模型和本轮模型作对照。保留能力通过并不等于课程完全学会。'),'',
    '当前模型只覆盖固定约 1 米高度、前后 0.30–0.65 米范围目标。实机控制仍需要 SDK 传输层、外部定位及独立验证。','',
    '## 文件','',
    '- multiseed-summary.json：全部逐组成绩、参数审计、价值拟合记录。',
    '- summary.json、sampled-before.json、sampled-after.json：seed 11 明细。',
    '- seed-23/、seed-37/：其余种子的协议、日志和明细。',
    '- protocol.json：固定预算、分组与判断标准。',
    '- failure-before.json / failure-after.json：seed 23 持续失败场景的完整决策与回放 ID。',
    '- FAILURE_ANALYSIS.md：从首次过早停止到反向远离目标的逐步证据。',
    '- models/seed-*/：initial.pt、critic-ready.pt、checkpoint.pt 与训练日志（Windows 交付目录）。',
    '', '复现：先执行 METHOD.md 中的 seed 11 命令，完成配对评估，再执行 repeat.py。所有输出目录必须全新。']
(folder/'README.md').write_text('\n'.join(lines)+'\n')
method="""# 复现本轮训练

WSL 根目录：/home/denny/projects/flybrain_lab_4spark_v0_1
Python：/home/denny/projects/flybrain-env/bin/python

```bash
python -m flydrone.tellosim.training.runner --graph data/male-v1.npz --device cuda --profile balanced_rate_v3 --curriculum C0-near-x --options 512 --rollout 128 --eval-cases 16 --seed 11 --init-from runs/tellosim-sdk9/nearx-bootstrap-s11-20260926/checkpoint.pt --critic-episodes 16 --freeze-body --learning-rate 0.00003 --no-publish --out runs/tellosim-sdk9/NEW_NAME
```

分别对源检查点和新检查点执行配对采样评估：

```bash
python -m flydrone.tellosim.training.evaluation --graph data/male-v1.npz --checkpoint CHECKPOINT --out NEW_RESULT.json --case-count 16 --repeats 1 --sample-only
```

seed 23 和 37 使用相同配置，只改变 seed 和对应源检查点。repeat.py 记录了本轮精确的顺序执行命令；再次运行前须将脚本中的 tag 与输出路径改为全新名称，不能覆盖原训练目录。build_report.py 与 finish.py 使用本轮路径生成审计、汇总和 Windows 交付。PPO 是显式 warm-start：优化器和环境重置，并非精确断点续训。

critic-data.npz 保存模型采样轨迹的神经特征和完整回合折扣回报。价值拟合不接触验证数据；共享隐藏特征 detach 后，只优化 critic 输出层。PPO 阶段固定 body，优化 actor 与 critic 输出层，维持已有编码、图、动作掩码、时间尺度、成功半径和稳定保持时长。
"""
(folder/'METHOD.md').write_text(method)
viewer_path=root/'reports/vis/tellosim/training-summary.json'
viewer=json.loads(viewer_path.read_text())
if retained:
    viewer['warmstart_comparison']=report
    atomic_json(viewer_path,viewer)
else:
    original=json.loads((root/'runs/tellosim-sdk9/nearx-bootstrap-s11-20260926/summary.json').read_text())
    original['warmstart_comparison']=report
    atomic_json(viewer_path,original)
destination=Path('/mnt/d/projects/果蝇训练/sdk9-warmstart-delivery')
destination.mkdir(parents=True,exist_ok=True)
shutil.copytree(folder,destination,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
checks=[]
for seed in (11,23,37):
    run=root/f'runs/tellosim-sdk9/nearx-warmstart-s{seed}-20260926'
    target=destination/f'models/seed-{seed}';target.mkdir(parents=True,exist_ok=True)
    for name in ('initial.pt','critic-ready.pt','checkpoint.pt','summary.json','cases.json','critic-warmup.json','updates.jsonl','transitions.jsonl','critic-data.npz'):
        shutil.copy2(run/name,target/name)
    expected=next(r['final_sha256'] for r in report['results'] if r['seed']==seed)
    actual=hashlib.sha256((target/'checkpoint.pt').read_bytes()).hexdigest()
    assert actual==expected;checks.append({'seed':seed,'sha256':actual})
atomic_json(destination/'delivery-audit.json',{'models':checks,'all_checkpoint_copies_verified':True})
print(json.dumps({'retained':retained,'delivered':str(destination),'models':checks},indent=2))
