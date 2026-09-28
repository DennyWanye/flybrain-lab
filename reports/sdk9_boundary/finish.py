from pathlib import Path
import json
import hashlib
import shutil
import torch
from flydrone.tellosim.visual import atomic_json
root=Path('.').resolve();folder=root/'reports/sdk9_boundary';summaries=[];audits=[]
for seed in (11,23,37):
    run=root/f'runs/tellosim-sdk9/boundary-repair-s{seed}-recovery-20260926'
    data=json.loads((run/'summary.json').read_text());summaries.append(data)
    initial=torch.load(run/'initial.pt',map_location='cpu',weights_only=False)
    final=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert initial['contract']==final['contract']
    assert data['checkpoint_sha256']==hashlib.sha256((run/'checkpoint.pt').read_bytes()).hexdigest()
    assert data['source_sha256']==hashlib.sha256((root/f'runs/tellosim-sdk9/nearx-warmstart-s{seed}-20260926/checkpoint.pt').read_bytes()).hexdigest()
    cases=json.loads((run/'cases.json').read_text());train={r['seed'] for stage in cases['training_stages'] for r in stage}
    old_cases=json.loads((root/f'runs/tellosim-sdk9/nearx-bootstrap-s{seed}-20260926/cases.json').read_text())
    warm_cases=json.loads((root/f'runs/tellosim-sdk9/nearx-warmstart-s{seed}-20260926/cases.json').read_text())
    train.update(old_cases['training_seeds']);train.update(cases['critic_seeds']);train.update(warm_cases['critic_seeds'])
    train.update(range(warm_cases['train_seed_start'],warm_cases['train_seed_start']+512))
    assert not train & {r['seed'] for r in cases['validation']}
    assert not cases['known_validation_case_used_in_training'] and not cases['sealed_test_opened']
    assert data['checkpoint_roundtrip_exact'] and data['critic_warmup']['actor_exactly_unchanged']
    audits.append({'seed':seed,'contract_unchanged':True,'training_validation_separate':True,
        'checkpoint_roundtrip_exact':True,'checkpoint_sha256':data['checkpoint_sha256']})
passed=all(s['KNOWN_FAILURE_REPAIRED'] and s['NO_CASE_REGRESSIONS'] for s in summaries)
report={'schema':'sdk9.boundary_repair_comparison/1.0','seeds':[11,23,37],
    'results':summaries,'BOUNDARY_REPAIR_CHECK_PASSED':passed,'MODEL_READY_FOR_NEXT_STAGE':False,
    'sealed_test_opened':False,'scope':'Three training seeds x same 40 scenarios x two action modes. Not independent 240-case testing. Validation-005 was previously examined; fresh cases are separate.'}
first=json.loads((root/'runs/tellosim-sdk9/boundary-repair-s23-20260926/summary.json').read_text())
atomic_json(folder/'first-attempt-summary.json',first)
report['first_attempt']={'seed':23,'KNOWN_FAILURE_REPAIRED':first['KNOWN_FAILURE_REPAIRED'],'NO_CASE_REGRESSIONS':first['NO_CASE_REGRESSIONS'],'comparison':first['comparison']}
atomic_json(folder/'summary.json',report);atomic_json(folder/'artifact-audit.json',{'models':audits})
lines=['# 目标边界纠错训练','',
    '本轮检查：'+('通过' if passed else '未全部通过')+'。MODEL_READY_FOR_NEXT_STAGE = false。','',
    '方法：保持果蝇连接图、编码与归一化不变，在动作读出上进行纠错示范及两轮 DAgger。每个种子使用 64+96+96 个独立训练回合，并保留此前示范数据；训练期间有明确记录的 STOP、越过目标及随机短动作探索。每阶段固定训练 200 个 epoch，学习率 0.0003。最终评估没有教师、强制停顿或观测旁路。本轮新增 PPO 更新为 0。','',
    '| 种子 | 原有 16 场景（固定） | 新近距离 8 场景（固定） | 边界 8 场景（固定） | 本轮保留 8 场景（固定） | 概率模式总计 |',
    '|---|---:|---:|---:|---:|---:|']
for s in summaries:
    def cell(cohort):
        row=next(r for r in s['comparison'] if r['mode']=='argmax' and r['cohort']==cohort)
        return f"{row['before']}/{row['episodes']} → {row['after']}/{row['episodes']}"
    sampled=[r for r in s['comparison'] if r['mode']=='sampled']
    score=f"{sum(r['before'] for r in sampled)}/40 → {sum(r['after'] for r in sampled)}/40"
    lines.append(f"| {s['seed']} | {cell('previous16')} | {cell('near')} | {cell('boundary')} | {cell('recovery_holdout')} | {score} |")
lines+=['','前后使用相同场景与相同动作采样随机数。三个模型共用同一组 40 个场景，不可视为 120 个独立场景；每种子每场景采样模式只有一次。32 个开发验证场景在此前或第一轮纠错后已经检查；另外 8 个保留场景在第二轮训练完成后首次评估。所有场景均未加入训练；模型方法的迭代受开发验证反馈影响，因此不作为封存验收。',
    '', '## 完成内容','',
    '- 第一轮 seed 23：确定性 32/32，采样 31/32，但有一个原成功场景回退。第一轮模型、日志和结果完整保留；第二轮增加恢复状态训练，没有把失败验证轨迹加入训练。',
    '- 主体编码由当前会话完成，训练主流程跑通后才执行相关范围回归；16 项 SDK9 测试通过。',
    '- 保存所有源模型、训练协议、逐场景前后结果、纠错标签来源、训练数据和最终检查点。未根据验证成绩挑选中间模型。',
    '- 共享隐藏层更新后重新拟合价值部分；拟合价值时动作参数保持完全一致。',
    '- 检查点重新加载后 32 个探针的决策逐项一致，图、映射、编码、物理世界契约保持一致。',
    '', '## 限制与决定','',
    ('同一个问题场景在三个种子的最终模型中均成功，所有已成功场景没有回退；可以开始规划和小规模验证四方向课程，同时继续保留这批前后方向回归场景。' if passed else '本轮仍有失败或退化，暂不进入四方向。具体回退场景见 summary.json 的 comparison 字段。'),
    '该结论只覆盖无噪声、固定高度的前后移动和边界恢复，不包含左右控制、避障、封存场景验收或实机飞行。',
    '', '## 文件','',
    '- summary.json：全部模型成绩与逐场景结果。',
    '- artifact-audit.json：契约、数据划分、模型哈希核对。',
    '- models/seed-*/：Windows 交付中的最终检查点、初始检查点、数据及协议。',
    '- feature-audit.json（各模型目录）：训练数据诊断，不作为验证成功率。',
    '- DIAGNOSIS.md：两轮纠错的依据与证据边界；BROWSER_CHECK.md：实际页面核对。',
    '', '复现入口：python -m flydrone.tellosim.training.boundary_repair --graph data/male-v1.npz --seed 23 --tag NEW_TAG --recovery。输出目录必须全新；当前基线为上一轮同种子的 warm-start 检查点。']
(folder/'README.md').write_text('\n'.join(lines)+'\n')
# Seed 23 is the predeclared failure-investigation representative, not best-seed selection.
if passed:
    chosen=next(s for s in summaries if s['seed']==23)
    chosen=dict(chosen);chosen['boundary_comparison']=report
    atomic_json(root/'reports/vis/tellosim/training-summary.json',chosen)
destination=Path('/mnt/d/projects/果蝇训练/sdk9-boundary-delivery');destination.mkdir(parents=True,exist_ok=True)
shutil.copytree(folder,destination,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
for seed in (11,23,37):
    run=root/f'runs/tellosim-sdk9/boundary-repair-s{seed}-recovery-20260926';target=destination/f'models/seed-{seed}'
    shutil.copytree(run,target,dirs_exist_ok=True)
    assert hashlib.sha256((target/'checkpoint.pt').read_bytes()).hexdigest()==next(s['checkpoint_sha256'] for s in summaries if s['seed']==seed)
print(json.dumps({'passed':passed,'delivery':str(destination)},indent=2))

shutil.copytree(root/'runs/tellosim-sdk9/boundary-repair-s23-20260926',destination/'first-attempt-seed23',dirs_exist_ok=True)
