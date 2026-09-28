from pathlib import Path
import argparse
import hashlib
import json
import torch
from flydrone.tellosim.visual import atomic_json

parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=11);seed=parser.parse_args().seed
root=Path('.').resolve();report=root/'reports/sdk9_warmstart'
if seed!=11:report=report/f'seed-{seed}'
run=root/f'runs/tellosim-sdk9/nearx-warmstart-s{seed}-20260926'
summary=json.loads((run/'summary.json').read_text())
source=root/f'runs/tellosim-sdk9/nearx-bootstrap-s{seed}-20260926/checkpoint.pt'
initial=torch.load(run/'initial.pt',map_location='cpu',weights_only=False)
old=torch.load(source,map_location='cpu',weights_only=False)
warm=torch.load(run/'critic-ready.pt',map_location='cpu',weights_only=False)
final=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=False)
assert all(torch.equal(v,initial['policy'][k]) for k,v in old['policy'].items())
assert all(torch.equal(v,warm['policy'][k]) for k,v in old['policy'].items() if not k.startswith('critic.'))
assert old['contract']==final['contract']
assert summary['checkpoint_roundtrip_exact']
parts={}
for prefix in ('body.','actor.','critic.'):
    parts[prefix[:-1]]={'unchanged':all(torch.equal(v,final['policy'][k]) for k,v in old['policy'].items() if k.startswith(prefix)),
        'delta_l2':float(torch.sqrt(sum((v-final['policy'][k]).square().sum() for k,v in old['policy'].items() if k.startswith(prefix))))}
assert parts['body']['unchanged'] and not parts['actor']['unchanged'] and not parts['critic']['unchanged']
before=next(x for x in summary['baselines'] if x['label']=='warm-start')['results']
after=summary['trained']['results']
sampled_before=json.loads((report/'sampled-before.json').read_text())
sampled_after=json.loads((report/'sampled-after.json').read_text())
assert sampled_before['checkpoint_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
assert sampled_after['checkpoint_sha256']==summary['checkpoint_sha256']
rows=[]
for mode,left,right in [('argmax',before,after),('sampled',sampled_before['results'],sampled_after['results'])]:
    assert len(left)==len(right)==16
    assert [r['case_id'] for r in left]==[r['case_id'] for r in right]
    for name,lo,hi in [('previous8',0,8),('additional8',8,16)]:
        a=sum(r['success'] for r in left[lo:hi]);b=sum(r['success'] for r in right[lo:hi])
        regressions=[a['case_id'] for a,b in zip(left[lo:hi],right[lo:hi]) if a['success'] and not b['success']]
        rows.append({'mode':mode,'cohort':name,'episodes':hi-lo,'before':a,'after':b,'nondecreasing':b>=a,'regressed_cases':regressions})
retained=all(row['nondecreasing'] and not row['regressed_cases'] for row in rows)
report_data={'schema':'sdk9.warmstart_check/1.0','status':'completed','seed':seed,'results':rows,
    'parameter_audit':parts,'critic_warmup_actor_unchanged':True,'checkpoint_roundtrip_exact':True,
    'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'final_sha256':summary['checkpoint_sha256'],
    'critic_fit':summary['critic_warmup'],'options':summary['options'],'ppo_updates':summary['updates'],
    'training_successes':sum(e['reason']=='success' for e in summary['episodes']),
    'training_completed_episodes':len(summary['episodes']),
    'PILOT_RETENTION_CHECK_PASSED':retained,'MODEL_READY_FOR_NEXT_STAGE':False,'sealed_test_opened':False,
    'scope':'One fixed training seed, 16 validation scenarios, one matched action RNG per case. Retention is not improvement or formal acceptance.',
    'next_step':'Repeat the fixed protocol on seeds 23 and 37 before four-direction training.' if retained else 'Keep supervised checkpoint as default; diagnose regressions before expanding curriculum.'}
atomic_json(report/'summary.json',report_data)
if retained and seed==11:
    published=dict(summary);published.update(artifact_directory=str(run.relative_to(root)),warmstart_retention=report_data)
    atomic_json(root/'reports/vis/tellosim/training-summary.json',published)
lines=['# 示范模型继续 PPO 训练：单种子保留能力检查','',
    f'固定 seed {seed}，先用 16 个独立训练回合拟合价值输出层，再执行 512 次 PPO 动作采样、4 次更新。学习率 0.00003；冻结果蝇连接组和共享隐藏层，只更新动作和价值输出层。','',
    '| 动作选择 | 验证场景 | 续训前 | 续训后 |','|---|---|---:|---:|']
for r in rows:lines.append(f"| {r['mode']} | {r['cohort']} | {r['before']}/8 | {r['after']}/8 |")
lines+=['','保留能力检查：'+('通过' if retained else '未通过')+'。正式阶段门禁 MODEL_READY_FOR_NEXT_STAGE 仍为 false。',
    '原有 8 场景与另外 8 场景分别报告；两种动作选择使用相同场景，不可相加解释为 32 个独立场景。采样动作使用匹配随机数。只评估固定预算的最终模型，没有按验证成绩挑选中间模型。',
    '', '价值拟合前后动作参数完全一致；PPO 后共享隐藏层完全一致，动作和价值输出层确实改变；检查点重新加载决策完全一致。',
    '', '价值拟合训练误差：'+str(summary['critic_warmup']['training_mse_before'])+' → '+str(summary['critic_warmup']['training_mse_after'])+'，这是训练拟合误差，不是独立的价值预测验证。',
    '', '后续：'+report_data['next_step'],'',
    'protocol.json 保存预先固定的配置；summary.json 保存成绩及参数审计；sampled-before.json / sampled-after.json 保存逐场景采样结果；s11.log 保存执行日志。',
    '', '范围：本轮仍只训练固定高度的近距离前后移动，未执行四方向、避障、300 场景封存评估或真机飞行。']
(report/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'retention':retained,'results':rows,'parameter_audit':parts},indent=2))
