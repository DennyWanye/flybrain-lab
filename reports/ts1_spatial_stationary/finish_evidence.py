"""C2 evidence audit. No training or held-out reruns."""
from pathlib import Path
from collections import Counter
import json,hashlib
import numpy as np
import torch
from flydrone.tellosim.training.spatial_stationary_campaign import SEEDS,STAGES,source_hashes
from flydrone.tellosim.training.spatial_stationary import SKILLS,sha
from flydrone.tellosim.training.contracts import digest
root=Path.cwd();out=root/'reports/ts1_spatial_stationary'
def load(n):return json.loads((out/n).read_text())
summary=load('summary.json');protocol=load('protocol.json');cases=load('cases.json');lock=load('frozen-checkpoints.json')
assert protocol['split_hashes']=={k:digest(v) for k,v in cases.items()}
heldout={c['seed'] for rows in cases.values() for c in rows}
training=[];failures={};recorded={}
for seed in SEEDS:
    folder=root/f'runs/tellosim-sdk9/spatial-stationary-s{seed}'
    origin=json.loads((folder/'MODEL_ORIGIN.json').read_text())
    assert origin['status']=='transferred' and origin['new_training_actions']==0 and origin['source_training_actions']==5632 and origin['parameter_identical'] and origin['roundtrip_exact']
    assert origin['source_hashes']==source_hashes(root) and origin['checkpoint_sha256']==sha(folder/'checkpoint.pt')
    source=root/origin['source_checkpoint'];assert sha(source)==origin['source_sha256']
    old=torch.load(source,map_location='cpu',weights_only=False);final=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert final['format']=='tellosim.spatial_stationary_readouts/1'
    assert all(torch.equal(v,final['models'][skill][key]) for skill in SKILLS for key,v in old['models'][skill].items())
    prior=json.loads(source.with_name('training.json').read_text());assert prior['status']=='completed' and prior['options']==5632 and not prior['smoke_only']
    provenance=json.loads(source.with_name('provenance.json').read_text());assert not heldout.intersection(x['seed'] for x in provenance)
    training.append({'seed':seed,**origin,'heldout_overlap':0,'prior_training_audit':'reports/ts1_spatial_orientation/TRAINING_AUDIT.json'})
    for split in cases:
        r=load(f's{seed}-{split}.json');assert r['case_hash']==digest(cases[split]) and r['checkpoint_sha256']==sha(folder/'checkpoint.pt')
        assert [x['case_id'] for x in r['results']]==[x['case_id'] for x in cases[split]]
    r=load(f's{seed}-sealed_test.json')
    failures[str(seed)]={'reasons':dict(Counter(x['reason'] for x in r['results'] if not x['success'])),
        'phases':dict(Counter(x['final_phase'] for x in r['results'] if not x['success'])),
        'sealed_cases':[x for x in r['results'] if not x['success']],
        'boundary_cases':[x for x in load(f's{seed}-boundary.json')['results'] if not x['success']],
        'instruction_cases':[x for x in load(f's{seed}-instruction_pairs.json')['results'] if not x['success']]}
    for x in r['results']:
        if x['run_id']:recorded[x['run_id']]=(x,sha(folder/'checkpoint.pt'))
(out/'TRAINING_AUDIT.json').write_text(json.dumps(training,indent=2))
(out/'failure-analysis.json').write_text(json.dumps(failures,indent=2))
replays=[];bootstrap_rows=[]
for name,(result,checkpoint) in recorded.items():
    folder=root/'reports/vis/tellosim'/name;m=json.loads((folder/'manifest.json').read_text())
    assert m['complete'] and not m.get('partial') and m['outcome']==result['reason']
    assert m['observation_schema']=='tellosim.spatial_observation26/1.0' and m['skill_bundle']['checkpoint_sha256']==checkpoint
    frames=[];chunks=0
    for stream,parts in m['streams'].items():
        for part in parts:
            p=folder/part['file'];assert sha(p)==part['sha256'];chunks+=1
            rows=[json.loads(line) for line in p.read_text().splitlines()]
            for r in rows:
                if r['episode_id']!=m['episode_id']:
                    assert stream=='trajectory' and r['episode_id']=='episode-0' and r['sim_tick']==0 and r['run_id']==name and r['epoch']==m['epoch']
                    bootstrap_rows.append({'run_id':name,'stream':stream,'tick':0,'episode_id':'episode-0','scope':'constructor bootstrap before task manifest identity; not a policy transition'})
            if stream=='transition':frames.extend(rows)
    assert frames and frames[-1]['finished'] and frames[-1]['termination_reason']==result['reason']
    for frame in frames:
        assert len(frame['observation'])==26 and 'altitude' in frame and 'heading' in frame
        if frame.get('policy'):
            assert frame['policy']['input_source']=='reservoir_v_trace' and frame['policy']['checkpoint_sha256']==checkpoint
    assert m['disturbance_evidence']==result['disturbance']
    replays.append({'run_id':name,'chunks':chunks,'frames':len(frames),'outcome':result['reason']})
assert len(replays)==12
(out/'REPLAY_AUDIT.json').write_text(json.dumps({'all_passed':True,'runs':replays,'constructor_bootstrap_rows':bootstrap_rows,'identity_check':'all task records match case identity; tick0 constructor trajectory explicitly audited separately'},indent=2))
inventory=load('historical-inventory.json');assert all(sha(root/p)==h for p,h in inventory.items())
(out/'HISTORY_AUDIT.json').write_text(json.dumps({'all_unchanged':True,'files':len(inventory)},indent=2))
history=load('process-cleanup.json');assert len(history)==6 and all(v['reaped'] and v['returncode']==0 and not Path(f"/proc/{v['pid']}").exists() for v in history.values())
(out/'PROCESS_VERIFICATION.json').write_text(json.dumps({'all_pids_absent':True,'jobs':history},indent=2))
lines=['# C2 连续三维位置、指定朝向与保持','',f"JOINT_3D_TASK_VERIFIED = {'YES' if summary['JOINT_3D_TASK_VERIFIED'] else 'NO'}",'',
'三种读出来自C2Q完整任务轨迹的正式训练，冻结完整MaleCNS。C2U仅修复阶段管理的高度测量去噪，三组权重逐张量不变；不声称C2U新增训练，也不是单一端到端策略或全连接组训练。',
'', '|种子|验证/100|封存/300|边界/12|指令/8|零特征/12|碰撞|','|---|---|---|---|---|---|---|']
for r in summary['models']:lines.append(f"|{r['seed']}|{r['validation']}|{r['sealed']}|{r['boundary']}|{r['instruction']}|{r['zero']}|{r['collisions']}|")
lines+=['',f"规则{summary['rule_successes']}/100，随机{summary['random_successes']}/300；二者不是神经策略结果。",'',
'来源C2Q每种子5632个真实训练动作、三组16896个；原始训练和参数更新审计见../ts1_spatial_orientation/TRAINING_AUDIT.json。C2U新训练动作0，权重与来源严格相同，重新冻结合同并以新划分正式评估。静止高度窗需20样本、跨度≤2cm、每个测量垂直速度≤0.08m/s，才求高度均值；其余情况采用当前高度。水平位置及所有速度始终使用当前测量。',
'', '阶段为高度→水平位置→朝向，测量漂移触发恢复；切换前后完整物理积分状态、控制目标及神经时钟一致。动作20cm/30度，STOP2秒；XY≤20cm、Z≤10cm、朝向≤16度、水平及垂直速度≤0.08m/s、yaw速率≤0.08rad/s，连续2秒。每阶段60秒、新三阶段任务总限时180秒。',
'', '定位噪声/延迟、三轴外力和偏航力矩沿用V1R；训练与评估均实际经历非1m高度的导航。', '', '未通过项：']
lines+=summary['acceptance']['reasons'] or ['全部预注册模型门槛通过。']
for seed,f in failures.items():lines.append(f"种子{seed}失败原因{f['reasons']}，阶段{f['phases']}。")
lines+=['','12项针对性检查、128场规则开发验证和8场新真实神经链路开发检查已完成；12份固定索引正式回放已通过数据审计。软件回归与浏览器证据见reports/ts1_completion。',
'', '历史V1/J2R/V1R源码、权重和报告保留不变。新封存仅用于评价，固定最终权重未以封存挑选。完整TS1须另看reports/ts1_completion最终交付验收；本模型报告不覆盖真机。']
(out/'README.md').write_text('\n'.join(lines)+'\n')
(out/'NEXT_STAGE.md').write_text('C2已通过，可继续软件验收R3—R8。\n' if summary['JOINT_3D_TASK_VERIFIED'] else 'C2门槛未通过，保留失败。下一训练实验须使用新版本、新开发/训练/封存数据；禁止按这批封存反复调参或放宽门槛。\n')
print(json.dumps({'models':3,'replays':12,'history_unchanged':True,'ready':summary['JOINT_3D_TASK_VERIFIED']}))
