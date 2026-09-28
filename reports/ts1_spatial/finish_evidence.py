"""C2 evidence audit. No training or held-out reruns."""
from pathlib import Path
from collections import Counter
import json,hashlib
import numpy as np
import torch
from flydrone.tellosim.training.spatial_campaign import SEEDS,STAGES,source_hashes
from flydrone.tellosim.training.spatial import SKILLS,sha
from flydrone.tellosim.training.contracts import digest
root=Path.cwd();out=root/'reports/ts1_spatial'
def load(n):return json.loads((out/n).read_text())
summary=load('summary.json');protocol=load('protocol.json');cases=load('cases.json');lock=load('frozen-checkpoints.json')
assert protocol['split_hashes']=={k:digest(v) for k,v in cases.items()}
heldout={c['seed'] for rows in cases.values() for c in rows}
training=[];failures={};recorded={}
for seed in SEEDS:
    folder=root/f'runs/tellosim-sdk9/spatial-s{seed}';t=json.loads((folder/'training.json').read_text());provenance=json.loads((folder/'provenance.json').read_text())
    assert t['status']=='completed' and not t['smoke_only'] and t['options']==sum(STAGES)
    assert t['source_hashes']==source_hashes(root) and t['checkpoint_sha256']==sha(folder/'checkpoint.pt')
    assert sum(x['kind']=='decision' for x in provenance)==sum(STAGES)
    assert not heldout.intersection(x['seed'] for x in provenance)
    initial=torch.load(folder/'initial.pt',map_location='cpu',weights_only=False);final=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert initial['contract']==final['contract'] and final['format']=='tellosim.spatial_readouts/1'
    skills={}
    for skill in SKILLS:
        info=t['skills'][skill];data=np.load(folder/skill/'demonstrations.npz');rows=[x for x in provenance if x['skill']==skill];old=info['rehearsal_examples']
        assert len(rows)==info['new_examples'] and data['features'].shape==(info['examples'],128)
        assert np.array_equal(data['labels'][old:],[x['label'] for x in rows])
        assert np.array_equal(data['weights'][old:],[x['weight'] for x in rows])
        delta=float(torch.sqrt(sum((initial['models'][skill][k]-v).square().sum() for k,v in final['models'][skill].items())))
        assert delta>0 and abs(delta-info['parameter_delta_l2'])<1e-6 and info['roundtrip_exact'] and info['options']>0
        skills[skill]={'parameter_delta_l2':delta,'real_actions':info['options'],'new_samples':len(rows),'rehearsal_samples':old,
            'measured_height_range_m':[min(x['measured_height_m'] for x in rows),max(x['measured_height_m'] for x in rows)]}
    training.append({'seed':seed,'skills':skills,'heldout_overlap':0,'checkpoint_sha256':sha(folder/'checkpoint.pt')})
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
replays=[]
for name,(result,checkpoint) in recorded.items():
    folder=root/'reports/vis/tellosim'/name;m=json.loads((folder/'manifest.json').read_text())
    assert m['complete'] and not m.get('partial') and m['outcome']==result['reason']
    assert m['observation_schema']=='tellosim.spatial_observation26/1.0' and m['skill_bundle']['checkpoint_sha256']==checkpoint
    frames=[];chunks=0
    for stream,parts in m['streams'].items():
        for part in parts:
            p=folder/part['file'];assert sha(p)==part['sha256'];chunks+=1
            rows=[json.loads(line) for line in p.read_text().splitlines()]
            assert all(r['episode_id']==m['episode_id'] for r in rows)
            if stream=='transition':frames.extend(rows)
    assert frames and frames[-1]['finished'] and frames[-1]['termination_reason']==result['reason']
    for frame in frames:
        assert len(frame['observation'])==26 and 'altitude' in frame and 'heading' in frame
        if frame.get('policy'):
            assert frame['policy']['input_source']=='reservoir_v_trace' and frame['policy']['checkpoint_sha256']==checkpoint
    assert m['disturbance_evidence']==result['disturbance']
    replays.append({'run_id':name,'chunks':chunks,'frames':len(frames),'outcome':result['reason']})
assert len(replays)==12
(out/'REPLAY_AUDIT.json').write_text(json.dumps({'all_passed':True,'runs':replays},indent=2))
inventory=load('historical-inventory.json');assert all(sha(root/p)==h for p,h in inventory.items())
(out/'HISTORY_AUDIT.json').write_text(json.dumps({'all_unchanged':True,'files':len(inventory)},indent=2))
history=load('process-cleanup.json');assert len(history)==7 and all(v['reaped'] and v['returncode']==0 and not Path(f"/proc/{v['pid']}").exists() for v in history.values())
(out/'PROCESS_VERIFICATION.json').write_text(json.dumps({'all_pids_absent':True,'jobs':history},indent=2))
lines=['# C2 连续三维位置、指定朝向与保持','',f"JOINT_3D_TASK_VERIFIED = {'YES' if summary['JOINT_3D_TASK_VERIFIED'] else 'NO'}",'',
'完整任务轨迹训练三种读出，冻结完整MaleCNS。不是只拼接旧技能，也不是单一端到端策略或全连接组训练。',
'', '|种子|验证/100|封存/300|边界/12|指令/8|零特征/12|碰撞|','|---|---|---|---|---|---|---|']
for r in summary['models']:lines.append(f"|{r['seed']}|{r['validation']}|{r['sealed']}|{r['boundary']}|{r['instruction']}|{r['zero']}|{r['collisions']}|")
lines+=['',f"规则{summary['rule_successes']}/100，随机{summary['random_successes']}/300；二者不是神经策略结果。",'',
'每种子4096个真实训练动作，三组12288个。1024教师+两轮各1536 DAgger，每阶段60优化轮；旧样本每四条取一条以半权重复习。三种读出均实际更新并通过保存加载一致性核对。',
'', '阶段为高度→水平位置→朝向，测量漂移触发恢复；切换前后完整物理积分状态、控制目标及神经时钟一致。动作20cm/30度，STOP2秒；XY≤20cm、Z≤10cm、朝向≤16度、水平及垂直速度≤0.08m/s、yaw速率≤0.08rad/s，连续2秒。每阶段60秒、新三阶段任务总限时180秒。',
'', '定位噪声/延迟、三轴外力和偏航力矩沿用V1R；训练与评估均实际经历非1m高度的导航。', '', '未通过项：']
lines+=summary['acceptance']['reasons'] or ['全部预注册模型门槛通过。']
for seed,f in failures.items():lines.append(f"种子{seed}失败原因{f['reasons']}，阶段{f['phases']}。")
lines+=['','4项针对性检查与640动作真实神经链路smoke已完成；12份固定索引正式回放已通过数据审计。未进行完整历史回归或本阶段浏览器实点验收。',
'', '历史V1/J2R/V1R源码、权重和报告保留不变。新封存仅用于评价，固定最终权重未以封存挑选。完整TS1和真机仍未就绪。']
(out/'README.md').write_text('\n'.join(lines)+'\n')
(out/'NEXT_STAGE.md').write_text('C2已通过，可继续软件验收R3—R8。\n' if summary['JOINT_3D_TASK_VERIFIED'] else 'C2门槛未通过，保留失败。下一训练实验须使用新版本、新开发/训练/封存数据；禁止按这批封存反复调参或放宽门槛。\n')
print(json.dumps({'models':3,'replays':12,'history_unchanged':True,'ready':summary['JOINT_3D_TASK_VERIFIED']}))
