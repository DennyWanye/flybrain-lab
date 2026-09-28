"""Audit V1 frozen training, evaluation and replay evidence; generate human report."""
from pathlib import Path
from collections import Counter
import json,hashlib,sys,xml.etree.ElementTree as ET
import numpy as np
import torch
root=Path.cwd();sys.path.insert(0,str(root));out=root/'reports/ts1_altitude_refined'
from flydrone.tellosim.training.joint import sha
from flydrone.tellosim.training.contracts import digest
from flydrone.tellosim.training.altitude_refined_campaign import SEEDS,STAGES

def load(n):return json.loads((out/n).read_text())
summary=load('summary.json');protocol=load('protocol.json');cases=load('cases.json');heldout=set()
for name in ['ts1_altitude_refined','ts1_altitude','ts1_stability','ts1_robust','ts1_joint_refined','ts1_joint']:
    document=json.loads((root/f'reports/{name}/cases.json').read_text())
    for split in ['validation','sealed_test','instruction_pairs','boundary','zero_cases']:
        heldout.update(r['seed'] for r in document.get(split,[]))
assert protocol['split_hashes']=={k:digest(v) for k,v in cases.items()}
training=[];recorded={};failures={}
for seed in SEEDS:
    folder=root/f'runs/tellosim-sdk9/altitude-refined-s{seed}';t=json.loads((folder/'training.json').read_text());records=json.loads((folder/'provenance.json').read_text())
    assert t['status']=='completed' and not t['smoke_only'] and t['options']==sum(STAGES)
    assert sum(r['kind']=='decision' for r in records)==sum(STAGES) and len(records)==t['new_examples']
    assert not heldout.intersection(r['seed'] for r in records)
    assert all(0<=r['seed']-(220000000+seed*100000+r['stage']*10000)<16*500 for r in records)
    assert set(r['profile'] for r in records)=={'clean','pose','force','combined'}
    assert set(r['label'] for r in records)=={0,5,6}
    data=np.load(folder/'demonstrations.npz');assert np.array_equal(data['labels'],[r['label'] for r in records])
    assert np.array_equal(data['weights'],[r['weight'] for r in records]) and data['features'].shape==(len(records),128)
    initial=torch.load(folder/'initial.pt',map_location='cpu',weights_only=False);final=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert initial['contract']==final['contract'] and final['format']=='tellosim.altitude_refined_readout/1'
    assert all(sha(root/p)==h for p,h in t['task_sources'].items())
    delta=float(torch.sqrt(sum((initial['policy'][k]-v).square().sum() for k,v in final['policy'].items())))
    assert delta>0 and abs(delta-t['parameter_delta_l2'])<1e-6 and t['roundtrip_exact']
    assert sha(folder/'checkpoint.pt')==t['checkpoint_sha256']
    assert all(sha(root/v['path'])==v['sha256'] for v in protocol['legacy_bundles'][str(seed)]['sources'].values())
    training.append(dict(seed=seed,options=t['options'],new_examples=len(records),settled_examples=sum(r['kind']=='settled' for r in records),parameter_delta_l2=delta,roundtrip_exact=True,heldout_overlap=0,legacy_checkpoints_unchanged=True,label_counts=dict(Counter(map(int,data['labels'])))))
    result=load(f's{seed}-sealed_test.json');assert result['case_hash']==protocol['split_hashes']['sealed_test']
    for r in result['results']:
        if r['run_id']:recorded[r['run_id']]=r
    failures[str(seed)]={'reasons':dict(Counter(r['reason'] for r in result['results'] if not r['success'])),'profiles':dict(Counter(r['disturbance']['profile'] for r in result['results'] if not r['success'])),'cases':[r for r in result['results'] if not r['success']],'boundary_cases':[r for r in load(f's{seed}-boundary.json')['results'] if not r['success']],'validation_cases':[r for r in load(f's{seed}-validation.json')['results'] if not r['success']]}
(out/'TRAINING_AUDIT.json').write_text(json.dumps(training,indent=2))
runs=[]
for name,result in sorted(recorded.items()):
    folder=root/'reports/vis/tellosim'/name;m=json.loads((folder/'manifest.json').read_text())
    assert m['complete'] and not m.get('partial') and m['outcome']==result['reason']
    assert m['disturbance_evidence']==result['disturbance'] and m['observation_schema']=='tellosim.altitude_refined_observation26/1.0'
    assert sha(root/m['altitude_checkpoint']['path'])==m['altitude_checkpoint']['sha256']
    chunks=0;frames=[];force_rows=0
    for stream,parts in m['streams'].items():
        for part in parts:
            p=folder/part['file'];assert sha(p)==part['sha256'];chunks+=1
            rows=[json.loads(line) for line in p.read_text().splitlines()]
            assert all(r.get('episode_id')==m['episode_id'] and r.get('env_id')==m['env_id'] for r in rows)
            if stream=='transition':frames.extend(rows)
            force_rows+=sum(abs(r.get('control',{}).get('external_force_world_n',[0,0,0])[2])>0 for r in rows)
    if result['disturbance']['profile'] in ['force','combined']:assert force_rows>0 and result['disturbance']['absolute_vertical_impulse_ns']>0
    assert frames and frames[-1]['finished'] and frames[-1]['termination_reason']==result['reason']
    for frame in frames:
        assert len(m['observation_names'])==len(frame['observation'])==26 and 'altitude' in frame and 'heading' not in frame
        a=frame['altitude'];assert a['target_z_m']==m['case']['goal'][2]
        if a['measured_z_m'] is not None:assert abs(a['error_m']-(a['target_z_m']-a['measured_z_m']))<1e-12
        if frame.get('policy'):
            policy=frame['policy'];assert policy['skill']=='altitude' and policy['input_source']=='reservoir_v_trace'
            assert policy['checkpoint_sha256']==m['altitude_checkpoint']['sha256']
    runs.append(dict(run_id=name,outcome=result['reason'],chunks_verified=chunks,frames_verified=len(frames),vertical_force_rows=force_rows,measured_altitude_verified=True,policy_checkpoint_verified=True))
assert len(runs)==15
(out/'REPLAY_AUDIT.json').write_text(json.dumps(dict(all_passed=True,runs=runs),indent=2))
history=load('process-cleanup.json');assert len(history)==7 and all(v['reaped'] and v['returncode']==0 for v in history.values())
assert all(not Path(f"/proc/{v['pid']}").exists() for v in history.values())
(out/'PROCESS_VERIFICATION.json').write_text(json.dumps(dict(all_campaign_children_reaped=True,all_recorded_pids_absent=True,jobs=history,observed_training_private_kib=sum(v['private_kib'] for v in load('owned-training-memory-sample.json').values()),observed_evaluation_private_kib=sum(v['private_kib'] for v in load('owned-evaluation-memory-sample.json').values())),indent=2))
tree=ET.parse(out/'focused.xml');assert all(int(n.get('failures',0))==0 and int(n.get('errors',0))==0 for n in tree.iter('testsuite'));tests=len(list(tree.iter('testcase')))
# Historical preservation and evaluation binding are checked independently of model success.
inventory=load('historical-inventory.json')
assert all(sha(root/path)==value for path,value in inventory.items())
for seed in SEEDS:
    entry=next(x for x in load('frozen-checkpoints.json')['models'] if x['seed']==seed)
    for split in cases:
        result=load(f's{seed}-{split}.json')
        assert result['case_hash']==protocol['split_hashes'][split]
        assert result['checkpoint_sha256']==entry['sha256']
        assert len(result['results'])==len(cases[split])
        assert [r['case_id'] for r in result['results']]==[c['case_id'] for c in cases[split]]
(out/'HISTORY_AUDIT.json').write_text(json.dumps({'all_unchanged':True,'files':len(inventory)},indent=2))
(out/'failure-analysis.json').write_text(json.dumps(failures,indent=2))
lines=['# V1R 高度边界与停止时机实验','',f"MODEL_READY_FOR_NEXT_STAGE = {'YES' if summary['MODEL_READY_FOR_NEXT_STAGE'] else 'NO'}",'',
'主体实现：平滑多尺度高度感官编码、决策点和已停止状态监督、阈值邻域对称加权、覆盖整个20厘米动作半步网格的训练分布。26维测量经34编码通道进入完整冻结MaleCNS，策略只读取128维下游神经特征。',
'', '动作、物理、噪声与外力、成功容差、连续保持2秒、最少8秒、最长60秒及全部模型门槛沿用V1。新源码、新权重和新数据划分；旧V1失败不变。',
'', '|种子|验证|封存|clean|pose|force|combined|边界|指令|零特征|旧J2R逐场相同|',
'|---|---|---|---|---|---|---|---|---|---|---|']
for r in summary['models']:
    profiles='|'.join(str(r['profiles'][name]['successes'])+'/75' for name in ['clean','pose','force','combined'])
    lines.append(f"|{r['seed']}|{r['validation']}|{r['sealed']}|{profiles}|{r['boundary_successes']}/12|{r['instruction_pairs_successes']}/4|{r['zero_features_successes']}/12|{r['legacy_exact']}|")
lines+=['','每种子768教师动作+两轮各1024 DAgger动作，三种子共8448个真实动作；每阶段80轮监督优化。神经状态样本不计额外动作。固定最终权重，全部冻结后才运行封存。不是PPO，不训练连接组突触或value head。',
'', '门禁结果：']
lines+=summary['acceptance']['reasons'] or ['全部预注册模型门槛通过。']
for seed,f in failures.items():
    lines.append(f"种子{seed}封存失败原因：{f['reasons']}；分布：{f['profiles']}。")
    for r in f['boundary_cases']:
        lines.append(f"边界 {seed}/{r['case_id']}：目标{r['target_height_m']:.3f}m，最终误差{r['height_error_m']:.9f}m，保持{r['stable_hold_s']:.1f}s，{r['reason']}。")
lines+=['',f"规则基线{summary['rule_successes']}/100；随机基线{summary['random_successes']}/300。两者均不是神经模型结果。",'',
f'{tests}项针对性自动检查通过；128动作真实链路smoke及8个独立开发场景完成；15个固定索引正式回放已审计；旧源码、权重、报告242个文件哈希未变。没有重跑全部历史回归。',
'', '真实全图静态诊断：V1在本次±9.5/10.5cm对照上神经特征差为0；V1R能区分。该结果仅支持编码修复假设，不代表整体任务成功。见development-probe.json。',
'', '本阶段未更新Viewer快捷入口或进行浏览器实点验收。回放已落盘且通过数据审计，不能据此宣称UI验收通过。',
'', '固定XY独立高度技能，起飞由mission manager完成，使用模拟外部定位。JOINT_3D_TASK_VERIFIED、FULL_TS1_READY、REAL_FLIGHT_READY仍为NO。',
'', '入口：python -m reports.ts1_altitude_refined.run_campaign。已有实验拒绝覆盖；后续实验另建版本。']
(out/'README.md').write_text('\n'.join(lines)+'\n')
next_step=('R1模型门禁通过。下一步R2须另立协议：连续三维位置、指定朝向和稳定保持，核验非1m导航观测分布，不得直接拼旧策略声称通过。' if summary['MODEL_READY_FOR_NEXT_STAGE'] else 'R1模型门禁未通过，R2—R8未启动。保留V1R失败，禁止用这批封存再调参或放宽标准。下一训练实验须新版本、新协议、新数据划分，只用独立开发证据决策。')
(out/'NEXT_STAGE.md').write_text('# 下一阶段\n\n'+next_step+'\n')
print(json.dumps(dict(training_models=len(training),formal_replays=len(runs),focused_tests=tests,history_unchanged=True,ready=summary['MODEL_READY_FOR_NEXT_STAGE'])))
