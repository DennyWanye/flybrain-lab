"""Audit V1 frozen training, evaluation and replay evidence; generate human report."""
from pathlib import Path
from collections import Counter
import json,hashlib,sys,xml.etree.ElementTree as ET
import numpy as np
import torch
root=Path.cwd();sys.path.insert(0,str(root));out=root/'reports/ts1_altitude'
from flydrone.tellosim.training.joint import sha
from flydrone.tellosim.training.contracts import digest
from flydrone.tellosim.training.altitude_campaign import SEEDS,STAGES

def load(n):return json.loads((out/n).read_text())
summary=load('summary.json');protocol=load('protocol.json');cases=load('cases.json');heldout=set()
for name in ['ts1_altitude','ts1_stability','ts1_robust','ts1_joint_refined','ts1_joint']:
    document=json.loads((root/f'reports/{name}/cases.json').read_text())
    for split in ['validation','sealed_test','instruction_pairs','boundary','zero_cases']:
        heldout.update(r['seed'] for r in document.get(split,[]))
assert protocol['split_hashes']=={k:digest(v) for k,v in cases.items()}
training=[];recorded={};failures={}
for seed in SEEDS:
    folder=root/f'runs/tellosim-sdk9/altitude-s{seed}';t=json.loads((folder/'training.json').read_text());records=json.loads((folder/'provenance.json').read_text())
    assert t['status']=='completed' and not t['smoke_only'] and t['options']==sum(STAGES)
    assert sum(r['kind']=='decision' for r in records)==sum(STAGES) and len(records)==t['new_examples']
    assert not heldout.intersection(r['seed'] for r in records)
    assert all(0<=r['seed']-(190000000+seed*100000+r['stage']*10000)<16*500 for r in records)
    assert set(r['profile'] for r in records)=={'clean','pose','force','combined'}
    assert set(r['label'] for r in records)=={0,5,6}
    data=np.load(folder/'demonstrations.npz');assert np.array_equal(data['labels'],[r['label'] for r in records])
    assert np.array_equal(data['weights'],[r['weight'] for r in records]) and data['features'].shape==(len(records),128)
    initial=torch.load(folder/'initial.pt',map_location='cpu',weights_only=False);final=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert initial['contract']==final['contract'] and final['format']=='tellosim.altitude_readout/1'
    assert all(sha(root/p)==h for p,h in t['task_sources'].items())
    delta=float(torch.sqrt(sum((initial['policy'][k]-v).square().sum() for k,v in final['policy'].items())))
    assert delta>0 and abs(delta-t['parameter_delta_l2'])<1e-6 and t['roundtrip_exact']
    assert sha(folder/'checkpoint.pt')==t['checkpoint_sha256']
    assert all(sha(root/v['path'])==v['sha256'] for v in protocol['legacy_bundles'][str(seed)]['sources'].values())
    training.append(dict(seed=seed,options=t['options'],new_examples=len(records),intermediate_examples=sum(r['kind']=='intermediate' for r in records),parameter_delta_l2=delta,roundtrip_exact=True,heldout_overlap=0,legacy_checkpoints_unchanged=True,label_counts=dict(Counter(map(int,data['labels'])))))
    result=load(f's{seed}-sealed_test.json');assert result['case_hash']==protocol['split_hashes']['sealed_test']
    for r in result['results']:
        if r['run_id']:recorded[r['run_id']]=r
    failures[str(seed)]={'reasons':dict(Counter(r['reason'] for r in result['results'] if not r['success'])),'profiles':dict(Counter(r['disturbance']['profile'] for r in result['results'] if not r['success'])),'cases':[r for r in result['results'] if not r['success']],'boundary_cases':[r for r in load(f's{seed}-boundary.json')['results'] if not r['success']],'validation_cases':[r for r in load(f's{seed}-validation.json')['results'] if not r['success']]}
(out/'TRAINING_AUDIT.json').write_text(json.dumps(training,indent=2))
runs=[]
for name,result in sorted(recorded.items()):
    folder=root/'reports/vis/tellosim'/name;m=json.loads((folder/'manifest.json').read_text())
    assert m['complete'] and not m.get('partial') and m['outcome']==result['reason']
    assert m['disturbance_evidence']==result['disturbance'] and m['observation_schema']=='tellosim.altitude_observation26/1.0'
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
tree=ET.parse(out/'pytest.xml');assert all(int(n.get('failures',0))==0 and int(n.get('errors',0))==0 for n in tree.iter('testsuite'));tests=len(list(tree.iter('testcase')))
lines=['# V1 独立高度控制训练与封存评估','',f"MODEL_READY_FOR_NEXT_STAGE = {'YES' if summary['MODEL_READY_FOR_NEXT_STAGE'] else 'NO'}",'',summary['note'],'','|种子|验证|封存|成功率|碰撞/越界|高度指令对照|边界|零特征|旧技能复测|','|---|---|---|---|---|---|---|---|---|']
for r in summary['models']:lines.append(f"|{r['seed']}|{r['validation']}|{r['sealed']}|{r['success_rate']:.1%}|{r['collision_or_bounds']}|{r['instruction_pairs_successes']}/4|{r['boundary_successes']}/12|{r['zero_features_successes']}/12|{r['legacy_exact']}|")
lines+=['','## 四类场景','', '|种子|无扰动|定位噪声和延迟|三轴外力|叠加扰动|','|---|---|---|---|---|']
for r in summary['models']:lines.append('|'+str(r['seed'])+'|'+ '|'.join(str(r['profiles'][p]['successes'])+'/75' for p in ['clean','pose','force','combined'])+'|')
lines+=['','## 训练与验收范围','', '冻结完整MaleCNS，只从随机初始化训练独立高度读出。每种子512个教师动作加两轮各768个DAgger动作，合计6144个真实训练动作。每两个真实10Hz神经采样附加一次监督标签；标签数不计作额外动作。最终权重在封存评估前统一冻结，未按封存成绩挑选种子或权重。原J2R导航与转向权重、代码和历史报告保留不变。', '', '起飞后从1米高度开始，原地到达0.25至1.75米内的目标高度。只允许STOP、上升20厘米、下降20厘米。高度误差≤10厘米、垂直及水平速度≤0.08米/秒、水平偏移≤20厘米、朝向漂移≤16度、角速度≤0.08弧度/秒，同时稳定2秒。每场至少8秒，最多60秒。上下安全动作屏蔽只读取测量高度，与目标方向无关。', '', '扰动：2毫米位置噪声和100毫秒延迟；水平与垂直各0.006牛顿外力，偏航力矩0.00003牛米，起飞后第5秒开始，每12秒持续2秒，通过MuJoCo物理积分施加。属于合成工程扰动，没有按真实飞行器标定。', '', '预注册门禁：每种子300场总成功率≥90%，无扰动≥90%，其他每类≥85%，碰撞/越界≤1%，较均匀随机至少高20个百分点；高度指令4/4，边界至少11/12，零特征0/12，规则基线≥99/100；每种子旧J2R前12场验证逐项相同。零特征使用预注册内部目标，避免常量下降碰到安全屏蔽高度后偶然到位。', '', '## 未通过项','']
lines+=summary['acceptance']['reasons'] or ['全部预注册门禁通过。']
for seed,f in failures.items():lines.append(f"种子{seed}：{f['reasons']}；场景{f['profiles']}。")
lines+=['','## 高度边界失败详情','']
for seed,f in failures.items():
    for r in f['boundary_cases']:lines.append(f"种子{seed} / {r['case_id']}：目标{r['target_height_m']:.3f}米，结束高度误差{r['height_error_m']:.6f}米，稳定保持{r['stable_hold_s']:.1f}秒，结果{r['reason']}。结束单帧高度达标也不等于已保持2秒。")
lines+=['','## 查看与验证','',f"{tests}项自动化测试通过；15份正式回放的数据块哈希、身份、训练权重来源、测量高度与垂直外力记录已核对。规则基线{summary['rule_successes']}/100、均匀随机{summary['random_successes']}/300均不计作神经策略成绩。",'', '打开 http://127.0.0.1:8765/tellosim ，Ctrl+R刷新，点击“V1 高度 11 · 上升”和“V1 高度 11 · 下降”；也可选择22、33。入口固定封存第4场上升、第8场下降，不按成功筛选。右侧显示目标高度、测量高度、误差与垂直速度。', '', '浏览器此前的工具策略限制尚未解除，本轮仅做离线源码、回放和交付包核验，未声称实际点击、播放暂停、拖动或画布交互已验证。', '', 'V1为独立高度技能；三维位置与朝向联合任务尚未验证。完整TS1与真实飞行均未就绪。正式入口：python reports/ts1_altitude/run_campaign.py；现有结果不覆盖，下一实验需新协议和输出。交付包回放无需外部大图，重训仍需要完整data/male-v1.npz和现有WSL依赖，未验证新机器安装。']
(out/'README.md').write_text('\n'.join(lines)+'\n');(out/'failure-analysis.json').write_text(json.dumps(failures,indent=2))
next_step='在新协议、新训练与封存场景下，把独立高度控制与位置、朝向连续组合；必须验证变化高度下旧导航特征分布及阶段切换，不能直接插入旧策略就宣称3D任务通过。' if summary['MODEL_READY_FOR_NEXT_STAGE'] else '先用开发验证分析高度误差边界、STOP时机和上下动作量化限制；保留本轮封存失败，不按封存反复调参，不放宽稳定标准。暂不进入3D联合任务。'
(out/'NEXT_STAGE.md').write_text('# V1 下一阶段\n\n'+next_step+'\n')
print(json.dumps(dict(training_models=len(training),formal_replays=len(runs),tests=tests,ready=summary['MODEL_READY_FOR_NEXT_STAGE'])))
