"""Assemble current evidence without substituting missing results."""
from pathlib import Path
import hashlib,json,shutil,zipfile,subprocess
from flydrone.tellosim.training.c0_campaign import acceptance
from evidence_checks import fault_safety
from flydrone.tellosim.training.contracts import digest
from flydrone.tellosim.visual import atomic_json
ROOT=Path('.').resolve();OUT=ROOT/'reports/sdk9_c0'
def read(path):return json.loads(path.read_text()) if path.exists() else None

def main():
    models=[]
    for seed in (11,22,33):
        directory=ROOT/f'runs/tellosim-sdk9/c0-s{seed}-20260927'
        train=read(directory/'training.json');valid=read(directory/'validation.json');sealed=read(directory/'sealed.json')
        models.append({'seed':seed,'directory':str(directory.relative_to(ROOT)),
            'status':train['status'] if train else ('INCOMPLETE_OR_RUNNING' if (directory/'protocol.json').exists() else 'NOT_RUN'),'training':train,'validation':valid,'sealed':sealed,
            'failure_replay':read(directory/'failure-009.json'),'faults':read(directory/'faults.json'),'near_regression':read(directory/'near-regression.json'),'near_sampled':read(directory/'near-sampled.json')})
    cases=read(OUT/'cases.json');random=read(OUT/'random-sealed.json')
    gate=acceptance({m['seed']:m['sealed'] for m in models if m['sealed']},random,digest(cases['sealed_test'])) if random else {'passed':False,'reasons':['sealed random baseline or final models not evaluated']}
    completed=all(m['training'] and m['training']['status']=='completed' for m in models)
    rule=read(OUT/'rule-validation.json');rule_ok=bool(rule and rule['episodes']==100 and rule['success_rate']>=.99)
    audit=read(OUT/'artifact-audit.json')
    integrity_ok=bool(audit and audit['complete_three_seeds'] and audit['same_contract'])
    retention_ok=all(m['near_regression'] and m['near_regression']['successes']==m['near_regression']['episodes']==40 for m in models)
    fault_ok=all(fault_safety(m['faults']) for m in models)
    ready=gate['passed'] and completed and rule_ok and integrity_ok and retention_ok and fault_ok
    if not retention_ok:gate['reasons'].append('near-distance deterministic regression incomplete or failed')
    if not fault_ok:gate['reasons'].append('fault protection checks incomplete or failed')
    if not integrity_ok:gate['reasons'].append('three-model artifact integrity audit incomplete')
    conclusion=('三个模型均通过 C0 封存学习门槛；可进入下一课程，实机飞行仍未开放。' if ready else 'C0 尚未完成正式验收；等待三个模型的完整封存成绩与文件核验。' if not all(m['sealed'] for m in models) else 'C0 正式验收未通过；至少一项成功率、安全指标或文件核验未达到门槛。')
    scope='独立初始化读出，经示范、DAgger 和 PPO 训练；连接组冻结。封存结果只评价该仿真空房间分布；不代表避障、转向、高度课程或真实无人机能力。'
    report={'schema':'sdk9.c0_campaign/1.0','models':models,'rule_validation':rule,
        'random_sealed':random,'raw_mlp_validation':read(OUT/'raw-mlp-validation.json'),
        'deployment_mode':'argmax','NEAR_SKILL_RETAINED':retention_ok,'FAULT_PROTECTION_VERIFIED':fault_ok,'artifact_audit':audit,'formal_gate':gate,'MODEL_READY_FOR_NEXT_STAGE':ready,'TS1_C0_TASK_LEARNED':ready,
        'FULL_TS1_ACCEPTANCE':False,'REAL_FLIGHT_READY':False,'conclusion':conclusion,'scope':scope}
    atomic_json(OUT/'summary.json',report)
    representative=models[0]
    if representative['training'] and representative['validation']:
        train=representative['training'];trained=representative['validation']
        viewer={'curriculum':'C0','c0_campaign':report,'options':train['options'],
            'updates':train['supervised_updates']+train['ppo_updates'],'ppo_updates':train['ppo_updates'],
            'parameter_delta_l2':next((m['parameter_delta_l2'] for m in (audit or {}).get('models',[]) if m['seed']==11),None),
            'training_method':train['training_method'],'checkpoint_roundtrip_exact':train['checkpoint_roundtrip_exact'],
            'baselines':[r for r in (rule,read(OUT/'untrained-validation.json'),read(OUT/'raw-mlp-validation.json'),random) if r],
            'trained':trained,'MODEL_READY_FOR_NEXT_STAGE':ready,'TS1_C0_TASK_LEARNED':ready}
        atomic_json(ROOT/'reports/vis/tellosim/training-summary.json',viewer)
    lines=['# 完整 C0 模拟训练交付','',conclusion,'',scope,'',
        '| 正式种子 | 开发验证 | 封存测试 | 训练状态 |','|---|---:|---:|---|']
    score=lambda r:f"{r['successes']}/{r['episodes']}" if r else 'NOT_RUN'
    for m in models:lines.append(f"| {m['seed']} | {score(m['validation'])} | {score(m['sealed'])} | {m['status']} |")
    lines += ['', '指标：目标半径 20 cm、水平速度 <=8 cm/s、高度误差 <=10 cm，并保持 2 秒。期限 60 模拟秒。',
        '正式部署模式为 argmax：每次选择模型给出最高概率的合法动作。概率采样只作探索诊断；不会重复抽样挑成功回合。',
        '正式通过要求每个种子 300 个相同封存场景成功率 >=90%、碰撞与越界 <=1%，比均匀随机至少提高 20 个百分点。',
        '', '## 文件', '', 'summary.json 保存逐场景结果；cases.json 是训练前固定的验证/封存场景；各模型目录保存初始化、分阶段、PPO 前及最终权重。',
        '训练结果以保存并重新加载后的权重进行评估；正式模型固定前不得开启封存集。旧模型和历史结果均保留。',
        'USER_TEST_GUIDE.md 提供浏览器操作步骤；DIAGNOSTICS.md 列出故障与全部失败；MODEL_INVENTORY.json 提供最终权重哈希；PACKAGE_AUDIT.json 位于 Windows 交付根目录。',
        '', '## 复现', '', '在 WSL 项目根目录激活 /home/denny/projects/flybrain-env 后运行：',
        '```bash', 'python -m flydrone.tellosim.training.c0_campaign train --seed 11 --out runs/tellosim-sdk9/NEW_NAME',
        'python -m flydrone.tellosim.training.c0_campaign evaluate --checkpoint runs/tellosim-sdk9/NEW_NAME/checkpoint.pt --out runs/tellosim-sdk9/NEW_NAME/validation.json',
        'python -m flyview serve --project-root . --host 127.0.0.1 --port 8765', '```',
        '', '浏览器：http://127.0.0.1:8765/tellosim 。默认展示固定种子 11，不根据最好成绩选代表。',
        '数据图路径 data/male-v1.npz；约 16.67 万神经元、2558 万条边，不重复打包原始图或虚拟环境。',
        '精确断点续训未开放：checkpoint 支持推理或显式热启动；不能将热启动称为相同轨迹续训。',
        '', '## 验收边界', '', 'FULL_TS1_ACCEPTANCE 和 REAL_FLIGHT_READY 仍为 NO。完整 SDK loopback/遥测、真实姿态力矩控制和硬件标定不能由 C0 成绩推断通过，详见 ACCEPTANCE.md。']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'models':[{k:v for k,v in m.items() if k in ('seed','status')} for m in models],'ready':ready,'conclusion':conclusion},ensure_ascii=False))
if __name__=='__main__':main()
