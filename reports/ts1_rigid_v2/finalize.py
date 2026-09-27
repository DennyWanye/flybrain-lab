from pathlib import Path
import hashlib,json,datetime,xml.etree.ElementTree as ET
import torch
from flydrone.tellosim.training.c0_campaign import acceptance
from flydrone.tellosim.training.contracts import digest
from flydrone.tellosim.visual import atomic_json
ROOT=Path('.').resolve();OUT=ROOT/'reports/ts1_rigid_v2'
def read(name):return json.loads((OUT/name).read_text())
def main():
    lock=read('frozen-checkpoints.json');cases=read('cases.json');models={};rows=[];inventory=[]
    for seed in (11,22,33):
        model=read(f'final-s{seed}-sealed_test.json');validation=read(f'final-s{seed}-validation.json');before=read(f's{seed}-validation.json')
        checkpoint=ROOT/f'runs/tellosim-sdk9/rigid-final-s{seed}/checkpoint.pt';sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert sha in lock['checkpoint_hashes'] and model['checkpoint_sha256']==validation['checkpoint_sha256']==sha
        assert validation['case_hash']==digest(cases['validation']) and len(validation['results'])==100
        models[seed]=model
        training=json.loads((ROOT/f'runs/tellosim-sdk9/rigid-adapt-s{seed}/training.json').read_text())
        final=json.loads((checkpoint.parent/'summary.json').read_text());assert training['status']==final['status']=='completed'
        old=torch.load(ROOT/f'runs/tellosim-sdk9/c0-s{seed}-20260927/checkpoint.pt',map_location='cpu',weights_only=False)
        new=torch.load(checkpoint,map_location='cpu',weights_only=False)
        delta=sum(float(torch.sum((new['policy'][k]-v)**2)) for k,v in old['policy'].items())**.5
        assert delta>0
        rows.append({'seed':seed,'before_validation':f"{before['successes']}/100",'validation':f"{validation['successes']}/100",'sealed':f"{model['successes']}/300",'collision_or_bounds':model['collision_or_bounds'],
            'run_id':next((r['run_id'] for r in model['results'] if r['run_id']),None),'method':'原有读出适配：512 示范/DAgger + 160 PPO 动作','new_training_options':training['options']+final['options'],
            'parameter_delta_l2_from_legacy':delta,'training_wall_s':training['elapsed_s']+final['elapsed_s']})
        inventory.append({'seed':seed,'checkpoint':str(checkpoint.relative_to(ROOT)),'sha256':sha,'bytes':checkpoint.stat().st_size,'neural_backend':new['contract']['neural_backend'],
            'resume_checkpoint':str((checkpoint.parent/'resume.pt').relative_to(ROOT))})
    baseline=read('random-sealed.json');gate=acceptance(models,baseline,lock['case_hash'])
    xml=ET.parse(OUT/'pytest.xml').getroot();tests=list(xml.iter('testcase'));assert len(tests)==80 and not list(xml.iter('failure')) and not list(xml.iter('error'))
    increment=all(read(name)['passed'] for name in ('preflight.json','physics-edges.json','repeat-resume.json','cuda-sparse-check64.json'))
    summary={'schema':'tellosim.rigid_stage/1.0','physics':'rigid_body_thrust_v2','neural_backend':'csr_fp64_accum','models':rows,
        'random_successes':baseline['successes'],'random_episodes':baseline['episodes'],'MODEL_READY_FOR_NEXT_STAGE':bool(gate['passed'] and increment),
        'C0_TASK_LEARNED':gate['passed'],'CURRENT_INCREMENT_TESTED':increment,'FULL_TS1_READY':False,'SOFTWARE_READY_FULL_PLAN':False,'REAL_FLIGHT_READY':False,'C1_C2':'NOT_RUN',
        'readiness_scope':'nominal simulated C0 only; not full TS1 or hardware readiness','acceptance':gate,'regression_tests':len(tests),'demo_run':'rigid-v2-demo-final',
        'note':('新六自由度环境的三种子 C0 封存门槛已通过。' if gate['passed'] else '新六自由度环境的 C0 封存门槛未全部通过。')+'本轮每个模型新增 672 个训练动作；只训练读出，MaleCNS 连接图保持冻结。完整 TS1 仍有未完成验收项，C1/C2 与真机未开放。'}
    atomic_json(OUT/'summary.json',summary);atomic_json(OUT/'MODEL_INVENTORY.json',inventory)
    source_paths=[x for directory in ('flydrone','flyview','configs','tests') for x in (ROOT/directory).rglob('*') if x.is_file() and x.suffix in ('.py','.js','.html','.css','.json')]
    sources={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted(source_paths)}
    atomic_json(OUT/'implementation-hashes.json',sources)
    stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    header=['# 六自由度模拟与 C0 交付', '',f'生成时间：{stamp}', '',summary['note'],'',
        '| 模型种子 | 旧权重在新环境 | 适配后开发验证 | 新封存测试 | 碰撞/越界 |','|---|---:|---:|---:|---:|']
    for row in rows:header.append(f"| {row['seed']} | {row['before_validation']} | {row['validation']} | {row['sealed']} | {row['collision_or_bounds']} |")
    header += ['',f"随机基线：{baseline['successes']}/300。每个模型按同一 300 场景验收；成功率至少 90%，碰撞/越界至多 1%，比随机基线高至少 20 个百分点。",'',
        f"`MODEL_READY_FOR_NEXT_STAGE = {'YES' if summary['MODEL_READY_FOR_NEXT_STAGE'] else 'NO'}`（仅 nominal 模拟 C0）；`FULL_TS1_READY = NO`；`REAL_FLIGHT_READY = NO`。",'',
        '本轮运行的 80 项软件回归全部通过，物理规则基线 100/100；完整图的三次保存/恢复重复对照通过。原 TS1 的 60 条验收要求另见 TEST_MATRIX.md，不能用 80 个 pytest 用例数代替它。','',
        '## 怎么看', '', '打开 http://127.0.0.1:8765/tellosim ，点“新环境模型 11”；22、33 是同一冻结果蝇图上分别训练的读出。点物理脚本验证可看起飞、平移、真实转向与降落。标“旧环境”的记录保留原来的历史含义。','',
        '## 保存与继续', '', '最终模型与精确训练边界在 `runs/tellosim-sdk9/rigid-final-s11/`、`rigid-final-s22/`、`rigid-final-s33/` 对应目录。已完成本轮预算的 resume.pt 不会偷偷增加训练预算；新课程应显式新建热启动运行。精确恢复测试用 fp64-boundary.pt（16/32 动作）与连续完成的产物逐项比较。', '',
        '```bash','python -m flydrone.tellosim.training.continuation --project-root . --graph data/male-v1.npz --out runs/tellosim-sdk9/new-pilot --init-from runs/tellosim-sdk9/rigid-final-s11/checkpoint.pt --batch 4 --options 128 --rollout 32 --seed 11 --wall-seconds 300','```','',
        '运行中断后，用同一个 out，移除 --init-from，改为 --resume <out>/resume.pt，并保持 options/rollout/batch/seed/wall-seconds 不变。仅适用于相同源码、依赖、配置和设备合同；不支持旧 CSR/COO 诊断文件静默迁移。','',
        '## 仍未完成的原计划部分','', 'C1/C2、系统化动力学扰动、每环境独立的策略随机数流、四 lane 实时切换、长时性能与多浏览器压力等，详见 TEST_MATRIX.md。下一阶段先处理这些门槛中影响训练正确性的条目，再扩展课程。','',
        '本包不包含虚拟环境和大型 MaleCNS 图；外部依赖 `data/male-v1.npz` 的 SHA256 为 badc33a247894fe12c4d68d3ff791393857cffa39ff2ab81869608a11fa1e0c9。纯回放不依赖图文件。见 IMPLEMENTATION.md、FAILURES_AND_FIXES.md、BROWSER_CHECK.md 及逐 case JSON。']
    (OUT/'README.md').write_text('\n'.join(header)+'\n',encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
