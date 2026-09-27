"""Render evidence-driven final C1 report; no model mutation."""
from pathlib import Path
import collections,hashlib,json,datetime,subprocess
import torch
ROOT=Path('.').resolve();OUT=ROOT/'reports/ts1_c1'
def read(name):return json.loads((OUT/name).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    summary=read('summary.json');assert summary['status']=='evaluated' and len(summary['models'])==3
    summary['comparison']=read('comparison.json')['runs'];(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    from flydrone.tellosim.view_export import normalize_episode_identity
    identity_audit={}
    for folder in (ROOT/'reports/vis/tellosim').glob('c1-*'):
        path=folder/'manifest.json'
        if not path.exists():continue
        manifest=json.loads(path.read_text())
        if manifest.get('schema_version')=='tellosim.view/2.0' and manifest.get('complete') and manifest.get('case'):
            identity_audit[folder.name]=normalize_episode_identity(folder)
    (OUT/'VIEW_EXPORT_AUDIT.json').write_text(json.dumps(identity_audit,ensure_ascii=False,indent=2))
    cases=read('cases.json');lock=read('frozen-checkpoints.json');training={};failures={};actions={};source_checks={}
    sets={k:{c['seed'] for c in v} for k,v in cases.items() if isinstance(v,list)}
    assert all(not a&b for i,a in enumerate(sets.values()) for b in list(sets.values())[i+1:])
    for row in lock['models']:
        seed=row['seed'];run=ROOT/f'runs/tellosim-sdk9/c1-s{seed}'
        training[seed]=json.loads((run/'training.json').read_text());assert training[seed]['options']==1536 and training[seed]['status']=='completed'
        assert sha(ROOT/row['path'])==row['sha256']
        data=json.loads((run/'provenance.json').read_text());train_seeds={x['seed'] for x in data}
        assert all(not train_seeds & v for v in sets.values())
        state=torch.load(run/'ppo/resume.pt',map_location='cpu',weights_only=False)
        assert state['format']=='tellosim.exact_training/3'
        code_root=ROOT/'flydrone/tellosim'
        checks={name:sha(code_root/name)==expected for name,expected in state['contract']['sources'].items()}
        assert all(checks.values());source_checks[seed]=checks
        ppo_seeds={e['fields']['case']['seed'] for e in state['envs'] if e}
        assert all(not ppo_seeds & v for v in sets.values())
        result=read(f's{seed}-sealed_test.json');assert result['episodes']==300
        failures[seed]=[x for x in result['results'] if not x['success']]
        actions[seed]=dict(sorted(collections.Counter(a for x in result['results'] for a in x['actions']).items()))
    summary['readiness_scope']='random_initial_heading_navigation_with_C0_retention'
    summary['C1_CONTROLLED_TURNING_VERIFIED']=False
    summary['sealed_turn_actions']={seed:actions[seed].get(7,0)+actions[seed].get(8,0) for seed in training}
    summary['next_experiment']='target_heading_control_curriculum_before_claiming_learned_turning'
    if '本次三组封存评估' not in summary['note']:summary['note']+='本次三组封存评估的转向动作均为0；仅验证随机初始朝向导航，未证明主动转向到指定角度。'
    for row in summary['models']:row['sample_outcome']=read(f"s{row['seed']}-sealed_test.json")['results'][0]['reason']
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    audit={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'frozen_model_hashes_unchanged':True,'all_splits_disjoint':True,'training_examples_disjoint':True,'resume_source_checks':source_checks,'sealed_action_counts':actions,'training_options':{s:t['options'] for s,t in training.items()}}
    (OUT/'FINAL_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    paths=[]
    for folder in ('flydrone','flyview','tests'):
        paths += [p for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.js','.html','.css') and '__pycache__' not in p.parts]
    (OUT/'implementation-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)},indent=2))
    (OUT/'git-status.txt').write_text(subprocess.run(['git','status','--short'],cwd=ROOT,capture_output=True,text=True,check=True).stdout)
    (OUT/'failed-cases.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2))
    rows=read('TEST_MATRIX.json')
    for r in rows:
        if r['test_id']=='T43':r.update(actual='C1规则100/100；同C1封存集随机3/300（随机碰撞1例）',artifact='rule-validation.json / random-sealed.json',code_version='implementation-hashes.json')
        if r['test_id']=='T45':r.update(actual='C1同一300场封存：'+','.join(f"{m["seed"]}={m["sealed"]}" for m in summary['models'])+'；权重先锁定、后验收、未重调',artifact='s*-sealed_test.json / frozen-checkpoints.json',code_version='implementation-hashes.json',elapsed_s=sum(read(f's{s}-sealed_test.json')['elapsed_s'] for s in training))
    (OUT/'TEST_MATRIX.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    md=['# TS1 逐项验收','', 'PASS是该行范围的证据，不等于完整TS1已就绪。历史证据位于上一版报告；本轮新增验收见本目录。null耗时未补造。','', '| ID | 状态 | 输入/范围 | 实际结果 | 证据 |','|---|---|---|---|---|']
    md += [f"| {r['test_id']} | {r['status']} | {r['input']} | {r['actual']} | {r['artifact']} |" for r in rows]
    (OUT/'TEST_MATRIX.md').write_text('\n'.join(md)+'\n')
    lines=['# C1 随机朝向训练与可靠性验收','',summary['note'],'',f"C1_TASK_LEARNED = {summary['C1_TASK_LEARNED']}；MODEL_READY_FOR_NEXT_STAGE = {summary['MODEL_READY_FOR_NEXT_STAGE']}；FULL_TS1_READY = False；REAL_FLIGHT_READY = False。",'',
       '## 查看结果','', '打开 http://127.0.0.1:8765/tellosim ，先看 C1 模型22的成功样例，再看11的同场景超时样例和训练前/后对照。快捷回放固定取封存第1场，没有挑掉失败样例。22、33是同一个冻结连接图的另两组独立训练读出，不是三个不同的果蝇连接组。训练观察保留最后快照，不是完整回放。','',
       '## 正式成绩','', '| 训练种子 | C1验证100场 | C1封存300场 | 碰撞/越界 | C0保持：训练前→后100场 | 保持门槛 |','|---|---|---|---|---|---|']
    for m in summary['models']:
       lines.append(f"| {m['seed']} | {m['validation']} | {m['sealed']} | {m['collision_or_bounds']} | {m['retention_before']} → {m['retention_after']} | {m['retention_passed']} |")
    lines += ['', 'C1规则基线100/100；封存均匀随机3/300。每个模型还跑了100场C0保持，与相同场景的训练前权重对照。封存门槛：成功率≥90%、碰撞或越界≤1%、超过随机至少20个百分点；C0保持≥90%且退步≤5个百分点。所有三个最终权重在打开封存结果前同时锁定，没有根据封存成绩二次调参。','',
       '## 训练与边界','', '每种子1536个新增动作：384规则示范 + 2×512 DAgger纠错 + 128 PPO。每4个训练episode有1个C0练习；训练示范标签拟合率不能当导航成功率。MaleCNS 166700神经元、25582938条边，连接图冻结，只训练动作/价值读出。物理为六自由度机身推力近似。','',
       'C1随机初始yaw，允许顺/逆时针动作；成功只要求到达目标并稳定保持2秒，不要求最终机头朝向。实际三个种子的300场封存转向动作均为0，因此没有主动转向学习的证据。下一项应加入指定朝向目标与独立验收，再宣称转向能力。起飞/降落和SDK握手由任务管理器执行，不是学出来的。C2高度/障碍/扰动课程及实机尚未验收。','',
       '## 可靠性证据','', '主体代码跑通后执行86项回归全部通过；另外7项SDK边界/状态机/UDP/看门狗验收全部通过，加上2项导出身份校验，共95个不同测试。真实全图边界恢复的策略、优化器、脑、四环境、随机流、物理和传感器逐项一致。真实SIGINT/SIGTERM停止已测，SIGINT诊断从8动作恢复完成64动作；wall耗尽再次恢复不执行新动作。','',
       '五个HTTP读取者持续120秒，含慢读和重复连接，最大帧22062字节，95%请求延迟约3毫秒；这是HTTP压力验证，不是五个浏览器渲染器或0/30/60FPS对照。短32动作benchmark中batch4约2.76动作/秒、batch1约3.41动作/秒，不声称4环境加速，亦不声称已完成万动作长测。','',
       '浏览器实际验证环境切换、只读暂停与恢复、reply loss设备/客户端状态分离、无效定位不补零、未知schema拒绝以及Golden兼容。详见BROWSER_CHECK.md。','',
       '## 未完成的完整TS1条目','']
    lines += [f"- {r['test_id']}：{r['input']}。{r['actual']}。" for r in rows if r['status']=='NOT_RUN']
    lines += ['', '## 失败与复现','', 'failed-cases.json保存全部封存失败，不删失败样例。FINAL_AUDIT.json记录权重、数据划分、动作计数和resume源码检查；implementation-hashes.json锁定本轮源码。', '', '精确恢复使用runs/tellosim-sdk9/c1-sXX/ppo/resume.pt，需相同执行合同；已有预算用完，不会凭恢复自动增加训练。监督/DAgger阶段只保证阶段权重与数据保存，不宣称任意中途精确恢复。旧v2 resume不能静默恢复；旧C0结果只保留历史含义。','', '交付包包含代码、模型、评估、回放和依赖锁，完整大图与虚拟环境作为外部依赖。新目录解包验证使用现有WSL依赖，不是全新机器安装验证。PACKAGE_AUDIT.json为ZIP与逐文件校验记录。','',
       '本轮排查中首次非零朝向reset缺少mujoco导入、部分旧测试case未带case_id；已修复后才完成全套86项回归。早期smoke/perf诊断快照与最终源码合同不一致，交付排除，避免冒充可恢复模型。']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'audit':'passed','models':summary['models'],'matrix':dict(collections.Counter(r['status'] for r in rows))},ensure_ascii=False))
if __name__=='__main__':main()
