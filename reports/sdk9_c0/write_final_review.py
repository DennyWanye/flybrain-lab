from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib
ROOT=Path('.').resolve();OUT=ROOT/'reports/sdk9_c0'
report=json.loads((OUT/'summary.json').read_text())
lines=['# 三模型诊断与失败清单','','部署模式为 argmax；采样结果仅作探索诊断。故障表独立于 nominal 封存成功率。','',
'| 种子 | 近距离固定动作 | 近距离概率采样 |','|---|---:|---:|']
score=lambda r: f"{r['successes']}/{r['episodes']}" if r else 'NOT_RUN'
for m in report['models']:lines.append(f"| {m['seed']} | {score(m['near_regression'])} | {score(m['near_sampled'])} |")
lines+=['','## 故障诊断','','定位丢失或通信状态不明时，保护终止是可接受的安全行为，但不能计作导航成功。噪声和延迟下的低成功率也不会隐藏。','',
'| 种子 | 故障条件 | 成功 | 终止原因计数 |','|---|---|---:|---|']
for m in report['models']:
    groups=defaultdict(list)
    for r in (m['faults'] or {}).get('results',[]):groups[r['case_id'].rsplit('-',1)[0]].append(r)
    for name,rows in groups.items():
        reasons=', '.join(f'{k}: {v}' for k,v in sorted(Counter(r['reason'] for r in rows).items()))
        lines.append(f"| {m['seed']} | {name} | {sum(r['success'] for r in rows)}/{len(rows)} | {reasons} |")
lines+=['','## 全部开发与封存失败','','| 种子 | 数据集 | case | 原因 | 终止距离 m |','|---|---|---|---|---:|']
for m in report['models']:
    for split in ('validation','sealed'):
        for r in (m[split] or {}).get('results',[]):
            if not r['success']:lines.append(f"| {m['seed']} | {split} | {r['case_id']} | {r['reason']} | {r['distance_m']:.4f} |")
lines+=['','失败后未对这些最终权重继续训练；所有正式结果留存。模型 11 的开发失败复现细节另见 FAILURES.md。']
(OUT/'DIAGNOSTICS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
inventory=[]
for m in report['models']:
    p=ROOT/m['directory']/'checkpoint.pt'
    if p.exists():inventory.append({'seed':m['seed'],'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'training_options':m['training']['options'],'training_elapsed_s':m['training']['elapsed_s']})
(OUT/'MODEL_INVENTORY.json').write_text(json.dumps(inventory,indent=2))
files=[]
for name in ('flydrone/tellosim','flyview','tests/flydrone'):
    files.extend(p for p in (ROOT/name).rglob('*') if p.is_file() and p.suffix in ('.py','.js','.css','.html') and '__pycache__' not in p.parts)
(OUT/'implementation-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},indent=2))
print(json.dumps({'diagnostics_written':True,'models':len(inventory)}))
