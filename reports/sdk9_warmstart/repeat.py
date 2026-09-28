"""Sequential fixed-budget replication; subprocesses release GPU memory after each run."""
from pathlib import Path
import json
import subprocess
import sys
from flydrone.tellosim.visual import atomic_json
root=Path('.').resolve();base=root/'reports/sdk9_warmstart'
protocol=json.loads((base/'protocol.json').read_text())
def execute(args,log):
    with log.open('w') as stream:
        child=subprocess.Popen([sys.executable,'-u',*args],cwd=root,stdout=stream,stderr=subprocess.STDOUT)
        try:
            if child.wait():raise RuntimeError(f'failed: {log}')
        finally:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait()
for seed in (23,37):
    report=base/f'seed-{seed}';report.mkdir(exist_ok=False)
    source=f'runs/tellosim-sdk9/nearx-bootstrap-s{seed}-20260926/checkpoint.pt'
    output=f'runs/tellosim-sdk9/nearx-warmstart-s{seed}-20260926'
    protocol=json.loads((base/'protocol.json').read_text())
    atomic_json(report/'protocol.json',{**protocol,'seed':seed,'source':source})
    print(json.dumps({'seed':seed,'phase':'training'}),flush=True)
    execute(['-m','flydrone.tellosim.training.runner','--graph','data/male-v1.npz',
        '--device','cuda','--profile','balanced_rate_v3','--curriculum','C0-near-x',
        '--options','512','--rollout','128','--eval-cases','16','--seed',str(seed),
        '--init-from',source,'--critic-episodes','16','--freeze-body',
        '--learning-rate','0.00003','--no-publish','--out',output],report/'train.log')
    for label,checkpoint in [('before',source),('after',output+'/checkpoint.pt')]:
        print(json.dumps({'seed':seed,'phase':'sampled-'+label}),flush=True)
        execute(['-m','flydrone.tellosim.training.evaluation','--graph','data/male-v1.npz',
            '--checkpoint',checkpoint,'--out',str(report/f'sampled-{label}.json'),
            '--case-count','16','--repeats','1','--sample-only'],report/f'sampled-{label}.log')
    execute(['-m','reports.sdk9_warmstart.build_report','--seed',str(seed)],report/'audit.log')
    print(json.dumps({'seed':seed,'phase':'completed'}),flush=True)
results=[json.loads((base/('summary.json' if seed==11 else f'seed-{seed}/summary.json')).read_text()) for seed in (11,23,37)]
retained=all(r['PILOT_RETENTION_CHECK_PASSED'] for r in results)
atomic_json(base/'multiseed-summary.json',{'schema':'sdk9.warmstart_multiseed/1.0','seeds':[11,23,37],
    'results':results,'RETENTION_CHECK_PASSED':retained,'MODEL_READY_FOR_NEXT_STAGE':False,
    'sealed_test_opened':False,'scope':'Three seeds on the same 16 validation scenarios; exploratory retention check, not 48 independent scenarios.'})
print(json.dumps({'phase':'all_completed','retention':retained}),flush=True)
