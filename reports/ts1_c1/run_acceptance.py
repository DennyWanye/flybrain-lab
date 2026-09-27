"""Wait for this bounded training batch, then freeze and evaluate all three seeds."""
from pathlib import Path
import subprocess,sys,time,json,os,signal
from flydrone.tellosim.training.c1_campaign import freeze
ROOT=Path('.').resolve();OUT=ROOT/'reports/ts1_c1';started=time.monotonic();jobs=[];handles=[]
while True:
    states=[]
    for seed in (11,22,33):
        directory=ROOT/f'runs/tellosim-sdk9/c1-s{seed}'
        if (directory/'failure.json').exists():raise RuntimeError(f'training seed {seed} failed; do not evaluate incomplete candidate')
        f=directory/'training.json'
        states.append(json.loads(f.read_text()) if f.exists() else None)
    if all(states):
        assert all(s['status']=='completed' and s['options']==1536 for s in states),states
        break
    if time.monotonic()-started>2400:raise TimeoutError('bounded training did not finish')
    time.sleep(2)
lock=freeze(ROOT);print(json.dumps({'phase':'all-models-frozen','models':lock['models']}),flush=True)
try:
    for seed in (11,22,33):
        log=(OUT/f'eval-s{seed}.log').open('w');handles.append(log)
        p=subprocess.Popen([sys.executable,'-u','-m','flydrone.tellosim.training.c1_campaign','evaluate','--seed',str(seed)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((seed,p))
    (OUT/'evaluation-processes.json').write_text(json.dumps({'pids':{s:p.pid for s,p in jobs},'frozen':lock},indent=2))
    deadline=time.monotonic()+3600
    while any(p.poll() is None for _,p in jobs):
        for seed,p in jobs:
            if p.poll() not in (None,0):raise RuntimeError(f'evaluation seed {seed} failed with {p.returncode}')
        if time.monotonic()>deadline:raise TimeoutError('evaluation wall budget')
        time.sleep(2)
    assert all(p.returncode==0 for _,p in jobs)
    subprocess.run([sys.executable,'-m','flydrone.tellosim.training.c1_campaign','finalize'],cwd=ROOT,check=True)
    print('FORMAL_ACCEPTANCE_COMPLETE',flush=True)
finally:
    for _,p in jobs:
        if p.poll() is None:p.send_signal(signal.SIGINT);p.wait(timeout=30)
    for handle in handles:handle.close()
