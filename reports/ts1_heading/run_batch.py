"""Bounded H1 run: preserve incomplete evidence and stop only owned children."""
from pathlib import Path
import subprocess,sys,time,json,signal
from flydrone.tellosim.training.heading_campaign import freeze,finalize
ROOT=Path.cwd();OUT=ROOT/'reports/ts1_heading';jobs=[];handles=[]
def launch(phase):
    result=[]
    for seed in (11,22,33):
        log=(OUT/f'{phase}-s{seed}.log').open('w');handles.append(log)
        p=subprocess.Popen([sys.executable,'-u','-m','flydrone.tellosim.training.heading_campaign',phase,'--seed',str(seed)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        jobs.append(p);result.append((seed,p))
    (OUT/f'{phase}-processes.json').write_text(json.dumps({str(s):p.pid for s,p in result},indent=2))
    return result
def wait(result,seconds):
    deadline=time.monotonic()+seconds
    while any(p.poll() is None for _,p in result):
        for seed,p in result:
            if p.poll() not in (None,0):raise RuntimeError(f'seed {seed} process failed: {p.returncode}')
        if time.monotonic()>deadline:raise TimeoutError('batch budget exhausted')
        time.sleep(2)
    assert all(p.returncode==0 for _,p in result)
try:
    deadline=time.monotonic()+240
    while not (OUT/'rule-validation.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('rule baseline absent')
        time.sleep(2)
    assert json.loads((OUT/'rule-validation.json').read_text())['successes']>=99
    print('RULE_VALIDATED_START_TRAINING',flush=True)
    wait(launch('train'),1900)
    print(json.dumps({'phase':'all-models-frozen','lock':freeze(ROOT)}),flush=True)
    wait(launch('evaluate'),3600)
    if not (OUT/'random-sealed.json').exists():raise RuntimeError('random baseline incomplete')
    finalize(ROOT);print('HEADING_FORMAL_EVALUATION_COMPLETE',flush=True)
finally:
    for p in jobs:
        if p.poll() is None:p.send_signal(signal.SIGINT)
    for p in jobs:
        if p.poll() is None:
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:p.kill();p.wait(timeout=10)
    for h in handles:h.close()
    (OUT/'process-cleanup.json').write_text(json.dumps({'owned_pids':[p.pid for p in jobs],'all_exited':all(p.poll() is not None for p in jobs)},indent=2))
