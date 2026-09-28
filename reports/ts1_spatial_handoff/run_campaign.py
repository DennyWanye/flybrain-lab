"""Bounded training then frozen evaluation; children are scoped and always reaped."""
import subprocess,sys,time,json,signal
from pathlib import Path
root=Path.cwd();out=root/'reports/ts1_spatial_handoff';jobs=[];history={}
if (out/'process-cleanup.json').exists():raise FileExistsError('campaign already attempted; preserve evidence')
def stop(signum,frame):raise KeyboardInterrupt('campaign interrupted')
signal.signal(signal.SIGTERM,stop)
def phase(specs,budget):
    jobs=[]
    try:
        for name,args in specs:
            log=(out/(name+'.log')).open('x');p=subprocess.Popen([sys.executable,'-u','-m','flydrone.tellosim.training.spatial_handoff_campaign',*args],stdout=log,stderr=subprocess.STDOUT)
            jobs.append((name,p,log));(out/'active-processes.json').write_text(json.dumps([{'job':n,'pid':q.pid} for n,q,_ in jobs]));print(json.dumps({'job':name,'pid':p.pid}),flush=True)
        deadline=time.monotonic()+budget
        while any(p.poll() is None for _,p,_ in jobs):
            if time.monotonic()>deadline:raise TimeoutError('phase wall budget')
            if any(p.poll() not in (None,0) for _,p,_ in jobs):raise RuntimeError('child failed; inspect scoped logs')
            time.sleep(2)
        assert all(p.returncode==0 for _,p,_ in jobs)
    finally:
        for _,p,_ in jobs:
            if p.poll() is None:p.terminate()
        for name,p,log in jobs:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            log.close();history[name]={'pid':p.pid,'returncode':p.returncode,'reaped':True,'proc_absent':not Path(f'/proc/{p.pid}').exists()}
        (out/'process-cleanup.json').write_text(json.dumps(history,indent=2))
        (out/'active-processes.json').write_text('[]')
phase([('transfer',['transfer'])],600)
# Development checks use a separate helper, with the exact transferred weights.
log=(out/'development.log').open('x')
p=subprocess.Popen([sys.executable,'-u','-m','reports.ts1_spatial_handoff.development_check'],stdout=log,stderr=subprocess.STDOUT)
try:p.wait(timeout=1800);assert p.returncode==0
finally:
    if p.poll() is None:p.terminate()
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:p.kill();p.wait()
    log.close();history['development']={'pid':p.pid,'returncode':p.returncode,'reaped':True,'proc_absent':not Path(f'/proc/{p.pid}').exists()};(out/'process-cleanup.json').write_text(json.dumps(history,indent=2))
small=json.loads((out/'smoke-evaluation.json').read_text());rule=json.loads((out/'development-rule128.json').read_text())
assert small['successes']>0 and small['episodes']==8
assert rule['successes']>=127 and rule['episodes']==128
# Let the preserved C2Q campaign finish before another full evaluation phase.
deadline=time.monotonic()+7200
while not (root/'reports/ts1_spatial_orientation/summary.json').exists():
    if time.monotonic()>deadline:raise TimeoutError('prior campaign did not finalize')
    time.sleep(5)
phase([('baselines',['baselines'])]+[(f'evaluate-{s}',['evaluate','--seed',str(s)]) for s in (11,22,33)],9000)
subprocess.run([sys.executable,'-m','flydrone.tellosim.training.spatial_handoff_campaign','finalize'],check=True)
