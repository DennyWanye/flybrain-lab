"""Bounded owned-process evaluation after main implementation and smoke success."""
import subprocess,sys,time,json,signal
from pathlib import Path
root=Path.cwd();out=root/'reports/ts1_joint';jobs=[]
def stop(signum,frame):raise KeyboardInterrupt('campaign interrupted')
signal.signal(signal.SIGTERM,stop)
subprocess.run([sys.executable,'-m','flydrone.tellosim.training.joint_campaign','prepare'],check=True)
try:
    for name,args in [('baselines',['baselines'])]+[(f'evaluate-{s}',['evaluate','--seed',str(s)]) for s in (11,22,33)]:
        log=(out/(name+'.log')).open('w');p=subprocess.Popen([sys.executable,'-u','-m','flydrone.tellosim.training.joint_campaign',*args],stdout=log,stderr=subprocess.STDOUT)
        jobs.append((name,p,log));print(json.dumps({'job':name,'pid':p.pid}),flush=True)
    (out/'owned-processes.json').write_text(json.dumps({n:p.pid for n,p,l in jobs},indent=2))
    deadline=time.monotonic()+3600
    while any(p.poll() is None for _,p,_ in jobs):
        if time.monotonic()>deadline:raise TimeoutError('campaign wall budget')
        if any(p.poll() not in (None,0) for _,p,_ in jobs):raise RuntimeError('evaluation failed; see scoped logs')
        time.sleep(5)
    assert all(p.returncode==0 for _,p,_ in jobs)
    subprocess.run([sys.executable,'-m','flydrone.tellosim.training.joint_campaign','finalize'],check=True)
finally:
    for _,p,_ in jobs:
        if p.poll() is None:p.terminate()
    for _,p,l in jobs:
        try:p.wait(timeout=10)
        except subprocess.TimeoutExpired:p.kill();p.wait()
        l.close()
    (out/'process-cleanup.json').write_text(json.dumps({n:{'pid':p.pid,'returncode':p.returncode,'reaped':p.poll() is not None} for n,p,l in jobs},indent=2))
