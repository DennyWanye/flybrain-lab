"""Bounded training then frozen evaluation; children are scoped and always reaped."""
import subprocess,sys,time,json,signal
from pathlib import Path
root=Path.cwd();out=root/'reports/ts1_stability';jobs=[];history={}
def stop(signum,frame):raise KeyboardInterrupt('campaign interrupted')
signal.signal(signal.SIGTERM,stop)
def phase(specs,budget):
    jobs=[]
    try:
        for name,args in specs:
            log=(out/(name+'.log')).open('w');p=subprocess.Popen([sys.executable,'-u','-m','flydrone.tellosim.training.stability_campaign',*args],stdout=log,stderr=subprocess.STDOUT)
            jobs.append((name,p,log));print(json.dumps({'job':name,'pid':p.pid}),flush=True)
        deadline=time.monotonic()+budget
        while any(p.poll() is None for _,p,_ in jobs):
            if time.monotonic()>deadline:raise TimeoutError('phase wall budget')
            if any(p.poll() not in (None,0) for _,p,_ in jobs):raise RuntimeError('child failed; inspect scoped logs')
            time.sleep(5)
        assert all(p.returncode==0 for _,p,_ in jobs)
    finally:
        for _,p,_ in jobs:
            if p.poll() is None:p.terminate()
        for name,p,log in jobs:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            log.close();history[name]={'pid':p.pid,'returncode':p.returncode,'reaped':True}
        (out/'process-cleanup.json').write_text(json.dumps(history,indent=2))
subprocess.run([sys.executable,'-m','flydrone.tellosim.training.stability_campaign','prepare'],check=True)
phase([(f'train-{s}',['train','--seed',str(s)]) for s in (11,22,33)],5400)
subprocess.run([sys.executable,'-m','flydrone.tellosim.training.stability_campaign','freeze'],check=True)
phase([('baselines',['baselines'])]+[(f'evaluate-{s}',['evaluate','--seed',str(s)]) for s in (11,22,33)],7200)
subprocess.run([sys.executable,'-m','flydrone.tellosim.training.stability_campaign','finalize'],check=True)
