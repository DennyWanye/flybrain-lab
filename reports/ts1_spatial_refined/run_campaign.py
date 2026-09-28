"""Bounded training then frozen evaluation; children are scoped and always reaped."""
import subprocess,sys,time,json,signal
from pathlib import Path
root=Path.cwd();out=root/'reports/ts1_spatial_refined';jobs=[];history={}
if (out/'process-cleanup.json').exists():raise FileExistsError('campaign already attempted; preserve evidence')
smoke=json.loads((root/'runs/tellosim-sdk9/spatial-refined-smoke-s11/training.json').read_text())
assert smoke['status']=='completed' and smoke['options']==640 and all(v['roundtrip_exact'] and v['options']>0 for v in smoke['skills'].values())
development=json.loads((out/'smoke-evaluation.json').read_text())
assert development['episodes']==8 and all(r['brain_tick']==4*(1+r['physics_ticks']//12) for r in development['results'])
def stop(signum,frame):raise KeyboardInterrupt('campaign interrupted')
signal.signal(signal.SIGTERM,stop)
def phase(specs,budget):
    jobs=[]
    try:
        for name,args in specs:
            log=(out/(name+'.log')).open('x');p=subprocess.Popen([sys.executable,'-u','-m','flydrone.tellosim.training.spatial_refined_campaign',*args],stdout=log,stderr=subprocess.STDOUT)
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
subprocess.run([sys.executable,'-m','flydrone.tellosim.training.spatial_refined_campaign','prepare'],check=True)
phase([(f'train-{s}',['train','--seed',str(s)]) for s in (11,22,33)],9000)
subprocess.run([sys.executable,'-m','flydrone.tellosim.training.spatial_refined_campaign','freeze'],check=True)
phase([('baselines',['baselines'])]+[(f'evaluate-{s}',['evaluate','--seed',str(s)]) for s in (11,22,33)],9000)
subprocess.run([sys.executable,'-m','flydrone.tellosim.training.spatial_refined_campaign','finalize'],check=True)
