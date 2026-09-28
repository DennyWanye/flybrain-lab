import subprocess,sys,time,json,signal
from pathlib import Path
out=Path('reports/ts1_mlp_pilot');history={}
if (out/'process-cleanup.json').exists():raise FileExistsError('pilot already attempted')
assert json.loads((out/'smoke-evaluation.json').read_text())['episodes']==8
p=None

def stop(signum,frame):raise KeyboardInterrupt('pilot interrupted')
signal.signal(signal.SIGTERM,stop)
try:
    for phase in ['train','evaluate']:
        log=(out/f'{phase}.log').open('x');p=subprocess.Popen([sys.executable,'-u','-m','flydrone.tellosim.training.direct_mlp',phase],stdout=log,stderr=subprocess.STDOUT)
        (out/'active-process.json').write_text(json.dumps({'pid':p.pid,'phase':phase}));print(json.dumps({'phase':phase,'pid':p.pid}),flush=True)
        try:
            p.wait(timeout=3600)
            if p.returncode:raise RuntimeError(f'{phase} failed: see log')
        finally:
            if p.poll() is None:p.terminate()
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            log.close();history[phase]={'pid':p.pid,'returncode':p.returncode,'reaped':True,'proc_absent':not Path(f'/proc/{p.pid}').exists()}
            (out/'process-cleanup.json').write_text(json.dumps(history,indent=2));(out/'active-process.json').write_text('{}')
finally:
    if p is not None and p.poll() is None:p.kill();p.wait()
