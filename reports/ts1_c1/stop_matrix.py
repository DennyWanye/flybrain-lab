"""Exercise real process interruption and bounded-wall resume, not simulated flags."""
from pathlib import Path
import subprocess,sys,time,signal,json,torch
ROOT=Path('.').resolve();OUT=ROOT/'reports/ts1_c1';results=[]
base=[sys.executable,'-u','-m','flydrone.tellosim.training.continuation','--batch','4','--rollout','8','--options','64','--seed','91','--curriculum','C1']
source=ROOT/'runs/tellosim-sdk9/rigid-final-s11/checkpoint.pt'
for mode in ('sigint','sigterm','wall'):
    directory=OUT/f'stop-{mode}';wall='3' if mode=='wall' else '180'
    log=(OUT/f'stop-{mode}.log').open('w');cmd=base+['--out',str(directory),'--wall-seconds',wall,'--init-from',str(source)]
    proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);deadline=time.monotonic()+120
    try:
        if mode!='wall':
            while not (directory/'progress.json').exists():
                if proc.poll() is not None:raise RuntimeError(f'{mode} exited early {proc.returncode}')
                if time.monotonic()>deadline:raise TimeoutError(mode)
                time.sleep(.05)
            proc.send_signal(signal.SIGINT if mode=='sigint' else signal.SIGTERM)
        code=proc.wait(timeout=40);assert code==0,mode
        report=json.loads((directory/'summary.json').read_text());snap=torch.load(directory/'resume.pt',weights_only=False)
        assert report['committed_options']==snap['counters']['options']
        assert report['this_process_activity']['attempted_options']>=report['committed_options']
        results.append({'mode':mode,'pid':proc.pid,'exited':True,'status':report['status'],'committed':report['committed_options'],'attempted':report['this_process_activity']['attempted_options']})
        if mode=='wall':
            args=base+['--out',str(directory),'--wall-seconds',wall,'--resume',str(directory/'resume.pt')]
            subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=40)
            resumed=json.loads((directory/'summary.json').read_text());assert resumed['this_process_activity']['attempted_options']==0
            results[-1]['wall_resume_cannot_reset_budget']=True
    finally:
        if proc.poll() is None:proc.send_signal(signal.SIGINT);proc.wait(timeout=30)
        log.close()
(OUT/'stop-matrix.json').write_text(json.dumps({'passed':True,'results':results},indent=2));print(json.dumps(results))
