import json,subprocess,sys
from pathlib import Path
from flydrone.tellosim.visual import atomic_json
ROOT=Path('.').resolve();OUT=ROOT/'reports/sdk9_c0'

def call(arguments,log):
    with (OUT/log).open('w') as stream:
        child=subprocess.Popen([sys.executable,'-u','-m',*arguments],stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT)
        print(json.dumps({'started':arguments,'pid':child.pid}),flush=True)
        try:
            code=child.wait()
            if code:raise RuntimeError(f'{log}: child exited {code}')
        finally:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait()

def main():
    rule=json.loads((OUT/'rule-validation.json').read_text())
    assert rule['episodes']==100 and rule['success_rate']>=.99
    raw=ROOT/'runs/tellosim-sdk9/c0-raw-ppo-s11-20260927'
    if not (raw/'summary.json').exists():
        call(['flydrone.tellosim.training.runner','--graph','data/male-v1.npz','--device','cuda',
            '--profile','balanced_rate_v3','--curriculum','C0','--feature-source','raw_observation_control',
            '--seed','11','--out',str(raw),'--options','1536','--rollout','128','--eval-cases','8','--no-publish'],'raw-ppo.log')
    module='flydrone.tellosim.training.c0_campaign'
    for seed in (11,22,33):
        path=ROOT/f'runs/tellosim-sdk9/c0-s{seed}-20260927'
        if not (path/'training.json').exists():
            call([module,'train','--seed',str(seed),'--out',str(path)],f'train-s{seed}.log')
        train=json.loads((path/'training.json').read_text())
        if train['status']!='completed':raise RuntimeError(f'seed {seed}: training budget reached')
        if not (path/'validation.json').exists():
            call([module,'evaluate','--checkpoint',str(path/'checkpoint.pt'),'--out',str(path/'validation.json')],f'validation-s{seed}.log')
        valid=json.loads((path/'validation.json').read_text())
        if valid['success_rate']<.9 or valid['collision_or_bounds']>1:
            atomic_json(OUT/'campaign-state.json',{'status':'development_gate_failed','seed':seed,'successes':valid['successes']})
            print(json.dumps({'status':'development_gate_failed','seed':seed,'successes':valid['successes']}),flush=True)
            return
    atomic_json(OUT/'campaign-state.json',{'status':'all_models_development_passed'})
if __name__=='__main__':main()
