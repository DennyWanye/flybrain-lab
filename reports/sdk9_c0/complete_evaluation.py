"""Open the sealed set once, after all formal checkpoints are frozen."""
import json,hashlib,subprocess,sys
from pathlib import Path
from flydrone.tellosim.training.contracts import digest
from flydrone.tellosim.visual import atomic_json
from run_campaign import call
from evidence_checks import fault_safety
ROOT=Path('.').resolve();OUT=ROOT/'reports/sdk9_c0'

def evaluation(checkpoint,out,split='validation',mode='argmax'):
    if out.exists():
        saved=json.loads(out.read_text())
        assert saved['checkpoint_sha256']==hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        return
    call(['flydrone.tellosim.training.c0_campaign','evaluate','--checkpoint',str(checkpoint),
        '--out',str(out),'--split',split,'--mode',mode],out.parent.name+'-'+out.stem+'.log')

def main():
    state=json.loads((OUT/'campaign-state.json').read_text())
    if state['status']!='all_models_development_passed':raise RuntimeError('development gate has not passed')
    models=[ROOT/f'runs/tellosim-sdk9/c0-s{s}-20260927' for s in (11,22,33)]
    for path in models:
        train=json.loads((path/'training.json').read_text());valid=json.loads((path/'validation.json').read_text())
        assert train['status']=='completed' and valid['episodes']==100 and valid['success_rate']>=.9 and valid['collision_or_bounds']<=1
    subprocess.run([sys.executable,str(OUT/'audit_models.py')],cwd=ROOT,check=True)
    assert json.loads((OUT/'artifact-audit.json').read_text())['complete_three_seeds']
    for path in models:
        evaluation(path/'checkpoint.pt',path/'near-regression.json','near-regression')
        evaluation(path/'checkpoint.pt',path/'near-sampled.json','near-regression','sampled')
        evaluation(path/'checkpoint.pt',path/'faults.json','faults')
    for path in models:
        near=json.loads((path/'near-regression.json').read_text())
        assert near['successes']==near['episodes']==40,'previous deterministic near-axis skill regressed; keep sealed set unopened'
    for path in models:
        assert fault_safety(json.loads((path/'faults.json').read_text())),'fault protection check failed; keep sealed set unopened'
    cases=json.loads((OUT/'cases.json').read_text())
    checkpoints=[p/'checkpoint.pt' for p in models]+[models[0]/'initial.pt']
    lock={'mode':'argmax','selection':'fixed final checkpoint for each independently initialized seed',
        'case_hash':digest(cases['sealed_test']),
        'checkpoint_hashes':[hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoints],
        'checkpoints':[str(p.relative_to(ROOT)) for p in checkpoints]}
    lockfile=OUT/'frozen-checkpoints.json'
    if lockfile.exists():assert json.loads(lockfile.read_text())==lock
    else:atomic_json(lockfile,lock)
    # Freeze everything before any sealed evaluation. No training follows these
    # scores in this campaign and failed results are never overwritten.
    evaluation(models[0]/'initial.pt',OUT/'random-sealed.json','sealed_test','random')
    for path in models:evaluation(path/'checkpoint.pt',path/'sealed.json','sealed_test')
    evaluation(ROOT/'runs/tellosim-sdk9/c0-raw-ppo-s11-20260927/checkpoint.pt',OUT/'raw-mlp-validation.json')
    evaluation(models[0]/'initial.pt',OUT/'untrained-validation.json')
    atomic_json(OUT/'evaluation-state.json',{'status':'completed','sealed_cases':300,'seeds':[11,22,33]})
if __name__=='__main__':main()
