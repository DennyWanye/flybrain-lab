import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('.').resolve();OUT=ROOT/'reports/sdk9_c0'
started=time.monotonic()
while not (OUT/'campaign-state.json').exists():
    if time.monotonic()-started>10800:raise RuntimeError('training campaign did not finish within 3 hours')
    try:os.kill(5694,0)
    except ProcessLookupError:raise RuntimeError('training coordinator exited without a final state')
    time.sleep(5)
state=json.loads((OUT/'campaign-state.json').read_text())
if state['status']!='all_models_development_passed':
    print(json.dumps(state),flush=True);raise SystemExit(2)
print('All development gates passed; starting frozen acceptance workflow.',flush=True)
subprocess.run([sys.executable,'-u',str(OUT/'complete_evaluation.py')],cwd=ROOT,check=True)
for name in ('audit_models.py','finalize.py'):
    subprocess.run([sys.executable,str(OUT/name)],cwd=ROOT,check=True)
print('Evaluation and evidence assembly completed.',flush=True)
