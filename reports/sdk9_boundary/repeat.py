from pathlib import Path
import json
import subprocess
import sys
root=Path('.').resolve();reports=root/'reports/sdk9_boundary'
pilot=json.loads((root/'runs/tellosim-sdk9/boundary-repair-s23-recovery-20260926/summary.json').read_text())
if not (pilot['KNOWN_FAILURE_REPAIRED'] and pilot['NO_CASE_REGRESSIONS']):
    raise RuntimeError('Pilot did not pass. Do not replicate an unsuccessful repair.')
for seed in (11,37):
    with (reports/f's{seed}-recovery.log').open('w') as log:
        process=subprocess.Popen([sys.executable,'-u','-m','flydrone.tellosim.training.boundary_repair',
            '--graph','data/male-v1.npz','--seed',str(seed),'--tag','recovery-20260926','--recovery'],stdout=log,stderr=subprocess.STDOUT)
        try:
            if process.wait():raise RuntimeError(f'seed {seed} failed')
        finally:
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()
    print(json.dumps({'completed_seed':seed}),flush=True)
