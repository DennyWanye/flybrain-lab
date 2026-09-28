from pathlib import Path
import json
from flydrone.tellosim.training.c0_campaign import load_agent,evaluate
ROOT=Path('.').resolve();OUT=ROOT/'runs/tellosim-sdk9/c0-s11-20260927'
agent=load_agent(ROOT,ROOT/'data/male-v1.npz',OUT/'checkpoint.pt')
cases=json.loads((ROOT/'reports/sdk9_c0/cases.json').read_text())['validation']
evaluate(ROOT,agent,[c for c in cases if c['case_id']=='c0-validation-009'],OUT/'failure-009.json','failure-009')
