from pathlib import Path
import json,os
from flydrone.tellosim.training.spatial_stationary import spatial_case
from flydrone.tellosim.training.spatial_stationary_campaign import baseline,evaluate
from flydrone.tellosim.training.altitude_refined import with_profile,PROFILES
root=Path.cwd();out=root/'reports/ts1_spatial_stationary'
cases=[with_profile(spatial_case(490000000+i),tuple(PROFILES)[i%4]) for i in range(8)]
(out/'development-cases.json').write_text(json.dumps(cases,indent=2))
baseline(root,json.loads((root/'reports/ts1_rule_development/cases.json').read_text()),'rule',out/'development-rule128.json')
assert json.loads((out/'development-rule128.json').read_text())['successes']>=127
evaluate(root,root/'runs/tellosim-sdk9/spatial-stationary-s11/checkpoint.pt',cases,out/'smoke-evaluation.json','spatial-stationary-development',batch=8)
(out/'owned-development-memory-sample.json').write_text(json.dumps({'pid':os.getpid(),'private_kib':sum(int(r.split()[1]) for r in Path('/proc/self/smaps_rollup').read_text().splitlines() if r.startswith(('Private_Clean:','Private_Dirty:')))}))
