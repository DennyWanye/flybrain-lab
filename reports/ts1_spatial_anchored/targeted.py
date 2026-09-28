from pathlib import Path
import json,os
from reports.ts1_control_development_v import diagnose as d
from flydrone.tellosim.training.spatial_anchored import SpatialEnv
from flydrone.tellosim.training.spatial_anchored_campaign import baseline
out=Path('reports/ts1_spatial_anchored')
d.OUT=out/'targeted';d.OUT.mkdir(exist_ok=False);d.SpatialEnv=SpatialEnv
(out/'targeted-process.json').write_text(json.dumps({'pid':os.getpid(),'cwd':str(Path.cwd())}))
cases=json.loads(Path('reports/ts1_rule_development/cases.json').read_text())
for index in [5,22,23,63,71,116,13,29]:d.episode(cases[index])
for i,p in enumerate(d.PROFILES):
 c=d.with_profile(d.spatial_case(520000000+i,f'command-{p}'),p);c['start']=[0.,0.,1.];c['goal']=[0.,0.,1.5];d.episode(c,True)
(out/'targeted-memory.json').write_text(json.dumps({'pid':os.getpid(),'private_kib':sum(int(x.split()[1]) for x in Path('/proc/self/smaps_rollup').read_text().splitlines() if x.startswith(('Private_Clean:','Private_Dirty:')))}))
