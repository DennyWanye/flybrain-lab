"""Small real-MaleCNS encoding probe, independent of held-out scenes."""
from pathlib import Path
import json
import numpy as np
from flydrone.tellosim.training.altitude_refined_campaign import pool_for
from flydrone.tellosim.training.altitude_refined import AltitudeEncoder34
from flydrone.tellosim.training.contracts import Encoder34
from flydrone.tellosim.visual import atomic_json
root=Path.cwd();out=root/'reports/ts1_altitude_refined/development-probe.json'
if out.exists():raise FileExistsError(out)
errors=np.array([-.125,-.105,-.095,-.075,.075,.095,.105,.125],np.float32)
pool=pool_for(root,11,8);rows={}
for name,encoder in [('V1',Encoder34('balanced_rate_v2')),('V1R',AltitudeEncoder34())]:
    pool.brain.encoder=encoder;pool.brain.reset()
    observations=np.zeros((8,26),np.float32);observations[:,2]=errors/3;observations[:,8]=1/3;observations[:,9]=1;observations[:,12:15]=1;observations[:,16]=1
    samples=[]
    for sample in range(40):
        pool.brain.advance(encoder(observations));samples.append(pool.brain.current_features().copy())
    features=np.stack(samples)
    rows[name]={'adjacent_boundary_feature_l2':[float(np.linalg.norm(features[-20:,a]-features[-20:,b])) for a,b in [(1,2),(5,6)]], 'finite':bool(np.isfinite(features).all()),'brain_advance_samples':40*8}
atomic_json(out,{'diagnostic_only':True,'static_sensor_probe_not_control_success':True,'graph_sha256':pool.brain.graph_sha256,'neurons':pool.brain.n,'errors_m':errors.tolist(),'results':rows})
print(json.dumps(rows))
