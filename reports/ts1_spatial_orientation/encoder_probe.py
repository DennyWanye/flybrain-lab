from pathlib import Path
import json,numpy as np,torch
from flydrone.tellosim.training.parallel import ReservoirPool
from flydrone.tellosim.training.contracts import Encoder34
from flydrone.tellosim.training.spatial_orientation import HeadingEncoder34
from flydrone.tellosim.training.checkpoint import configure_exact_execution
root=Path.cwd();torch.set_num_threads(4);configure_exact_execution('cuda')
pool=ReservoirPool(root/'data/male-v1.npz','cuda',seed=11,profile='balanced_rate_v3',batch=6,physics_profile='rigid_body_thrust_v2',neural_backend='csr_fp64_accum')
x=np.zeros((3,26),np.float32);x[:,12:15]=1;x[:,7]=1;x[:,8]=1/3;x[:,9]=.8
angles=np.deg2rad([-8,-21,-8]);x[:,6]=np.sin(angles);x[:,7]=np.cos(angles);x[2,:3]=[.01,-.02,-.075/3];x[2,8]=1.6/3
encoded=np.concatenate([Encoder34('balanced_rate_v2')(x),HeadingEncoder34()(x)],axis=0)
for _ in range(20):pool.brain.advance(encoded,np.ones(6,bool))
f=pool.brain.current_features()
r={'scope':'controlled representation fixture, full MaleCNS, not a model task result','graph_sha256':pool.brain.graph_sha256,'old_near_boundary_feature_l2':float(np.linalg.norm(f[0]-f[1])),'new_near_boundary_feature_l2':float(np.linalg.norm(f[3]-f[4])),'old_nuisance_feature_l2':float(np.linalg.norm(f[0]-f[2])),'new_nuisance_feature_l2':float(np.linalg.norm(f[3]-f[5]))}
assert r['new_near_boundary_feature_l2']>0 and r['new_nuisance_feature_l2']==0
(root/'reports/ts1_spatial_orientation/ENCODER_PROBE.json').write_text(json.dumps(r,indent=2));print(r)
