"""Supplementary smoke readout fit on recorded real neural features; no formal model selection."""
from pathlib import Path
import json,copy,torch,numpy as np,subprocess,sys
from flydrone.tellosim.training.spatial_precision import SpatialPool,SKILLS,load_spatial,save_spatial,spatial_case,sha
from flydrone.tellosim.training.spatial_precision_campaign import evaluate
from flydrone.tellosim.training.altitude_refined import with_profile,PROFILES
from flydrone.tellosim.training.stability_campaign import fit
from flydrone.tellosim.training.checkpoint import configure_exact_execution
from flydrone.tellosim.training.runtime import distribution
root=Path.cwd();report=root/'reports/ts1_spatial_precision';source=root/'runs/tellosim-sdk9/spatial-precision-smoke-s11';out=root/'runs/tellosim-sdk9/spatial-precision-smoke-fit-s11';out.mkdir(exist_ok=False)
torch.set_num_threads(4);configure_exact_execution('cuda');pool=SpatialPool(root,11,batch=8);load_spatial(source/'checkpoint.pt',pool,root)
data=np.load(source/'navigation/demonstrations.npz');rows=[(x,m,int(y),float(w)) for x,m,y,w in zip(data['features'],data['masks'],data['labels'],data['weights'])]
model=pool.models['navigation'];opt=torch.optim.AdamW(list(model.body.parameters())+list(model.actor.parameters()),lr=.0003,weight_decay=.001);torch.manual_seed(370001011)
metrics=fit(model,opt,rows,80)
expected={k:{n:v.clone() for n,v in m.state_dict().items()} for k,m in pool.models.items()}
save_spatial(out/'checkpoint.pt',pool,{'navigation':opt},root,640,{'smoke_only':True,'supplementary_epochs':80,'source_features':str(source.relative_to(root)),'method':'fresh navigation needs actual readout fit; original2-epoch smoke and0/8 retained'})
load_spatial(out/'checkpoint.pt',pool,root)
assert all(torch.equal(expected[k][n],v) for k,m in pool.models.items() for n,v in m.state_dict().items())
t=json.loads((source/'training.json').read_text());initial=torch.load(source/'initial.pt',map_location='cpu',weights_only=False)
t.update(checkpoint_sha256=sha(out/'checkpoint.pt'),supplementary_navigation_epochs=80,source_training=str(source.relative_to(root)),supplementary_metrics=metrics)
t['skills']['navigation']['parameter_delta_l2']=float(torch.sqrt(sum((initial['models']['navigation'][k]-v).square().sum() for k,v in pool.models['navigation'].state_dict().items())))
(out/'training.json').write_text(json.dumps(t,indent=2));print(metrics,flush=True)
del pool
cases=[with_profile(spatial_case(370000008+i),tuple(PROFILES)[i%4]) for i in range(8)]
result=evaluate(root,out/'checkpoint.pt',cases,report/'smoke-fit-evaluation.json','spatial-precision-smoke-fit',batch=8)
assert result['successes']>0
subprocess.run([sys.executable,'-m','reports.ts1_spatial_precision.run_campaign'],check=True)
