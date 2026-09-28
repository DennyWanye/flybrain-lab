import subprocess,sys,json
from pathlib import Path
from flydrone.tellosim.training.spatial_orientation_campaign import evaluate,baseline
from flydrone.tellosim.training.spatial_orientation import spatial_case
from flydrone.tellosim.training.altitude_refined import with_profile,PROFILES
root=Path.cwd();out=root/'reports/ts1_spatial_orientation'
subprocess.run([sys.executable,'-m','flydrone.tellosim.training.spatial_orientation_campaign','smoke'],check=True)
cases=[with_profile(spatial_case(440000000+i),tuple(PROFILES)[i%4]) for i in range(8)]
baseline(root,cases,'rule',out/'development-rule.json')
evaluate(root,root/'runs/tellosim-sdk9/spatial-orientation-smoke-s11/checkpoint.pt',cases,out/'smoke-evaluation.json','spatial-orientation-dev',batch=8)
subprocess.run([sys.executable,'-m','reports.ts1_spatial_orientation.run_campaign'],check=True)
