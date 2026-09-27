from pathlib import Path
import numpy as np,scipy.sparse as sp,torch
from flydrone.tellosim.training.runtime import ReservoirAgent
from flydrone.tellosim.training.checkpoint import configure_exact_execution
from flydrone.tellosim.visual import atomic_json
ROOT=Path('.').resolve();torch.set_num_threads(4);configure_exact_execution('cuda')
graph=ROOT/'data/male-v1.npz'
a=ReservoirAgent(graph,'cuda',profile='balanced_rate_v3',physics_profile='rigid_body_thrust_v2',neural_backend='csr_fp64_accum')
with np.load(graph) as z:w=sp.csr_matrix((z['data'].astype(np.float64),z['indices'],z['indptr']),shape=(int(z['n']),int(z['n'])))
x=np.random.default_rng(414).integers(0,2,(w.shape[0],4)).astype(np.float64);reference=w@x
actual=torch.sparse.mm(a.brain.w,torch.tensor(x,device='cuda')).cpu().numpy()
error=float(np.max(np.abs(reference-actual)));same=np.array_equal(reference.astype(np.float32),actual.astype(np.float32))
report={'graph_sha256':a.brain.graph_sha256,'neurons':w.shape[0],'edges':w.nnz,'canonical_csr':w.has_canonical_format,
    'comparison':'SciPy float64 CSR vs CUDA float64 CSR on identical original edge weights and binary spike inputs',
    'max_absolute_error':error,'float32_current_exact':bool(same),'passed':bool(error<1e-10 and same)}
atomic_json(ROOT/'reports/ts1_rigid_v2/cuda-sparse-check.json',report);print(report);assert report['passed']
