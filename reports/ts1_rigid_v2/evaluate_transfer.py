"""Freeze legacy readout weights under an explicit NEW physics contract."""
from pathlib import Path
import argparse,hashlib,json,time
import torch
from flydrone.tellosim.training.runtime import ReservoirAgent,save_checkpoint,load_checkpoint
from flydrone.tellosim.training.continuation import initialize_from
from flydrone.tellosim.training.c0_campaign import evaluate
from flydrone.tellosim.physics.rigid import PROFILE
from flydrone.tellosim.visual import atomic_json
ROOT=Path('.').resolve();OUT=ROOT/'reports/ts1_rigid_v2'
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);parser.add_argument('--split',choices=['validation','sealed_test'],default='validation');a=parser.parse_args()
    assert json.loads((OUT/'preflight.json').read_text())['passed']
    torch.set_num_threads(4)
    agent=ReservoirAgent(ROOT/'data/male-v1.npz','cuda',seed=a.seed,profile='balanced_rate_v3',physics_profile=PROFILE)
    target=OUT/f'transferred-s{a.seed}.pt'
    if not target.exists():
        assert a.split=='validation'
        lineage=initialize_from(ROOT/f'runs/tellosim-sdk9/c0-s{a.seed}-20260927/checkpoint.pt',agent,ROOT)
        save_checkpoint(target,agent,None,ROOT,0,0,{'method':'Legacy C0 readout transferred unchanged to six-DOF physics; NO new training','seed':a.seed,'lineage':lineage})
    load_checkpoint(target,agent,ROOT)
    cases=json.loads((OUT/'cases.json').read_text())[a.split]
    if a.split=='sealed_test':
        lock=json.loads((OUT/'frozen-checkpoints.json').read_text())
        assert hashlib.sha256(target.read_bytes()).hexdigest() in lock['checkpoint_hashes']
        from flydrone.tellosim.training.contracts import digest
        assert digest(cases)==lock['case_hash']
    evaluate(ROOT,agent,cases,OUT/f's{a.seed}-{a.split}.json',f'rigid-s{a.seed}-{a.split}','argmax',record_indices=(0,1) if a.split=='validation' else (0,))
if __name__=='__main__':main()
