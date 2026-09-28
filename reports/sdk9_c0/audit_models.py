from pathlib import Path
import json,hashlib
import numpy as np
import torch
from flydrone.tellosim.visual import atomic_json
from flydrone.tellosim.training.contracts import digest
ROOT=Path('.').resolve();OUT=ROOT/'reports/sdk9_c0'

def main():
    cases=json.loads((OUT/'cases.json').read_text());held={c['seed'] for name in ('validation','sealed_test') for c in cases[name]}
    results=[];contracts=[];hashes=[]
    for seed in (11,22,33):
        path=ROOT/f'runs/tellosim-sdk9/c0-s{seed}-20260927'
        if not (path/'training.json').exists():continue
        initial=torch.load(path/'initial.pt',map_location='cpu',weights_only=False)
        final=torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=False)
        assert final['extra']['seed']==seed and final['extra']['curriculum']=='C0'
        assert final['contract']==initial['contract'];contracts.append(final['contract'])
        assert final['contract']['graph_sha256']=='badc33a247894fe12c4d68d3ff791393857cffa39ff2ab81869608a11fa1e0c9'
        protocol=json.loads((path/'protocol.json').read_text())
        provenance=json.loads((path/'data-provenance.json').read_text())
        train={row['case_seed'] for row in provenance}|set(protocol['critic_seeds'])
        # The PPO option budget bounds the maximum possible number of episodes.
        train |= set(range(3600000+seed*1000,3600000+seed*1000+protocol['ppo_options']))
        assert not train&held
        data=np.load(path/'demonstrations.npz')
        assert len(provenance)==len(data['teacher_actions'])
        assert np.isfinite(data['features']).all() and np.all(data['masks'][np.arange(len(data['teacher_actions'])),data['teacher_actions']])
        delta=float(torch.sqrt(sum((final['policy'][k]-v).square().sum() for k,v in initial['policy'].items())))
        assert delta>0 and final['value_trained']
        sha=hashlib.sha256((path/'checkpoint.pt').read_bytes()).hexdigest();hashes.append(sha)
        training=json.loads((path/'training.json').read_text())
        assert training['checkpoint_sha256']==sha
        checked_results=[]
        for label in ('validation','sealed','near-regression','near-sampled','faults'):
            result_path=path/(label+'.json')
            if result_path.exists():
                result=json.loads(result_path.read_text())
                assert result['checkpoint_sha256']==sha,(seed,label,'checkpoint differs')
                assert result['episodes']==len(result['results'])
                assert result['successes']==sum(row['success'] for row in result['results'])
                if label in ('validation','sealed'):
                    split='validation' if label=='validation' else 'sealed_test'
                    assert result['case_hash']==digest(cases[split])
                    assert [r['case_id'] for r in result['results']]==[c['case_id'] for c in cases[split]]
                checked_results.append(label)
        results.append({'seed':seed,'checkpoint_sha256':sha,'checked_result_sources':checked_results,'parameter_delta_l2':delta,
            'graph_unchanged':True,'contract_unchanged':True,'dataset_disjoint':True,
            'training_examples':len(provenance),'value_trained':final['value_trained'],
            'source_hashes_recorded':'source_hashes' in protocol})
    lockfile=OUT/'frozen-checkpoints.json'
    if lockfile.exists():
        lock=json.loads(lockfile.read_text())
        assert lock['case_hash']==digest(cases['sealed_test'])
        assert [hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in lock['checkpoints']]==lock['checkpoint_hashes']
        random_file=OUT/'random-sealed.json'
        if random_file.exists():assert json.loads(random_file.read_text())['checkpoint_sha256']==lock['checkpoint_hashes'][-1]
    assert all(c==contracts[0] for c in contracts)
    assert len(hashes)==len(set(hashes))
    result={'models':results,'complete_three_seeds':len(results)==3,'same_contract':True,
        'validation_hash':digest(cases['validation']),'sealed_hash':digest(cases['sealed_test']),
        'note':'Seed 11 began before additional metadata/input-validation changes; nominal observation, model, physics and optimizer math unchanged.'}
    atomic_json(OUT/'artifact-audit.json',result);print(json.dumps(result))
if __name__=='__main__':main()
