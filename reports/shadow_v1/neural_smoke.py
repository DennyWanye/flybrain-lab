"""Small full-connectome integration check, not a flight or success-rate eval."""
import sys, os, json, time, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from flydrone.shadow.inputs import Profile
from flydrone.shadow.__main__ import fixture
from flydrone.shadow.recording import Recorder, samples, verify
from flydrone.shadow.runtime import C2WPolicy, ShadowSession
OUT=ROOT/'reports/shadow_v1/fullgraph-smoke'
OUT.mkdir(exist_ok=False)
(OUT/'process.json').write_text(json.dumps({'pid':os.getpid(),'cwd':str(Path.cwd()),'argv':sys.argv,'start_time':time.time()}))
network=[]
def audit(event,args):
    if event in ('socket.connect','socket.sendto'):
        network.append(event)
        raise RuntimeError('network forbidden during shadow probe')
sys.addaudithook(audit)
profile=Profile('analytic-interface-fixture-v1','synthetic_fixture','fixture_z_up')
with Recorder(OUT/'recording',profile) as recording:
    for row in fixture():recording.append(row)
verified=verify(OUT/'recording')
results=[]
for seed in (11,22,33):
    policy=None
    try:
        policy=C2WPolicy(ROOT,seed)
        graph={'neurons':policy.pool.brain.n,'edges':policy.pool.brain.w._nnz(),**policy.metadata}
        hashes=[]; passes=[]
        for repeat in range(2):
            if repeat:
                policy.pool.reset();policy.lane.reset();policy.phase=None
            target=OUT/f's{seed}-repeat{repeat}'
            with ShadowSession(profile,policy,target) as session:
                rows=[session.step(packet) for packet in samples(OUT/'recording')]
            hashes.append(hashlib.sha256((target/'decisions.jsonl').read_bytes()).hexdigest())
            assert policy.brain_tick==96
            assert all(r['transmitted'] is False for r in rows)
            assert all(r['policy']['input_source']=='reservoir_v_trace' for r in rows)
            assert len({r['features_sha256'] for r in rows})>3
            passes.append({'samples':len(rows),'brain_substeps':policy.brain_tick,'actions':[r['proposed_action'] for r in rows]})
        assert hashes[0]==hashes[1]
        results.append({'seed':seed,'graph':graph,'repeat_exact':True,'sha256':hashes[0],'passes':passes})
        print(json.dumps({'seed':seed,'repeat_exact':True,'samples':48}),flush=True)
    finally:
        if policy:policy.close()
private=sum(int(line.split()[1]) for line in Path('/proc/self/smaps_rollup').read_text().splitlines() if line.startswith(('Private_Clean:','Private_Dirty:')))*1024
report={'scope':'interface and full neural replay only; synthetic fixtures; no task-success claim','runs':results,'network_attempts':network,'transmitted_commands':0,'REAL_FLIGHT_READY':False,'sampled_private_bytes':private}
(OUT/'summary.json').write_text(json.dumps(report,indent=2))
print('FULLGRAPH_SHADOW_SMOKE_PASS',flush=True)
