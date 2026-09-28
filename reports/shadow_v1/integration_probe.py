import sys,os,time,json,subprocess,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from dataclasses import asdict
from flydrone.shadow.inputs import Profile
from flydrone.shadow.__main__ import fixture
from flydrone.shadow.recording import verify,verify_shadow
OUT=ROOT/'reports/shadow_v1/cli-probe';OUT.mkdir(exist_ok=False)
profile=Profile('fault-interface-fixture-v1','synthetic_fixture','fixture_z_up')
(OUT/'profile.json').write_text(json.dumps(asdict(profile)))
rows=list(fixture())
rows[0]['pose']['valid']=False
rows[1]['pose']['captured_ns']=rows[0]['pose']['captured_ns']
rows[1]['time_ns'] # .1 age alone is valid; use state capture for stale fault instead
rows[1]['state']['captured_ns']=0
rows[2]['state']['valid']=False
rows[3]['state']['airborne']=False
rows[4]['context']['operation_status']='busy'
rows[5]['context']['operation_status']='unknown'
rows[6]['context']['remaining_s']=0
rows[7]['pose'].update(valid=False,position_m=None,velocity_mps=None,yaw_rad=None)
# state capture monotonicity: first packet must start no later than the stale second.
rows[0]['state']['captured_ns']=0
(OUT/'input.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
processes=[]
def run(name,args,expected=0):
    cmd=[sys.executable,'-m','flydrone.shadow',*args]
    with (OUT/f'{name}.stdout.log').open('x') as stdout,(OUT/f'{name}.stderr.log').open('x') as stderr:
        process=subprocess.Popen(cmd,cwd=ROOT,stdout=stdout,stderr=stderr)
        entry={'pid':process.pid,'cmd':cmd,'cwd':str(ROOT)};processes.append(entry)
        (OUT/'processes.json').write_text(json.dumps(processes,indent=2))
        try:code=process.wait(timeout=60)
        except BaseException:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            raise
    entry['returncode']=code;entry['exited']=not Path(f'/proc/{process.pid}').exists()
    assert code==expected,(name,code,(OUT/f'{name}.stderr.log').read_text())
    (OUT/'processes.json').write_text(json.dumps(processes,indent=2))
run('record',['record','--input',str(OUT/'input.jsonl'),'--profile',str(OUT/'profile.json'),'--output',str(OUT/'recording')])
run('replay',['replay','--recording',str(OUT/'recording'),'--root',str(ROOT),'--output',str(OUT/'replay')])
run('shadow',['shadow','--input',str(OUT/'input.jsonl'),'--profile',str(OUT/'profile.json'),'--root',str(ROOT),'--output',str(OUT/'direct')])
a=verify_shadow(OUT/'replay/shadow');b=verify_shadow(OUT/'direct/shadow')
assert a==b and a['samples']==24 and a['blocked_samples']==8 and a['brain_substeps']==96
for root in ('replay','direct'):verify(OUT/root/'input')
assert (OUT/'replay/shadow/decisions.jsonl').read_bytes()==(OUT/'direct/shadow/decisions.jsonl').read_bytes()
# Malformed duplicate input must fail and retain the partial record and decisions.
(OUT/'duplicate.jsonl').write_text(json.dumps(rows[0])+'\n'+json.dumps(rows[0])+'\n')
run('duplicate',['shadow','--input',str(OUT/'duplicate.jsonl'),'--profile',str(OUT/'profile.json'),'--root',str(ROOT),'--output',str(OUT/'duplicate')],expected=1)
assert not json.loads((OUT/'duplicate/shadow/result.json').read_text())['completed']
assert not json.loads((OUT/'duplicate/input/footer.json').read_text())['completed']
for seed in (11,22,33):
    for repeat in (0,1):verify_shadow(ROOT/f'reports/shadow_v1/fullgraph-smoke/s{seed}-repeat{repeat}')
summary={'samples_per_valid_run':24,'blocked_samples_per_run':8,'direct_replay_byte_identical':True,'duplicate_rejected':True,'all_children_exited':all(p['exited'] for p in processes),'transmitted_commands':0,'REAL_FLIGHT_READY':False}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
