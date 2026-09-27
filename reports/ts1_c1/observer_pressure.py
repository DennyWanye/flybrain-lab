"""Five bounded HTTP readers, including slow/disconnected clients, during training."""
from pathlib import Path
import concurrent.futures,json,time,urllib.request,statistics
ROOT=Path('.').resolve();BASE='http://127.0.0.1:8765/api/tellosim/'
def get(path):
    with urllib.request.urlopen(BASE+path,timeout=10) as r:return json.load(r)
rows=[r for r in get('runs')['runs'] if r.get('training_group')=='c1-s11-learn' and r['active']]
assert len(rows)==4
started=time.monotonic()
def reader(index):
    times=[];identities=[];maximum=0;count=0;last={}
    while time.monotonic()-started<120:
        r=rows[count%4];t=time.monotonic();f=get('runs/'+r['run_id']+'/latest');times.append(time.monotonic()-t)
        assert f['run_id']==r['run_id'] and f['env_id']==r['env_id'] and f['epoch']==r['epoch']
        assert f['seq']>=last.get(r['run_id'],-1);last[r['run_id']]=f['seq']
        identities.append((f['env_id'],f['episode_id']));maximum=max(maximum,len(json.dumps(f).encode()));count+=1
        time.sleep(.7 if index==4 else (.05 if index==3 else .1))
    return {'reader':index,'requests':count,'max_frame_bytes':maximum,'p95_latency_s':float(__import__('numpy').percentile(times,95)),
            'distinct_episodes':len(set(identities)),'last_sequences':last}
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:results=list(pool.map(reader,range(5)))
report={'passed':all(r['distinct_episodes']>=4 and r['max_frame_bytes']<=65536 for r in results),'duration_s':time.monotonic()-started,'readers':results,
    'scope':'five concurrent real HTTP consumers, one slow; new connections each request; not five browser-renderer FPS variants'}
(ROOT/'reports/ts1_c1/observer-pressure.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
assert report['passed']
