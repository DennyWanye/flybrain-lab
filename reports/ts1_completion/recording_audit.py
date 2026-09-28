"""Actual full256MiB storage pressure plus actual long rigid-body command."""
from pathlib import Path
import json,hashlib,time,resource
from flydrone.tellosim.recording_v3 import RecordingV3,AuditedVisualSession
from flydrone.tellosim.visual import scene_config
root=Path.cwd();out=root/'reports/ts1_completion';folder=out/'quota256';folder.mkdir(exist_ok=False)
quota=256*1024*1024;started=time.monotonic()
r=RecordingV3(folder,{'run_id':'storage-pressure256','epoch':'fixture','episode_id':'storage-only','policy_source':'storage_pressure_fixture','neural_activity':None},quota=quota)
payload={'storage_pressure_fixture':'x'*16384,'neural_activity':None}
for tick in range(24000):r.add('fixture',tick,payload)
r.close('storage_pressure_complete');m=r.manifest
assert m['partial'] and m['complete'] and not r.buffers and .97*quota<m['bytes']<=quota
size=0;last=-1;count=0
for index,c in enumerate(m['streams']['fixture']):
 body=(folder/c['file']).read_bytes();assert len(body)==c['bytes'] and hashlib.sha256(body).hexdigest()==c['sha256']
 rows=[json.loads(x) for x in body.splitlines()];assert len(rows)==c['count']
 assert rows[0]['sim_tick']==c['first_tick']==last+1 and rows[-1]['sim_tick']==c['last_tick']
 last=c['last_tick'];size+=len(body);count+=len(rows)
assert size==m['bytes'] and m['missing_intervals'][0]['first_tick']==last+1 and m['missing_intervals'][0]['last_tick']==23999
scene=scene_config(root,False);scene['controller']='rigid_body_thrust_v2'
s=AuditedVisualSession(root,mode='manual',scene=scene,run_id='ts1-long-command-v3',seed=340000000)
def command(wire):
 op=s.command(wire,f'long-{len(s.requests)}');begin=s.tick
 for i in range(1500):
  s.advance(12)
  if s.operation['client']!='sent':break
 assert s.operation['client']=='ack_ok',(wire,s.operation)
 return s.tick-begin
try:
 command('command');command('takeoff');command('speed 10');duration=command('forward 100')
 assert duration>1200
 s.close('long_command_verified');lm=s.recording.manifest
 assert not lm['partial'] and len(lm['streams']['trajectory'])>=6
 for stream,chunks in lm['streams'].items():
  previous=-1
  for c in chunks:
   body=(s.recording.directory/c['file']).read_bytes();assert hashlib.sha256(body).hexdigest()==c['sha256'] and c['first_tick']>=previous
   previous=c['last_tick']
 assert not s.recording.buffers
finally:
 if not s.finished:s.close('interrupted')
result={'status':'PASS','quota_bytes':quota,'committed_bytes':size,'chunk_count':len(m['streams']['fixture']),'valid_prefix_rows':count,'missing_intervals':m['missing_intervals'],'long_command_ticks':duration,'long_replay':s.run_id,'long_chunks':len(lm['streams']['trajectory']),'buffers_released':True,'elapsed_s':time.monotonic()-started,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'scope':'quota fixture is storage only, no simulated neural activity; long command uses actual rigid physics'}
(out/'T55-full.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
