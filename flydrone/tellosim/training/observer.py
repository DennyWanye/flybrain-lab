"""Read-only, latest-only observer lanes. No commands or full trajectory recording."""
import copy,os,re,uuid
from pathlib import Path
from ...vis.live import LatestWriter
from ..visual import atomic_json

class TrainingObserver:
    def __init__(self,root,run_id,batch,method):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,55}',run_id):raise ValueError('observer run id must be 1..55 safe characters')
        self.root=Path(root);self.group=run_id;self.batch=batch;self.method=method
        self.attempt=uuid.uuid4().hex[:8];self.writers={};self.descriptors={};self.latest={};self.sequence=[0]*batch
        self.progress={'options':0,'updates':0,'status':'starting'}
    def publish(self,env,frame):
        i=env.env_id
        if i not in self.writers:
            run=f'{self.group}-{self.attempt}-e{i}'
            descriptor=copy.deepcopy(env.session.recording.manifest)
            descriptor.update(schema_version='tellosim.observer/1.0',run_id=run,epoch=self.attempt,source_epoch=self.attempt,
                training_group=self.group,env_id=i,env_count=self.batch,observer_only=True,complete=False,streams={},partial=False,
                policy_source='training_readonly_malecns',training_method=self.method,owner_pid=os.getpid(),
                limitations=descriptor.get('limitations',[])+['Latest-only training observation; not a full replay'])
            directory=self.root/'reports/vis/tellosim'/run
            if directory.exists():raise FileExistsError(directory)
            directory.mkdir(parents=True);atomic_json(directory/'manifest.json',descriptor)
            self.descriptors[i]=descriptor;self.writers[i]=LatestWriter(directory,descriptor)
        self.sequence[i]+=1;d=self.descriptors[i]
        snapshot={**frame,'run_id':d['run_id'],'epoch':self.attempt,'episode_epoch':frame['epoch'],
            'episode_finished':frame['finished'],'finished':False,'seq':self.sequence[i],
            'training_group':self.group,'training_progress':dict(self.progress)}
        self.latest[i]=snapshot;self.writers[i].publish(snapshot)
    def update(self,counters,status='running'):
        self.progress={k:copy.deepcopy(counters[k]) for k in ('options','updates','base_ticks','episodes','elapsed_s') if k in counters}
        self.progress['status']=status
    def close(self,status):
        for i,writer in self.writers.items():
            d=self.descriptors[i];d.update(complete=True,outcome=status,duration_s=self.latest[i]['time_s'])
            atomic_json(writer.directory/'manifest.json',d)
            self.sequence[i]+=1
            writer.publish({**self.latest[i],'seq':self.sequence[i],'finished':True,'training_progress':{**self.progress,'status':status}})
            writer.close()
