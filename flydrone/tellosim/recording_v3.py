"""Versioned quota closure: explicit missing intervals and bounded buffer release.

Frozen training Recording/VisualSession sources remain unchanged. New interactive
sessions use this class; existing view/2.0 replay remains readable.
"""
from .visual import Recording,VisualSession,packed
import hashlib

class RecordingV3(Recording):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.manifest.update(recording_contract='tellosim.recording/3',missing_intervals=[])
        self.save()
    def missing(self,kind,first,last):
        entry=next((x for x in self.manifest['missing_intervals'] if x['stream']==kind),None)
        if entry is None:self.manifest['missing_intervals'].append({'stream':kind,'first_tick':int(first),'last_tick':int(last),'reason':'recording_quota_exceeded'})
        else:entry['first_tick']=min(entry['first_tick'],int(first));entry['last_tick']=max(entry['last_tick'],int(last))
    def add(self,kind,tick,data):
        if self.manifest['partial']:
            self.missing(kind,tick,tick);return
        super().add(kind,tick,data)
    def flush(self,kind=None):
        if self.manifest['partial']:
            for stream,rows in self.buffers.items():
                if rows:self.missing(stream,rows[0]['sim_tick'],rows[-1]['sim_tick'])
            self.buffers.clear();self.save();return
        for key in ([kind] if kind else list(self.buffers)):
            rows=self.buffers.get(key,[])
            if not rows:continue
            body=b'\n'.join(packed(row) for row in rows)+b'\n'
            if self.manifest['bytes']+len(body)>self.quota:
                self.manifest.update(partial=True,partial_reason='recording_quota_exceeded')
                self.flush();return
            self.buffers[key]=[];chunks=self.manifest['streams'].setdefault(key,[]);name=f'{key}-{len(chunks):05d}.jsonl'
            (self.directory/name).write_bytes(body)
            chunks.append({'file':name,'first_tick':rows[0]['sim_tick'],'last_tick':rows[-1]['sim_tick'],'count':len(rows),'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()})
            self.manifest['bytes']+=len(body)
        self.save()
    def close(self,outcome='stopped'):
        self.flush();self.buffers.clear();self.manifest.update(complete=True,outcome=outcome);self.save()

class AuditedVisualSession(VisualSession):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        old=self.recording
        if isinstance(old,Recording):
            if old.manifest['bytes'] or old.manifest['streams']:raise RuntimeError('recorder migration is only allowed before first flush')
            self.recording=RecordingV3(old.directory,old.manifest,quota=old.quota)
            self.recording.buffers=old.buffers
