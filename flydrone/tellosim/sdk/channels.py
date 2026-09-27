"""Bounded deterministic request/reply queues for the shared simulator."""
from dataclasses import dataclass
import numpy as np
from .codec import decode_command

@dataclass(frozen=True)
class FaultProfile:
    request_drop:float=0.
    reply_drop:float=0.
    request_delay_ticks:int=0
    reply_delay_ticks:int=0
    def __post_init__(self):
        for v in (self.request_drop,self.reply_drop):
            if not isinstance(v,(int,float)) or not np.isfinite(v) or not 0<=v<=1:raise ValueError('invalid drop probability')
        for v in (self.request_delay_ticks,self.reply_delay_ticks):
            if type(v) is not int or not 0<=v<=1200:raise ValueError('invalid delay ticks')

class CommandChannel:
    def __init__(self,session,profile=FaultProfile(),seed=0):
        self.session=session;self.profile=profile;self.rng=np.random.default_rng(seed)
        self.operations={};self.active=None
    def submit(self,wire,request_id):
        if not isinstance(request_id,str) or not 1<=len(request_id)<=100:raise ValueError('request_id must be 1..100 characters')
        cmd=decode_command(wire)
        if request_id in self.operations:
            old=self.operations[request_id]
            if old['wire']!=wire:raise ValueError('duplicate request id with different command')
            return self.poll(request_id)
        if self.active and self.operations[self.active]['client'] in ('sent','unknown_execution'):
            if cmd.verb!='stop':raise ValueError('busy or unknown operation')
            old=self.operations[self.active]
            if old['client']=='sent':old['client']='cancelled_local'
        if len(self.operations)>=256:raise ValueError('channel episode command limit')
        # Conservative client timeout is chosen before execution, without consulting
        # completion truth. No automatic retry of relative motion is possible.
        distance=max(abs(v) for v in cmd.args[:3])/100 if cmd.args and cmd.verb=='go' else (cmd.args[0]/100 if cmd.args and cmd.verb in ('forward','back','left','right','up','down') else 0)
        timeout=60+distance/max(.05,self.session.world.speed_limit)*2
        op={'operation_id':request_id,'request_id':request_id,'wire':wire,'client':'sent','response':None,
            'sent_tick':self.session.tick,'verb':cmd.verb,'due':self.session.tick+self.profile.request_delay_ticks,'deadline':self.session.tick+int(timeout*120),
            'drop_request':self.rng.random()<self.profile.request_drop,'drop_reply':self.rng.random()<self.profile.reply_drop,
            'device_id':None,'reply_due':None,'device_response':None}
        self.operations[request_id]=op;self.active=request_id;self.pump();return self.poll(request_id)
    def pump(self):
        for op in self.operations.values():
            if op['client']!='sent':continue
            tick=self.session.tick
            if op['device_id'] is None and not op['drop_request'] and tick>=op['due']:
                try:
                    result=self.session.command(op['wire'],'channel-'+op['request_id'])
                    op['device_id']=result['request_id']
                except ValueError:
                    op['device_id']='rejected';op['device_response']='error';op['reply_due']=tick+self.profile.reply_delay_ticks
            if op['device_id'] not in (None,'rejected') and op['reply_due'] is None:
                actual=self.session.requests[op['device_id']]
                if actual['client'] in ('ack_ok','ack_error','cancelled_local'):
                    op['device_response']=actual['response'] or 'error'
                    op['reply_due']=tick+self.profile.reply_delay_ticks
            if op['reply_due'] is not None and tick>=op['reply_due'] and not op['drop_reply']:
                op['response']=op['device_response'];op['client']='ack_error' if op['response']=='error' else 'ack_ok'
            elif tick>=op['deadline']:op['client']='unknown_execution'
    def advance(self,ticks=12):
        for _ in range(ticks):
            self.pump();self.session.advance(1);self.pump()
    def poll(self,operation_id):
        op=self.operations[operation_id]
        return {key:op[key] for key in ('operation_id','request_id','wire','client','response')}
