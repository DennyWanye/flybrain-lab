"""Simulation-only UDP fixture, strictly literal 127.0.0.1 at both ends.

One client owns a run. This endpoint never forwards packets to an aircraft.
"""
import argparse
import select
import socket
import time
from pathlib import Path
from ..visual import VisualSession
from .channels import CommandChannel,FaultProfile
from .state_codec import serialize_state

class LoopbackGateway:
    def __init__(self,root,host='127.0.0.1',port=0,state_port=None,profile=FaultProfile(),seed=0):
        if host!='127.0.0.1':raise ValueError('only literal 127.0.0.1 is supported')
        if type(port) is not int or not 0<=port<=65535:raise ValueError('invalid port')
        if state_port is not None and (type(state_port) is not int or not 1<=state_port<=65535):raise ValueError('invalid state port')
        self.socket=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        try:self.socket.bind((host,port))
        except BaseException:self.socket.close();raise
        self.socket.setblocking(False);self.address=self.socket.getsockname();self.state_port=state_port
        self.session=VisualSession(root,mode='sdk-loopback',recording_enabled=False)
        self.channel=CommandChannel(self.session,profile,seed);self.client=None;self.pending=None
        self.counter=0;self.closed=False;self.last_contact=0;self.last_state=0
    def receive(self):
        try:data,address=self.socket.recvfrom(2048)
        except BlockingIOError:return
        if address[0]!='127.0.0.1':return
        if self.client is not None and address!=self.client:return
        if self.pending is not None:return  # Never mix a second error with a pending first response.
        try:
            wire=data.decode('ascii')
            op=self.channel.submit(wire,str(self.counter));self.counter+=1
        except (ValueError,UnicodeError):
            self.socket.sendto(b'error',address);return
        self.client=address;self.pending=op['operation_id'];self.last_contact=self.session.tick
    def tick(self):
        self.receive();self.channel.advance(1)
        if self.pending is not None:
            op=self.channel.poll(self.pending)
            if op['client'] in ('ack_ok','ack_error'):
                self.socket.sendto(op['response'].encode('ascii'),self.client)
                self.pending=None;self.last_contact=self.session.tick
            elif op['client']=='unknown_execution':self.pending=None  # Drop response, do not invent a wire ID.
        if self.state_port and self.client and self.session.tick-self.last_state>=12:
            self.socket.sendto(serialize_state(self.session).encode('ascii'),('127.0.0.1',self.state_port));self.last_state=self.session.tick
        # Watchdog only after an idle interval; long in-flight commands are not timed
        # out merely because the protocol is serial. This is an engineering profile.
        if self.session.airborne and self.pending is None and self.session.tick-self.last_contact>=1800:
            try:self.session.command('land',f'watchdog-{self.session.tick}')
            except ValueError:pass
            self.last_contact=self.session.tick
    def serve(self,stop_event):
        deadline=time.monotonic()
        try:
            while not stop_event.is_set() and not self.session.finished:
                self.tick();deadline+=1/120
                stop_event.wait(max(0,deadline-time.monotonic()))
        finally:self.close()
    def close(self):
        if self.closed:return
        self.closed=True
        try:self.session.close('loopback_gateway_closed')
        finally:self.socket.close()

def main():
    import threading
    p=argparse.ArgumentParser();p.add_argument('--project-root',type=Path,default=Path('.'))
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=18889)
    p.add_argument('--state-port',type=int);a=p.parse_args()
    gateway=LoopbackGateway(a.project_root,a.host,a.port,a.state_port)
    print(f'SIMULATION ONLY: {gateway.address}',flush=True)
    try:gateway.serve(threading.Event())
    except KeyboardInterrupt:gateway.close()
if __name__=='__main__':main()
