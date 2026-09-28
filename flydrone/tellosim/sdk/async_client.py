"""Async waiting over the existing simulated command channel.

Cancellation or timeout detaches one waiter. Only an explicit SDK stop command
can request device stopping; this adapter never advances simulation time.
"""
from dataclasses import dataclass
import asyncio

@dataclass(frozen=True)
class PendingOperation:
    operation_id: str
    request_id: str
    completion: asyncio.Task

class AsyncCommandClient:
    def __init__(self,channel,poll_interval_s=.01):
        if poll_interval_s<=0:raise ValueError('poll interval must be positive')
        self.channel=channel;self.poll_interval_s=poll_interval_s

    def submit(self,wire,request_id):
        loop=asyncio.get_running_loop()
        snapshot=self.channel.submit(wire,request_id)
        operation_id=snapshot['operation_id']
        return PendingOperation(operation_id,request_id,loop.create_task(self.wait(operation_id,request_id)))

    async def wait(self,operation_id,request_id):
        while True:
            snapshot=self.channel.poll(operation_id)
            if snapshot['operation_id']!=operation_id or snapshot['request_id']!=request_id:
                raise RuntimeError('operation receipt ownership mismatch')
            if snapshot['client']!='sent':
                return {**snapshot,'device_stop_confirmed':snapshot['wire']=='stop' and snapshot['client']=='ack_ok',
                    'wait_status':'completed','execution_status':snapshot['client']}
            # Cancelling this await leaves channel ownership and execution intact.
            await asyncio.sleep(self.poll_interval_s)
