import asyncio
import numpy as np
import pytest
from tests.flydrone.test_rigid_training import project,complete
from flydrone.tellosim.visual import VisualSession
from flydrone.tellosim.sdk.channels import CommandChannel,FaultProfile
from flydrone.tellosim.sdk.async_client import AsyncCommandClient

def session(project):
    s=VisualSession(project,mode='training',recording_enabled=False)
    complete(s,'command','sdk');complete(s,'takeoff','up');return s

@pytest.mark.parametrize('cancel_mode',['cancel','timeout'])
def test_executing_future_detaches_without_stop_or_receipt_theft(project,cancel_mode):
    s=session(project);channel=CommandChannel(s,FaultProfile(reply_delay_ticks=24));client=AsyncCommandClient(channel,.0001)
    async def scenario():
        pending=client.submit('forward 100','move');channel.advance(60);before=s.world.position.copy()
        if cancel_mode=='cancel':
            pending.completion.cancel()
            with pytest.raises(asyncio.CancelledError):await pending.completion
        else:
            with pytest.raises(asyncio.TimeoutError):await asyncio.wait_for(pending.completion,.001)
        assert channel.poll('move')['client']=='sent'
        with pytest.raises(ValueError,match='busy'):channel.submit('right 20','wrong-new')
        other=asyncio.create_task(client.wait('move','move'))
        for _ in range(1000):
            channel.advance(12);await asyncio.sleep(0)
            if channel.poll('move')['client']=='ack_ok':break
        result=await other
        assert result['request_id']=='move' and not result['device_stop_confirmed']
        assert np.linalg.norm(s.world.position-before)>.5
        assert pending.completion.cancelled()
        explicit=client.submit('stop','explicit-stop')
        for _ in range(50):
            channel.advance(12);await asyncio.sleep(0)
            if channel.poll('explicit-stop')['client']=='ack_ok':break
        result=await explicit.completion
        assert result['request_id']=='explicit-stop' and result['device_stop_confirmed']
        assert channel.poll('move')['wire']=='forward 100'
    try:asyncio.run(scenario())
    finally:s.close()

def test_end_simulation_does_not_forge_stopped_operation(project):
    s=session(project);s.command('forward 100','move');s.advance(60)
    operation=s.operation.copy();velocity=s.world.velocity.copy();s.close()
    assert s.finished and s.operation==operation and s.operation['client']=='sent'
    np.testing.assert_array_equal(s.world.velocity,velocity)
    assert np.linalg.norm(velocity)>0
