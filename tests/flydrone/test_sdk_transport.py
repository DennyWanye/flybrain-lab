from pathlib import Path
import socket
import numpy as np
import pytest
from flydrone.tellosim.visual import VisualSession
from flydrone.tellosim.sdk.codec import decode_command,CommandError
from flydrone.tellosim.sdk.channels import CommandChannel,FaultProfile
from flydrone.tellosim.sdk.loopback_gateway import LoopbackGateway
from flydrone.tellosim.sdk.state_codec import parse_state,serialize_state
ROOT=Path(__file__).resolve().parents[2]

def session():return VisualSession(ROOT,recording_enabled=False)
def settle(s):
    for _ in range(1000):
        s.advance(12)
        if s.operation['client']!='sent':return s.operation
    pytest.fail('did not settle')

def test_go_query_share_execution_and_query_does_not_move_target():
    s=session()
    try:
        s.command('command','c');s.command('takeoff','t');assert settle(s)['client']=='ack_ok'
        before=s.world.position.copy();s.command('go 30 30 0 30','g')
        assert settle(s)['client']=='ack_ok'
        np.testing.assert_allclose(s.world.position[:2],before[:2]+.3,atol=.04)
        assert s.world.speed_limit==.2
        target=s.world.target.copy()
        assert s.command('speed?','q')['response']=='20'
        np.testing.assert_array_equal(target,s.world.target)
        assert s.command('sn?','sn')['response']=='SIMULATED-NOT-AIRCRAFT'
        assert parse_state(serialize_state(s))['position_world_m'] is None
    finally:s.close()

def test_request_and_reply_loss_are_distinct_and_never_retried():
    s=session()
    try:
        channel=CommandChannel(s,FaultProfile(request_drop=1))
        op=channel.submit('command','a');channel.advance(7201)
        assert channel.poll('a')['client']=='unknown_execution' and not s.sdk
        assert not s.requests
        with pytest.raises(ValueError):channel.submit('takeoff','b')
        assert channel.submit('command','a')['operation_id']==op['operation_id']
    finally:s.close()
    s=session()
    try:
        channel=CommandChannel(s,FaultProfile(reply_drop=1))
        channel.submit('command','a');channel.advance(7201)
        assert s.sdk and len(s.requests)==1 and channel.poll('a')['client']=='unknown_execution'
        assert s.requests['channel-a']['client']=='ack_ok'
    finally:s.close()

def test_delayed_channel_and_cancelled_request_cannot_execute_later():
    s=session()
    try:
        channel=CommandChannel(s,FaultProfile(request_delay_ticks=12,reply_delay_ticks=12))
        channel.submit('command','a');channel.advance(11);assert not s.sdk
        channel.advance(1);assert s.sdk and channel.poll('a')['client']=='sent'
        channel.advance(12);assert channel.poll('a')['client']=='ack_ok'
        channel.submit('takeoff','b');channel.submit('stop','stop');channel.advance(24)
        assert not s.airborne and channel.poll('b')['client']=='cancelled_local'
    finally:s.close()

def test_loopback_address_and_one_client_isolation():
    for host in ('0.0.0.0','localhost','192.168.10.1','::1'):
        with pytest.raises(ValueError):LoopbackGateway(ROOT,host)
    gateway=LoopbackGateway(ROOT);a=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);b=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    try:
        a.bind(('127.0.0.1',0));a.settimeout(.5);b.bind(('127.0.0.1',0));b.settimeout(.05)
        a.sendto(b'command',gateway.address);gateway.tick();assert a.recv(100)==b'ok'
        b.sendto(b'takeoff',gateway.address);gateway.tick()
        with pytest.raises(socket.timeout):b.recv(100)
        assert not gateway.session.airborne
        a.sendto(b'sdk?',gateway.address);gateway.tick();assert a.recv(100)==b'3.0-sim-subset'
        a.sendto(b'rc 0 0 0 0',gateway.address);gateway.tick();assert a.recv(100)==b'error'
    finally:gateway.close();a.close();b.close()

def test_state_units_finite_values_and_control_characters():
    state=parse_state('h:100;vgx:10;vgy:0;vgz:-10;baro:1.2;bat:50;')
    assert state['height_m']==1 and state['velocity_body_mps']==[1,0,-1] and state['barometer_m']==1.2
    for value in ('h:NaN;','h:1;h:2;','bat:101;','h:-1;'):
        with pytest.raises(ValueError):parse_state(value)
    for value in ('command\v','command\f','command\x7f'):
        with pytest.raises(CommandError):decode_command(value)


def test_approved_command_vectors():
    import json
    from flydrone.tellosim.sdk.codec import encode_command,TelloCommand
    cases=json.loads((ROOT/'tests/fixtures/tellosim-command-vectors.json').read_text())['cases']
    for case in cases:
        if case['valid']:
            command=decode_command(case['wire'])
            assert command.verb==case['canonical']['verb'] and list(command.args)==case['canonical']['args']
            assert decode_command(encode_command(command))==command
        else:
            with pytest.raises(CommandError):decode_command(case['wire'])
    for value in (20.,True,float('nan'),float('inf')):
        with pytest.raises(CommandError):encode_command(TelloCommand('forward',(value,)))
