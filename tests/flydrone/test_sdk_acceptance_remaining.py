"""Missing SDK acceptance cases; no real aircraft/network destination."""
from pathlib import Path
import socket
import numpy as np
import pytest
from flydrone.tellosim.visual import VisualSession
from flydrone.tellosim.sdk.loopback_gateway import LoopbackGateway
from tests.flydrone.test_rigid_training import complete
ROOT=Path(__file__).resolve().parents[2]

def udp_command(g,s,wire,limit=20000):
    s.sendto(wire.encode(),g.address)
    for _ in range(limit):
        g.tick()
        try:return s.recv(1024).decode()
        except BlockingIOError:pass
    pytest.fail('UDP fixture response timeout')

def test_sdk_state_errors_do_not_move_or_fake_flight():
    s=VisualSession(ROOT,recording_enabled=False);before=s.world.position.copy()
    with pytest.raises(ValueError):s.command('takeoff','no-sdk')
    np.testing.assert_array_equal(s.world.position,before);assert not s.airborne
    complete(s,'command','sdk')
    with pytest.raises(ValueError):s.command('forward 20','ground')
    complete(s,'takeoff','up');before=s.world.position.copy()
    with pytest.raises(ValueError):s.command('takeoff','duplicate-up')
    np.testing.assert_array_equal(s.world.position,before);assert s.airborne
    complete(s,'land','down')
    with pytest.raises(ValueError):s.command('land','duplicate-down')
    assert not s.airborne;s.close()

def test_udp_and_inprocess_same_wire_sequence():
    g=LoopbackGateway(ROOT);sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);sock.bind(('127.0.0.1',0));sock.setblocking(False)
    direct=VisualSession(ROOT,recording_enabled=False)
    try:
        wires=['takeoff','command','forward 20','speed 10','speed?','takeoff','go 20 0 0 20','go 21 0 0 20','cw 30','ccw 30','forward 20','stop','land','rc 0 0 0 0','streamon','sdk?','sn?']
        for i,wire in enumerate(wires):
            expected='error'
            try:complete(direct,wire,str(i));expected=direct.requests[str(i)]['response']
            except ValueError:pass
            assert udp_command(g,sock,wire)==expected,wire
        np.testing.assert_allclose(g.session.world.position,direct.world.position,atol=.005)
        assert not g.session.airborne and not direct.airborne
    finally:g.close();sock.close();direct.close()

def test_watchdog_idle_query_and_long_busy_command():
    g=LoopbackGateway(ROOT);sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);sock.bind(('127.0.0.1',0));sock.setblocking(False)
    try:
        g.session.world.reset((-2.5,0,.047))
        assert udp_command(g,sock,'command')=='ok';assert udp_command(g,sock,'speed 10')=='ok';assert udp_command(g,sock,'takeoff')=='ok'
        sock.sendto(b'forward 500',g.address)
        for _ in range(1801):g.tick()
        assert g.pending is not None and not any(o['verb']=='land' for o in g.session.history)
        for _ in range(8000):
            g.tick()
            if g.pending is None:break
        assert sock.recv(100)==b'ok';assert g.session.world.position[0]==pytest.approx(2.5,abs=.04)
        for _ in range(1798):g.tick()
        assert not any(o['verb']=='land' for o in g.session.history)
        assert udp_command(g,sock,'battery?').isdigit()
        for _ in range(1798):g.tick()
        assert not any(o['verb']=='land' for o in g.session.history)
        for _ in range(4):g.tick()
        assert g.session.operation['verb']=='land'
        for _ in range(1200):g.tick()
        assert not g.session.airborne and not g.session.collision_latched
    finally:g.close();sock.close()

@pytest.mark.parametrize('speed',[10,100])
@pytest.mark.parametrize('distance',[20,500])
def test_distance_speed_boundaries_use_physical_trajectory(speed,distance):
    s=VisualSession(ROOT,recording_enabled=False);s.world.reset((-2.5,0,.047))
    try:
        complete(s,'command','sdk');complete(s,'takeoff','up');complete(s,f'speed {speed}','speed')
        start=s.world.position.copy();s.command(f'forward {distance}','move');points=[]
        for _ in range(12000):
            s.advance(1);points.append(s.world.position)
            if s.operation['client']!='sent':break
        assert s.operation['client']=='ack_ok' and not s.collision_latched
        assert len(points)>120
        assert s.world.position[0]-start[0]==pytest.approx(distance/100,abs=.04)
        assert np.max(np.linalg.norm(np.diff(points,axis=0),axis=1))<.02
    finally:s.close()
