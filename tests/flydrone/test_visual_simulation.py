import hashlib
import json
from pathlib import Path
import shutil
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from flydrone.tellosim.visual import Recording, VisualSession, VisualWorld, scene_config, packed
from flyview.server import Handler, Viewer
from flyview.tellosim_api import Observatory
from http.server import ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def project(tmp_path):
    (tmp_path / "configs").mkdir()
    shutil.copytree(ROOT / "configs/tellosim",tmp_path / "configs/tellosim")
    return tmp_path


def settle(session,limit=10000):
    for _ in range(limit//12):
        session.advance()
        if session.operation["device"] != "running":
            return session.operation.copy()
    pytest.fail("motion did not finish")


def test_route_physics_and_chunk_reconstruction(project):
    session=VisualSession(project,"script")
    for _ in range(2000):
        session.advance()
        if session.finished: break
    assert session.finished
    assert session.recording.manifest["outcome"]=="script_completed"
    assert len(session.history)==8
    assert all(o["client"]=="ack_ok" for o in session.history)
    assert not session.world.powered and not session.airborne
    assert abs(session.world.position[2]-.045)<.003
    positions=[]
    for chunk in session.recording.manifest["streams"]["trajectory"]:
        data=(session.recording.directory/chunk["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest()==chunk["sha256"]
        rows=[json.loads(line) for line in data.splitlines()]
        assert len(rows)==chunk["count"]
        positions.extend(rows)
    assert len(positions)>1200  # every 120 Hz point, not 51 preselected waypoints
    assert [p["sim_tick"] for p in positions]==list(range(len(positions)))
    xyz=np.asarray([[p["x_m"],p["y_m"],p["z_m"]] for p in positions])
    assert np.linalg.norm(np.diff(xyz,axis=0),axis=1).max()<.01
    assert xyz[:,2].max()>.95
    assert not any(p["collision"] for p in positions)  # normal landing isn't collision
    assert positions[-1]["ground_contact"]


def test_lost_ack_idempotency_busy_and_protective_stop(project):
    session=VisualSession(project)
    session.command("command","sdk")
    session.command("takeoff","takeoff",True)
    result=settle(session)
    assert result["device"]=="completed" and result["client"]=="unknown_execution"
    assert session.command("takeoff","takeoff",True)==result
    with pytest.raises(ValueError,match="reused"):
        session.command("land","takeoff")
    with pytest.raises(ValueError,match="busy or unknown"):
        session.command("forward 20","blocked")
    session.command("stop","protective")
    assert settle(session)["client"]=="ack_ok"
    assert session.requests["takeoff"]["client"]=="unknown_execution"
    session.close()


def test_scene_collision_uses_config_geometry(project):
    world=VisualWorld(scene_config(project,True))
    world.reset((0,0,.6));world.powered=True;world.speed_limit=1
    world.set_target((2,0,.6))
    contacts=[]
    for _ in range(1200):contacts.extend(world.step())
    assert any(row["collision"] for row in contacts)
    assert world.position[0]<.8
    # Empty room has a real wall at +3 m, not just the outline.
    world=VisualWorld(scene_config(project,False))
    world.reset((2.7,0,1));world.powered=True;world.speed_limit=1
    world.set_target((4,0,1))
    assert any(world.step()[0]["collision"] for _ in range(1200))
    assert world.position[0]<3


def test_yaw_360_and_frame_budget(project):
    session=VisualSession(project)
    session.command("command","sdk");session.command("takeoff","up");settle(session)
    session.command("cw 360","yaw")
    assert settle(session)["client"]=="ack_ok"
    assert session.world.yaw_rad < -6.2
    assert len(packed(session.snapshot))<=65536
    session.close()


def test_invalid_sensor_is_null_and_reserved_channels_invalid(project):
    session=VisualSession(project)
    for _ in range(46):session.advance()
    frame=session.snapshot
    assert not frame["sensor"]["valid"] and frame["sensor"]["position_m"] is None
    assert frame["sensor"]["sample_tick"]<frame["sim_tick"]
    assert len(frame["observation"])==26
    assert frame["observation_valid"]==[True]*15+[False]*11
    assert frame["brain"] is None and frame["policy"] is None
    session.close()


def test_recording_quota_preserves_valid_prefix(tmp_path):
    r=Recording(tmp_path,{"run_id":"quota","epoch":"one","episode_id":"e"},quota=400)
    r.add("trajectory",0,{"x_m":0});r.flush()
    r.add("trajectory",1,{"huge":"x"*400});r.flush();r.close()
    assert r.manifest["partial"] and r.manifest["partial_reason"]=="recording_quota_exceeded"
    assert r.manifest["bytes"]<=400
    assert len(r.manifest["streams"]["trajectory"])==1


def test_whitespace_command_rejected(project):
    s=VisualSession(project)
    with pytest.raises(ValueError):s.command("   ","bad")
    s.close()


@pytest.fixture
def server(project):
    httpd=ThreadingHTTPServer(("127.0.0.1",0),Handler)
    httpd.viewer=Viewer(project,project / "reports/vis/views")
    httpd.tellosim=Observatory(project)
    t=threading.Thread(target=httpd.serve_forever,daemon=True);t.start()
    yield httpd
    httpd.shutdown();httpd.tellosim.close();httpd.server_close();t.join(2)


def call(server,path,body=None,headers=None):
    url=f"http://127.0.0.1:{server.server_port}"
    defaults={"Content-Type":"application/json","Origin":url,"X-Sandbox-Token":server.tellosim.token}
    defaults.update(headers or {})
    req=urllib.request.Request(url+path,data=packed(body) if body is not None else None,headers=defaults)
    try:
        with urllib.request.urlopen(req,timeout=3) as response:
            data=response.read();return response.status,json.loads(data) if 'json' in response.headers.get('Content-Type','') else data
    except urllib.error.HTTPError as e:return e.code,json.loads(e.read())


@pytest.mark.parametrize("headers",[{"Host":"evil.example"},{"Origin":"http://evil.example"},{"X-Sandbox-Token":"wrong"},{"Origin":"null"}])
def test_post_security(server,headers):
    assert call(server,"/api/tellosim/sessions",{},headers)[0]==403
    assert not server.tellosim.sessions


def test_sandbox_epoch_isolation_and_readonly_sources(server):
    _,one=call(server,"/api/tellosim/sessions",{"mode":"manual"})
    _,two=call(server,"/api/tellosim/sessions",{"mode":"manual"})
    assert one["epoch"]!=two["epoch"] and one["run_id"]!=two["run_id"]
    endpoint=f'/api/tellosim/sessions/{one["run_id"]}/command'
    assert call(server,endpoint,{"epoch":two["epoch"],"wire":"command","request_id":"wrong"})[0]==422
    assert call(server,endpoint,{"epoch":one["epoch"],"wire":"command","request_id":"sdk"})[0]==200
    assert not server.tellosim.sessions[two["run_id"]].sdk
    _,script=call(server,"/api/tellosim/sessions",{"mode":"script"})
    assert call(server,f'/api/tellosim/sessions/{script["run_id"]}/command',{"epoch":script["epoch"],"wire":"stop","request_id":"x"})[0]==403
    assert call(server,"/api/views/golden-episode/command",{"wire":"takeoff"})[0]==405
    assert call(server,"/api/tellosim/runs/../manifest")[0] in {404,422}
    assert call(server,f'/api/tellosim/runs/{one["run_id"]}/chunks/secrets.json')[0]==404
    assert call(server,"/")[0]==200 and call(server,"/tellosim")[0]==200


@pytest.mark.parametrize('payload',[b'{"mode":"manual","mode":"script"}', b'{"seed":NaN}', b'{"obstacles":"false"}', b'{"real_ip":"192.168.10.1"}'])
def test_ambiguous_or_unknown_json_never_creates_session(server,payload):
    url=f"http://127.0.0.1:{server.server_port}"
    req=urllib.request.Request(url+'/api/tellosim/sessions',data=payload,
        headers={"Content-Type":"application/json","Origin":url,"X-Sandbox-Token":server.tellosim.token})
    with pytest.raises(urllib.error.HTTPError) as error:urllib.request.urlopen(req,timeout=3)
    assert error.value.code==422 and not server.tellosim.sessions
