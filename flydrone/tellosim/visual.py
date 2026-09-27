"""Local-only visual simulation. No sockets, aircraft SDK, or training state.

The browser is a renderer: every trajectory point is integrated by MuJoCo.
New sessions use a six-DOF body-axis thrust controller; legacy replay remains versioned.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import threading
import time
import uuid
from collections import deque

import numpy as np

from .physics.world import SimWorld, WorldConfig
from .pose import PoseSnapshot
from .sdk.codec import decode_command


def packed(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_bytes(packed(value))
    temp.replace(path)


class Recording:
    """Independent, indexed streams; bounded buffers and a strict disk quota."""
    def __init__(self, directory, manifest, quota=256 * 1024 * 1024):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest = {"schema_version": "tellosim.view/2.0", **manifest,
                         "streams": {}, "partial": False, "bytes": 0, "complete": False}
        self.buffers = {}
        self.quota = quota
        self.save()

    def save(self):
        atomic_json(self.directory / "manifest.json", self.manifest)

    def add(self, kind, tick, data):
        if self.manifest["partial"]:
            return
        row = {"run_id": self.manifest["run_id"], "epoch": self.manifest["epoch"],
               "episode_id": self.manifest["episode_id"], "env_id": 0,
               "sim_tick": int(tick), "time_s": int(tick) / 120, **data}
        self.buffers.setdefault(kind, []).append(row)
        if len(self.buffers[kind]) >= 240:
            self.flush(kind)

    def flush(self, kind=None):
        for key in ([kind] if kind else list(self.buffers)):
            rows = self.buffers.get(key, [])
            if not rows:
                continue
            self.buffers[key] = []
            body = b"\n".join(packed(row) for row in rows) + b"\n"
            if self.manifest["bytes"] + len(body) > self.quota:
                self.manifest.update(partial=True, partial_reason="recording_quota_exceeded")
                self.save()
                return
            chunks = self.manifest["streams"].setdefault(key, [])
            name = f"{key}-{len(chunks):05d}.jsonl"
            (self.directory / name).write_bytes(body)
            chunks.append({"file": name, "first_tick": rows[0]["sim_tick"],
                           "last_tick": rows[-1]["sim_tick"], "count": len(rows),
                           "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()})
            self.manifest["bytes"] += len(body)
        self.save()

    def close(self, outcome="stopped"):
        self.flush()
        self.manifest.update(complete=True, outcome=outcome)
        self.save()


class NullRecording:
    """Training rollouts can share the executor without allocating disk traces."""
    def __init__(self, manifest):
        self.manifest = {**manifest, "partial": False}

    def add(self, *args, **kwargs):
        pass

    def save(self):
        pass

    def close(self, outcome="stopped"):
        self.manifest.update(complete=True, outcome=outcome)


class VisualWorld(SimWorld):
    """Room geometry is used for both MJCF collisions and the scene descriptor."""
    def __init__(self, scene):
        self.scene = scene
        self.powered = False
        self.speed_limit = 0.2
        self.trajectory = None
        gains=scene.get('low_level',{})
        super().__init__(WorldConfig(controller_profile=scene.get('controller','rigid_body_thrust_v2'),room_half_extent_m=scene["room_size_m"][0] / 2,
                                    position_kp=float(gains.get('position_kp',.7)),
                                    velocity_kd=float(gains.get('velocity_kd',1.8))))
        self.reset((0, 0, 0.047))

    def _xml(self):
        sx, sy, sz = self.scene["room_size_m"]
        geometries = [f'<geom name="floor" type="plane" size="{sx/2} {sy/2} .05"/>']
        boxes = [([sx/2+.05,0,sz/2],[.05,sy/2,sz/2]),
                 ([-sx/2-.05,0,sz/2],[.05,sy/2,sz/2]),
                 ([0,sy/2+.05,sz/2],[sx/2,.05,sz/2]),
                 ([0,-sy/2-.05,sz/2],[sx/2,.05,sz/2]),
                 ([0,0,sz+.05],[sx/2,sy/2,.05])]
        boxes += [(o["center_m"], o["half_extents_m"]) for o in self.scene["obstacles"]]
        for i, (position, size) in enumerate(boxes):
            geometries.append(f'<geom name="solid_{i}" type="box" pos="{" ".join(map(str,position))}" size="{" ".join(map(str,size))}"/>')
        return f'''<mujoco model="tellosim-visual"><option timestep="{1/120}" gravity="0 0 -9.81" integrator="implicitfast"/>
        <worldbody>{''.join(geometries)}<body name="tello" pos="0 0 .047"><freejoint/>
        <geom name="body" type="box" size=".1 .1 .045" mass=".1"/></body></worldbody></mujoco>'''

    def reset(self, position=None):
        super().reset(position)
        self.trajectory = None
        self.reference_velocity = np.zeros(3)
        self.reference_acceleration = np.zeros(3)

    def set_target(self, target):
        super().set_target(target)
        if self.config.controller_profile == 'rigid_body_thrust_v2':
            self.trajectory={'start':self.position.tolist(),'goal':self.target.tolist(),
                'speed':self.speed_limit,'acceleration':self.config.max_accel_mps2*.5,'elapsed':0.}

    def _apply_controller(self):
        if not self.powered:
            return
        if self.config.controller_profile == 'rigid_body_thrust_v2':
            from .physics.rigid import translation_reference
            target=self.target.copy()
            if self.trajectory is not None:
                q=self.trajectory
                self.target,self.reference_velocity,self.reference_acceleration=translation_reference(
                    q['start'],q['goal'],q['speed'],q['acceleration'],q['elapsed'])
                q['elapsed']+=self.config.dt
            super()._apply_controller()
            self.target=target
            return
        # Limit the desired velocity, retaining the original bounded PD force.
        target = self.target.copy()
        delta = target - self.position
        distance = float(np.linalg.norm(delta))
        velocity_error_limit=self.speed_limit*self.config.velocity_kd/self.config.position_kp
        if distance > velocity_error_limit:
            self.target = self.position + delta / distance * velocity_error_limit
        super()._apply_controller()
        self.target = target

    def step(self, ticks=1):
        # Unwrapped target allows a genuine 360 degree command.
        saved_yaw = self.target_yaw
        difference = saved_yaw - self.yaw_rad
        if self.config.controller_profile != 'rigid_body_thrust_v2':
            self.target_yaw = self.yaw_rad + max(-math.pi, min(math.pi, difference))
        rows = super().step(ticks)
        self.target_yaw = saved_yaw
        row = rows[-1]
        contacts = [(self.data.contact[i].geom1, self.data.contact[i].geom2) for i in range(self.data.ncon)]
        row["ground_contact"] = any(0 in pair for pair in contacts)
        row["collision"] = any(0 not in pair for pair in contacts)
        row["out_of_bounds"] = False  # closed room is enforced by solid geometry
        return rows


def scene_config(root, obstacles=False):
    path = Path(root) / "configs/tellosim" / ("world_obstacles_demo.json" if obstacles else "world_room6.json")
    scene = json.loads(path.read_text())
    scene.update(solid_walls=True, target_xyz_m=[0.4, -0.4, 1.0], target_radius_m=0.15,
                 stable_hold_required_s=2.0, controller="rigid_body_thrust_v2")
    return scene


class VisualSession:
    ROUTE = ["command", "takeoff", "speed 20", "forward 40", "cw 90", "forward 40", "stop", "land"]

    def __init__(self, root, mode="manual", obstacles=False, seed=11, run_id=None, quota=256*1024*1024,
                 scene=None, start_position=None, recording_enabled=True, initial_yaw_rad=0.):
        self.run_id = run_id or f"{mode}-{uuid.uuid4().hex[:12]}"
        self.epoch = uuid.uuid4().hex
        self.mode = mode
        self.scene = scene or scene_config(root, obstacles)
        self.world = VisualWorld(self.scene)
        if start_position is not None:
            self.world.reset(tuple(start_position))
        if not np.isfinite(initial_yaw_rad):raise ValueError("nonfinite initial yaw")
        if initial_yaw_rad:
            import mujoco
            self.world.data.qpos[3:7]=[math.cos(initial_yaw_rad/2),0,0,math.sin(initial_yaw_rad/2)]
            self.world.target_yaw=float(initial_yaw_rad);self.world._yaw_state=float(initial_yaw_rad)
            mujoco.mj_forward(self.world.model,self.world.data)
        self.collision_latched = False
        self.tick = 0
        self.seq = 0
        self.sdk = False
        self.airborne = False
        self.operation = None
        self.history = deque(maxlen=100)
        self.requests = {}
        self.rng = np.random.default_rng(seed)
        self.sensor_history = deque(maxlen=4)
        self.trail = deque(maxlen=100)
        self.queue = deque(self.ROUTE if mode == "script" else [])
        self.lock = threading.RLock()
        self.finished = False
        self.stop_event = threading.Event()
        self.snapshot = {}
        self.operation_hold = 0.
        self.stable_hold = 0.
        self.thread = None
        descriptor = {
            "run_id": self.run_id, "epoch": self.epoch, "episode_id": "episode-0", "seed": seed,
            "policy_source": "script_not_learned_policy" if mode == "script" else "manual_sandbox",
            "mode": "LIVE", "scene": self.scene, "physics_hz": 120,
            "limitations": ["Engineering surrogate, not calibrated to a real Tello",
                "26D pose telemetry has 11 reserved channels; not a TS1 26/34 policy",
                "No model/neural activity in manual or scripted runs"],
        }
        self.recording = (Recording(Path(root) / "reports/vis/tellosim" / self.run_id, descriptor, quota)
                          if recording_enabled else NullRecording(descriptor))
        position=self.world.position.tolist()
        self.recording.add("trajectory",0,{"x_m":position[0],"y_m":position[1],"z_m":position[2],
            "yaw_rad":self.world.yaw_rad,"quaternion_wxyz":self.world.data.qpos[3:7].copy().tolist(),"velocity_mps":[0.,0.,0.],"speed_mps":0.,"ground_contact":False,"collision":False})
        self._publish()

    def command(self, wire, request_id, lose_reply=False):
        with self.lock:
            if self.finished:
                raise ValueError("session is closed")
            if not isinstance(request_id, str) or not 1 <= len(request_id) <= 100:
                raise ValueError("request_id required (1..100 characters)")
            if request_id in self.requests:
                old = self.requests[request_id]
                if old["wire"] != wire or old["lose_reply"] != lose_reply:
                    raise ValueError("request_id reused with different content")
                return old.copy()
            if len(self.requests) >= 256 and wire != "stop":
                raise ValueError("sandbox command limit reached; close and create a new session")
            cmd = decode_command(wire)
            if cmd.verb not in {"command", "takeoff", "land", "stop", "speed", "forward", "back", "left", "right", "up", "down", "cw", "ccw", "go", "speed?", "battery?", "time?", "sdk?", "sn?", "hardware?"}:
                raise ValueError("command not supported by visual sandbox")
            active = self.operation
            if active and active["client"] in {"sent", "unknown_execution"}:
                if cmd.verb != "stop":
                    raise ValueError("busy or unknown execution: only protective stop is allowed")
                if "restore_speed_mps" in active:self.world.speed_limit=active["restore_speed_mps"]
                if active["device"] == "running":
                    active.update(device="cancelled_local", completed_tick=self.tick)
                # Unknown remains unknown in history even after protective stop.
                if active["client"] == "sent":
                    active["client"] = "cancelled_local"
                self._operation_event(active)
            if cmd.verb != "command" and not self.sdk:
                raise ValueError("enter SDK mode with command first")
            if cmd.verb == "takeoff" and self.airborne:
                raise ValueError("already airborne")
            if cmd.verb not in {"command", "takeoff", "speed", "stop"} and not cmd.verb.endswith("?") and not self.airborne:
                raise ValueError("drone is grounded")
            op = {"operation_id": f"op-{len(self.requests)+1}", "request_id": request_id,
                  "wire": wire, "verb": cmd.verb, "sent_tick": self.tick, "device": "running",
                  "client": "sent", "lose_reply": bool(lose_reply), "response": None}
            self.requests[request_id] = op
            self.history.append(op)
            self.operation = op
            self.operation_hold = 0.
            if cmd.verb.endswith("?"):
                response={"speed?":str(round(self.world.speed_limit*100)),
                    "battery?":str(round(max(0.,1-self.tick/120/1200)*100)),
                    "time?":str(self.tick//120),"sdk?":"3.0-sim-subset",
                    "sn?":"SIMULATED-NOT-AIRCRAFT","hardware?":"TelloSim-surrogate"}[cmd.verb]
                self._complete(True)
                op["response"]=None if lose_reply else response
                self._operation_event(op);self._publish()
                return op.copy()
            target = self.world.position.copy()
            yaw = self.world.yaw_rad
            if cmd.verb == "command":
                self.sdk = True
            elif cmd.verb == "takeoff":
                self.world.powered = True
                self.airborne = True
                target[2] = 1.
            elif cmd.verb == "land":
                target[2] = -0.08
            elif cmd.verb == "speed":
                self.world.speed_limit = cmd.args[0] / 100
            elif cmd.verb == "go":
                x,y,z,speed=cmd.args
                target += np.asarray([math.cos(yaw)*x-math.sin(yaw)*y,
                                      math.sin(yaw)*x+math.cos(yaw)*y,z])/100
                op["restore_speed_mps"]=self.world.speed_limit
                self.world.speed_limit=speed/100
            elif cmd.verb in {"cw", "ccw"}:
                self.world.set_yaw(yaw + math.radians(cmd.args[0]) * (-1 if cmd.verb == "cw" else 1))
            elif cmd.verb in {"forward", "back", "left", "right", "up", "down"}:
                vectors = {"forward": [math.cos(yaw), math.sin(yaw), 0], "back": [-math.cos(yaw),-math.sin(yaw),0],
                           "left": [-math.sin(yaw),math.cos(yaw),0], "right": [math.sin(yaw),-math.cos(yaw),0],
                           "up": [0,0,1], "down": [0,0,-1]}
                target += np.asarray(vectors[cmd.verb]) * cmd.args[0] / 100
            self.world.set_target(tuple(target))
            if cmd.verb == "stop":
                self.world.set_yaw(yaw)
            op["deadline_tick"] = self.tick + int((np.linalg.norm(target-self.world.position) / max(.05,self.world.speed_limit/1.8) + abs(self.world.target_yaw-yaw)/.6 + 18) * 120)
            self._operation_event(op)
            if cmd.verb in {"command", "speed"}:
                self._complete(True)
            self._publish()
            return op.copy()

    def _operation_event(self, op):
        self.recording.add("operation", self.tick, {"operation": op.copy()})

    def _complete(self, ok):
        op = self.operation
        op.update(device="completed" if ok else "failed", completed_tick=self.tick,
                  client="unknown_execution" if op["lose_reply"] else ("ack_ok" if ok else "ack_error"),
                  response=None if op["lose_reply"] else ("ok" if ok else "error"))
        self._operation_event(op)
        if "restore_speed_mps" in op:self.world.speed_limit=op["restore_speed_mps"]
        if not ok:
            self.world.hold()
            self.world.set_yaw(self.world.yaw_rad)
            self.queue.clear()

    def advance(self, count=12):
        with self.lock:
            if self.finished:
                return
            if self.mode == "script" and self.queue and (not self.operation or self.operation["client"] == "ack_ok"):
                self.command(self.queue.popleft(), f"route-{len(self.requests)}")
            for _ in range(count):
                row = self.world.step()[0]
                self.tick = row["sim_tick"]
                row["airborne"] = self.airborne
                if row["ground_contact"] and self.airborne and self.operation and self.operation["verb"] not in {"takeoff", "land"}:
                    row["collision"] = True
                self.collision_latched |= row["collision"]
                self.recording.add("trajectory", self.tick, row)
                op = self.operation
                if op and op["device"] == "running":
                    duration = (self.tick-op["sent_tick"])/120
                    settled = np.linalg.norm(self.world.target-self.world.position) < .035 and row["speed_mps"] < .035
                    if op["verb"] in {"cw", "ccw"}:
                        settled = abs(self.world.target_yaw-self.world.yaw_rad) < math.radians(1)
                        if self.world.config.controller_profile=="rigid_body_thrust_v2":
                            settled = settled and np.linalg.norm(self.world.data.qvel[3:6])<math.radians(5)
                    if op["verb"] == "stop":
                        settled = row["speed_mps"] < .035 and duration > .5
                    if op["verb"] == "land":
                        settled = row["ground_contact"] and row["speed_mps"] < .04
                        if settled:
                            self.world.powered = False
                            self.airborne = False
                    if row["collision"] or self.tick > op["deadline_tick"]:
                        self._complete(False)
                    else:
                        self.operation_hold = self.operation_hold+1/120 if settled else 0.
                        if settled and duration > .2 and (self.world.config.controller_profile!='rigid_body_thrust_v2' or self.operation_hold>=.25):
                            self._complete(True)
                distance = np.linalg.norm(self.world.position-np.asarray(self.scene["target_xyz_m"]))
                self.stable_hold = self.stable_hold+1/120 if distance <= .15 and row["speed_mps"] <= .15 else 0.
                if self.tick % 12 == 0:
                    self._publish(row)
            if self.mode == "script" and not self.queue and self.operation and self.operation["device"] in {"completed", "failed"}:
                self.close("script_completed" if self.operation["client"] == "ack_ok" else "script_failed")
            elif self.tick >= 120 * 180:
                self.close("sandbox_time_limit_180s")

    def _publish(self, row=None, publish=True):
        self.seq += 1
        pos = self.world.position.tolist()
        velocity = self.world.velocity.tolist()
        truth = {"position_m": pos, "velocity_mps": velocity, "yaw_rad": self.world.yaw_rad,
                 "quaternion_wxyz":self.world.data.qpos[3:7].copy().tolist(),
                 "angular_velocity_body_rad_s":self.world.data.qvel[3:6].copy().tolist(),
                 "airborne": self.airborne, "collision": bool(row and row["collision"]),
                 "ground_contact": bool(row and row.get("ground_contact"))}
        self.sensor_history.append((self.tick, pos))
        sensor_tick, delayed_pos = self.sensor_history[0]
        valid = self.tick % 600 < 540
        observed = (np.asarray(delayed_pos)+self.rng.normal(0,.015,3)).tolist() if valid else None
        sensor = {"position_m": observed, "valid": valid, "sample_tick": sensor_tick,
                  "age_s": (self.tick-sensor_tick)/120, "noise_sigma_m": .015,
                  "source": "simulated_external_pose_10hz", "sdk_height_m": pos[2], "battery": None}
        obs = PoseSnapshot(tuple(pos),tuple(velocity),self.world.yaw_rad,self.airborne,self.tick).observation26(self.scene["target_xyz_m"]).tolist()
        frame = {"run_id": self.run_id, "epoch": self.epoch, "episode_id": "episode-0", "env_id": 0,
                 "seq": self.seq, "sim_tick": self.tick, "time_s": self.tick/120, "truth": truth,
                 "sensor": sensor, "observation": obs, "observation_valid": [True]*15+[False]*11,
                 "observation_source": "truth_derived_pose_telemetry_not_policy_input",
                 "policy": None, "brain": None, "stable_hold_s": self.stable_hold,
                 "operation": self.operation.copy() if self.operation else None,
                 "history": [o.copy() for o in self.history], "finished": self.finished,
                 "recording_partial": self.recording.manifest["partial"]}
        self.trail.append(pos)
        if self.mode != "training":
            self.recording.add("transition", self.tick, {k:v for k,v in frame.items() if k not in {"history", "seq"}})
        frame["trail"] = list(self.trail)
        assert len(packed(frame)) <= 65536
        if publish:
            self.snapshot = frame
        return frame

    def export_state(self):
        import copy
        if self.thread is not None:raise RuntimeError('snapshot requires a synchronous session')
        excluded={'world','lock','stop_event','thread','recording','rng'}
        return {'world':self.world.export_state(),
            'fields':copy.deepcopy({k:v for k,v in self.__dict__.items() if k not in excluded}),
            'rng':copy.deepcopy(self.rng.bit_generator.state)}

    def restore_state(self,state):
        import copy
        if self.thread is not None or not isinstance(self.recording,NullRecording):
            raise RuntimeError('exact session restore requires unrecorded synchronous training; replay exports are separate runs')
        if self.scene!=state['fields']['scene']:raise ValueError('scene mismatch')
        self.world.restore_state(state['world'])
        self.__dict__.update(copy.deepcopy(state['fields']))
        self.rng.bit_generator.state=copy.deepcopy(state['rng'])
        self.stop_event.clear()
        if self.finished:self.stop_event.set()

    def start(self):
        def run():
            try:
                while not self.stop_event.is_set():
                    started = time.monotonic()
                    self.advance()
                    self.stop_event.wait(max(0, .1-(time.monotonic()-started)))
            except Exception as exc:
                with self.lock:
                    self.recording.manifest["error"] = f"{type(exc).__name__}: {exc}"
                    self.close("runtime_error")
        self.thread = threading.Thread(target=run, name=self.run_id, daemon=True)
        self.thread.start()

    def close(self, outcome="closed_by_user"):
        with self.lock:
            if self.finished:
                return
            self.finished = True
            self.stop_event.set()
            final_frame = self._publish(publish=False)
            self.recording.manifest["duration_s"] = self.tick/120
            self.recording.close(outcome)
            self.snapshot = final_frame


def import_golden(root, source_directory="artifacts/golden_episode_verified", run_id="golden-episode", policy_source="flybrain_reservoir_features"):
    """Adapt existing evidence without claiming new walls/sensors/policy channels."""
    root = Path(root)
    source = root / source_directory / "replay.jsonl"
    if not source.exists():
        return
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    output = root / "reports/vis/tellosim" / run_id
    if (output / "manifest.json").exists():
        previous = json.loads((output / "manifest.json").read_text())
        if previous.get("source_sha256") == source_hash and previous.get("complete"):
            return
        raise ValueError("Golden visual import already exists with a different source")
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    scene = {"room_size_m": [6,6,3], "solid_walls": False, "obstacles": [],
             "landing_pad": {"center_m": [0,0], "radius_m": .35},
             "target_xyz_m": rows[0]["world"]["target_xyz_m"], "target_radius_m": .3,
             "stable_hold_required_s": 1., "controller": "legacy_golden_surrogate"}
    recording = Recording(output, {"run_id": run_id, "epoch": source_hash[:16],
        "episode_id": rows[0]["episode_id"], "mode": "REPLAY", "policy_source": policy_source,
        "source_sha256": source_hash, "source": str(source.relative_to(root)), "scene": scene,
        "policy_checkpoint_sha256": rows[0]["policy_checkpoint_sha256"],
        "graph_sha256": rows[0]["graph_sha256"],
        "limitations": ["Legacy Golden: floor physics only; room outline is a boundary, not solid walls",
            "Actual policy input is 8D / encoded 14D / features 128D; 26D is pose telemetry",
            "Only neural substep 3 of 4 recorded; no noisy external pose recorded"]})
    for r in rows:
        start = round(r["decision_time_s"]*120)
        end = round(r["sim_time_s"]*120)
        command = r["command"]
        op = {"operation_id":command["command_id"], "wire":" ".join([command["command_type"],*map(str,command["command_args"])]) ,
              "sent_tick":start, "device":"running", "client":"sent", "response":None}
        recording.add("operation",start,{"operation":op.copy()})
        recording.add("transition",start,{"observation":r["observation"], "observation_valid":[True]*15+[False]*11,
                      "brain_observation":r["brain_observation"], "policy":r["policy"],
                      "reward":None, "operation":op.copy(), "sensor":None, "brain":r["brain"], "neural_substep":r["neural_substep"]})
        recording.add("neural",start,{"brain":r["brain"],"neural_substep":r["neural_substep"]})
        if start == 0:
            initial = r["state_before"]["position_xyz_m"]
            recording.add("trajectory",0,{"x_m":initial[0],"y_m":initial[1],"z_m":initial[2],"yaw_rad":0,"speed_mps":0})
        for sample in r["physics_samples"]:
            recording.add("trajectory",sample["sim_tick"],sample)
        op.update(device=command["result"]["device_execution"],client=command["result"]["phase"],
                  response=command["result"]["raw_response"],completed_tick=end)
        recording.add("operation",end,{"operation":op.copy()})
        recording.add("transition",end,{"reward":r["reward"], "stable_hold_s":r["stable_hold_s"],
                      "success":r["success"],"operation":op.copy(),"result_only":True})
    recording.manifest["duration_s"] = rows[-1]["sim_time_s"]
    recording.close("golden_success" if rows[-1]["success"] else "golden_failure")
