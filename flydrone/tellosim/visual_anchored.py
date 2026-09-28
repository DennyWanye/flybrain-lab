"""Opt-in SDK executor: retain orthogonal station-keeping references.

The legacy VisualSession is untouched. This executor has no task goal access.
Motion-interrupting STOP captures actual position for braking. Repeated idle STOP,
yaw and speed changes do not ratchet translational disturbance into the target.
"""
import math
import numpy as np
from .visual import VisualSession
from .sdk.codec import decode_command

EXECUTOR_SPEC = {"version": "VisualSession/anchored-2.0", "translation": "current position relative on commanded axes; previous target on orthogonal axes", "idle_stop": "preserve position target; arrest yaw", "protective_stop": "capture current position when interrupting translation", "yaw": "preserve XYZ target", "legacy_default": "VisualSession/1.0 unchanged"}

class AnchoredVisualSession(VisualSession):
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
            # Idle commands retain the existing station-keeping reference. A stop
            # interrupting an active translation still brakes at the current position.
            target = self.world.target.copy()
            if cmd.verb in {"takeoff", "land", "go"}:
                target = self.world.position.copy()
            if cmd.verb == "stop" and active and active.get("verb") in {"takeoff", "land", "go", "forward", "back", "left", "right", "up", "down"} and active.get("device") == "cancelled_local":
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
                # Relative travel is measured from the current position on the
                # commanded axes; orthogonal station-keeping targets are retained.
                if cmd.verb in {"up", "down"}:
                    target[2] = self.world.position[2]
                else:
                    target[:2] = self.world.position[:2]
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

