#!/usr/bin/env python3
"""Optional FlyGym/MuJoCo-Warp GPU availability and parallel physics smoke.

Neutral posture only. This is not locomotion learning or brain-controlled behavior.
Reference: https://neuromechfly.org/tutorials/3_gpu_accelerated_simulation/
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worlds", type=int, default=4)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--out", default="reports/body-gpu-smoke.json")
    a = p.parse_args()
    if min(a.worlds, a.steps) < 1:
        raise ValueError("Positive worlds/steps required.")
    import warp as wp
    from flygym.warp import GPUSimulation
    from flygym.warp.utils import check_gpu
    from flygym.compose import NeuroMechFly, ActuatorType, FlatGroundWorld, KinematicPosePreset
    from flygym.anatomy import Skeleton, JointPreset, ActuatedDOFPreset, AxisOrder
    from flygym.utils.math import Rotation3D
    check_gpu()
    if not wp.is_cuda_available():
        raise RuntimeError("Warp CUDA unavailable; not falling back to CPU.")
    fly = NeuroMechFly()
    skeleton = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL, joint_preset=JointPreset.LEGS_ONLY)
    fly.add_joints(skeleton, neutral_pose=KinematicPosePreset.NEUTRAL)
    dofs = fly.skeleton.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
    fly.add_actuators(dofs, actuator_type=ActuatorType.POSITION, kp=50.,
                      neutral_input=KinematicPosePreset.NEUTRAL)
    fly.add_leg_adhesion()
    world = FlatGroundWorld()
    world.add_fly(fly, (0., 0., .8), Rotation3D("quat", (1, 0, 0, 0)))
    sim = GPUSimulation(world, a.worlds)
    sim.set_leg_adhesion_states(fly.name, np.ones((a.worlds, 6), dtype=np.float32))
    sim.warmup(); wp.synchronize()
    start = time.perf_counter()
    for _ in range(a.steps):
        sim.step_with_profile()
    wp.synchronize()
    elapsed = time.perf_counter()-start
    result = {"worlds": a.worlds, "batch_physics_steps": a.steps, "seconds": elapsed,
              "aggregate_world_steps_per_second": a.worlds*a.steps/elapsed,
              "scope": "execution smoke only, neutral control, no rendering, no learning"}
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2)); sim.print_performance_report()


if __name__ == "__main__":
    main()
