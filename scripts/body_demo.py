#!/usr/bin/env python3
"""Optional FlyGym 2.1.0 integration smoke: a hand-written turning controller.

Not driven by a learned connectome; not a full brain-body reconstruction.
Uses public FlyGym APIs illustrated in the official turning-controller tutorial.
Reference: https://neuromechfly.org/tutorials/4d_turning_controller/
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seconds", type=float, default=1.)
    p.add_argument("--out", default="runs/body-demo")
    p.add_argument("--no-video", action="store_true")
    a = p.parse_args()
    if a.seconds <= 0:
        raise ValueError("Positive --seconds required.")
    from flygym import Simulation
    from flygym.anatomy import BodySegment, ContactBodiesPreset
    from flygym.compose import FlatGroundWorld
    from flygym_demo.complex_terrain import (
        HybridTurningController, HybridControllerObservation, LocomotionAction,
        PreprogrammedSteps, apply_locomotion_action, make_locomotion_fly)
    from flygym.utils.math import Rotation3D

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fly = make_locomotion_fly(name="body_test", add_adhesion=True, colorize=True)
    camera = None if a.no_video else fly.add_tracking_camera(
        name="body_cam", pos_offset=(-.5, -7.5, 0.),
        rotation=Rotation3D("euler", (1.57, 0., 0.)), fovy=35.)
    world = FlatGroundWorld()
    world.add_fly(fly, [0., 0., .8], Rotation3D("quat", [1, 0, 0, 0]),
                  bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
                  add_ground_contact_sensors=False)
    sim = Simulation(world)
    if camera is not None:
        sim.set_renderer([camera], camera_res=(240, 320), playback_speed=.1, output_fps=25)
    steps = PreprogrammedSteps()
    order = fly.get_actuated_jointdofs_order("position")
    controller = HybridTurningController(timestep=sim.timestep,
        preprogrammed_steps=steps, output_dof_order=order)
    sim.reset(); controller.reset(seed=0)
    apply_locomotion_action(sim, fly.name, LocomotionAction(
        joint_angles=steps.default_pose_by_dof_order(order), adhesion_onoff=np.ones(6, dtype=bool)))
    sim.warmup()
    thorax = fly.get_bodysegs_order().index(BodySegment("c_thorax"))
    positions = []
    ticks = int(a.seconds / sim.timestep)
    if ticks < 1:
        raise ValueError("Duration shorter than one simulation step.")
    for i in range(ticks):
        signal = np.array([1.2, .4]) if i < ticks // 2 else np.array([.4, 1.2])
        observation = HybridControllerObservation.from_sim(sim, fly.name)
        apply_locomotion_action(sim, fly.name, controller.step(signal, observation))
        sim.step_with_profile()
        positions.append(sim.get_body_positions(fly.name)[thorax].copy())
        if camera is not None:
            sim.render_as_needed_with_profile()
    trajectory = np.asarray(positions)
    if not np.isfinite(trajectory).all():
        raise FloatingPointError("Nonfinite body trajectory.")
    np.save(out / "thorax_xyz.npy", trajectory)
    if camera is not None:
        sim.renderer.save_video(out / "turning.mp4")
    report = {"physics_steps": ticks, "timestep": float(sim.timestep),
              "actuated_dofs": len(order), "displacement_mm": (trajectory[-1]-trajectory[0]).tolist(),
              "control": "handwritten left/right drive; NOT trained connectome output"}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2)); sim.print_performance_report()


if __name__ == "__main__":
    main()
