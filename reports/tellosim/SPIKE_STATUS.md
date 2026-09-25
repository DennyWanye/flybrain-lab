# TS1 Spike Status

Date: 2026-09-25

## Passed

- Existing P1 environment remains dependency-clean after installing only
  `mujoco==3.3.7`; Torch CUDA remains available.
- SDK codec accepts the approved command subset and canonicalizes repeated
  ordinary spaces. It rejects `forward 15`, `go 20 0 0 20`, and `rc` with
  explicit error classes.
- Operation state separates device execution from client acknowledgement. A
  completed movement with a lost reply becomes `unknown_execution`; a new
  ordinary movement is blocked, while protective `stop` remains available.
- Headless MuJoCo rigid-body smoke runs at 120Hz for 600 ticks. The bounded
  body-axis-thrust surrogate reaches x=0.756m from x=0 toward a 0.8m target,
  while z remains 0.9998m near the 1m takeoff height.

Run command:

```bash
PYTHONPATH=. /home/denny/projects/flybrain-env/bin/python -m flydrone.tellosim spike \
  --world-config configs/tellosim/world_room6.json \
  --out reports/tellosim/spike.json
```

## Not proven by this spike

- Tello hardware compatibility or real UDP safety.
- A calibrated Tello flight model; the controller is explicitly an engineering
  surrogate.
- Takeoff/land command lifecycle, collision geometry, obstacles, PoseProvider,
  TS1 observation encoding, PPO option returns, or browser TS1 data.
- C0 learning or any formal success threshold.
