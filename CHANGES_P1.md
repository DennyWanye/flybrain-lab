# P1 Changes

- Added the offline `flydrone` package without modifying the legacy `flylab.py` implementation.
- Added Gymnasium-based two-dimensional point-mass flight dynamics with five fixed actions, inertia, boundaries, dwell success, and deadline termination.
- Added a signed-to-nonnegative sensory encoder, frozen CSR MaleCNS adapter, 256-dimensional readout features, Actor-Critic policy, GAE, and PPO update.
- Added smoke and P1 JSON configurations plus environment/brain/PPO tests.
- Added startup and progress evidence under `reports/p1/`.
- No real-drone SDK, socket, UDP, or hardware flight code was added.
