# P1 Startup

## Scope

P1 implementation has started in the existing FlyBrainLab project. The scope is
the offline `FlyToTarget2D-v1` simulation and a frozen MaleCNS reservoir. Real
Tello communication, UDP, takeoff, landing, and motor control are excluded.

## Baseline

- P0 report: `reports/ENVIRONMENT_ACCEPTANCE.md`, `P0_ENVIRONMENT_READY = YES`
- Python: 3.12.3 in `/home/denny/projects/flybrain-env`
- PyTorch: 2.14.0+cu130
- GPU: NVIDIA GeForce RTX 5070 Ti, CUDA CSR SpMM PASS
- MaleCNS graph: `N=166700`, `nnz=25582938`
- Legacy tests before P1: 13 passed
- Gymnasium: 1.3.0

## Implemented and verified

- `flydrone.contracts`, `dynamics`, `env`, and `encoder`
- `flydrone.brain.FrozenReservoir` with CSR graph, explicit mappings, independent batch state, and read-only features
- `flydrone.policy.ActorCritic` and PPO clipped update/GAE
- smoke config, case generation, reference/random baseline, probe, train, and eval paths
- atomic checkpoint helper and P1 unit tests
- environment tests and old tests: 22 passed
- synthetic probe/train/eval smoke completed
- full MaleCNS CUDA probe completed at batch 1 and batch 4; feature dimension 256, finite output
- reference validation: 100/100 success
- brain seed=11 at 131072 transitions with `envs=16`: 86/100 validation success, 0 boundary failures, 14 deadline endings
- brain seed=22 at 131072 transitions with `envs=16`: 93/100 validation success, 0 boundary failures, 7 deadline endings
- brain seed=33 at 131072 transitions with `envs=16`: 71/100 validation success, 0 boundary failures, 29 deadline endings
- formal three-seed result: median 86%, minimum 71%; learning gate not met

## Not yet complete

- resume command and exact environment/RNG restoration
- formal 500-case test report; the test set remains sealed because the independent three-seed gate is not met

This file is a progress record. The software and simulation harness are ready, but the learning gate is not passed.
