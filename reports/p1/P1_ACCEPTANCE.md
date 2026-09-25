# P1 FlyToTarget2D Acceptance

## Status

P0_ENVIRONMENT_READY = YES
P1_SOFTWARE_READY = YES
P1_SIM_TASK_LEARNED = NO
P1_REAL_FLIGHT_READY = NO

## Executed scope

- Offline Gymnasium `FlyToTarget2D-v1`; no Tello, UDP, takeoff, landing, or motor code.
- Frozen full MaleCNS CSR graph: `N=166700`, `nnz=25582938`.
- Real graph CUDA probe: PASS, feature dimension 256, finite output.
- Rule reference validation: 100/100 success.
- Brain seed 11, 131072 environment transitions, `envs=16`: 86/100 validation success, 0 boundary failures, 14 deadline endings.
- Brain seed 22, 131072 environment transitions, `envs=16`: 93/100 validation success, 0 boundary failures, 7 deadline endings.
- Brain seed 33, 131072 environment transitions, `envs=16`: 71/100 validation success, 0 boundary failures, 29 deadline endings.
- Independent formal seed median: 86%; per-seed minimum: 71%; formal learning gate: FAIL.

## Tests

- Legacy and P1 tests: 22 passed.
- Smoke probe, train, checkpoint load, trace, replay: PASS.
- Formal three-seed learning gate: FAIL; seed 33 is below the per-seed 80% threshold and the median is below 90%.
- Formal 500-case test set: kept sealed; not used for tuning or checkpoint selection.

## Known gaps

- Exact resume restoration of environment/RNG/brain state is pending.
- `P1_SIM_TASK_LEARNED` is `NO`; the three independent `envs=16` seeds did not meet the stated threshold.

## Boundary

This report covers a simulation-only software milestone. It does not authorize or validate real flight.
