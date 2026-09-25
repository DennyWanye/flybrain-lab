# P1-VIS Startup

## Current status

- `flyview` local read-only server: implemented
- fixture view: registered as `demo-fixture`
- real P1 recorded view: registered as `p1-real-s11`
- live telemetry: implemented for the opt-in trainer path; use the registered
  `p1-live-brain-smoke` view for the last completed CUDA snapshot
- real Tello: not implemented and not authorized

## Start

From the project root in WSL:

```bash
PYTHONPATH=. python3 -m flyview serve --project-root . --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765` in Windows Edge or Chrome. The viewer is read-only;
it does not load Torch, call `env.step`, control training, or access hardware.

## Views

- `demo-fixture`: synthetic contract sample; never a training result.
- `p1-real-s11`: a new recorded trace generated from the existing seed-11 P1
  checkpoint on validation case 0. It is a recorded simulation view, not a
  historical training transcript.
- `p1-real-s11-topology-v2`: the same trace with a bounded local MaleCNS
  topology export (maximum 129 nodes and 512 edges).
- `p1-live-smoke`: a short opt-in direct-policy live writer smoke; it verifies
  latest-only file transport, not full brain telemetry.
- `p1-live-brain-smoke`: a short opt-in CUDA brain run with 8D observation,
  five action probabilities, value estimate, 64 readout neurons, and metrics.

## Known gaps

- Live latest-slot telemetry and WebSocket push are implemented for the opt-in
  trainer path; long-running performance acceptance is pending.
- Select a recorded neuron row to request its bounded local topology. The view
  reports `not_in_export` when that node is not part of the exported subgraph.
- The metrics panel imports read-only training windows, validation summaries,
  and the sealed test count. It never runs evaluation from the browser.
- The legacy trace does not contain physical per-tick paths; the viewer labels
  this as unavailable instead of interpolating reward or neural events.
