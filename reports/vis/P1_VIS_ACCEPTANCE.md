# P1-VIS Acceptance Progress

| Gate | Status | Evidence |
|---|---|---|
| V0 contract/source boundary | PASS | bundled JSON contracts and 26 negative cases pass |
| V1 fixture UI | PASS | `demo-fixture`, browser page and replay controls |
| V2 real recorded replay | PASS | `p1-real-s11`, 60 recorded decisions |
| V3 live summary | PARTIAL | opt-in CUDA brain latest snapshot smoke passed; WebSocket push and long-run acceptance pending |
| V4 local topology | PASS | `p1-real-s11-topology-v2`, bounded CSR export (129 nodes / 70 retained edges), center-node selection API/UI |
| V5 performance/security | PARTIAL | local host allowlist, read-only methods, and WS origin checks implemented; full budget not run |

Additional completed viewer work:

- Neuron rows are selectable; the selected ID is sent as `center_id` and the
  response contains only its exported upstream/downstream edges.
- Metrics API imports read-only training windows, validation summaries, and the
  sealed test-set count without running evaluation. Current inventory is 16
  training windows, 11 validation summaries, and a sealed 500-case test split.

The page never presents fixture data as training data. Missing probabilities,
physics paths, metadata, or topology are shown as unavailable. The remaining
partial gates require long-running performance/multi-client evidence and a
manual Windows browser acceptance pass; they are not inferred from this smoke.
