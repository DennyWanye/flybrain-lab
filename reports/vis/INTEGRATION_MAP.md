# P1 to Viewer Integration Map

| Contract | Current source | Status |
|---|---|---|
| 8D observation | `flydrone.env.FlyToTargetEnv._observation()` | exported |
| 14 channel encoding | `flydrone.encoder.SensoryEncoder` | retained in trainer; not repeated by viewer |
| action/probability/value | `flydrone.__main__.trace()` and `ActorCritic` | exported for newly generated traces |
| reward parts | `FlyToTargetEnv.step()` info | exported |
| state before/after | trace adapter and environment state | exported |
| readout v/spike/trace | `FrozenReservoir` readout state | exported for selected nodes |
| run/epoch/episode/decision identity | `flydrone.vis.exporter` | generated for recorded view |
| topology | MaleCNS CSR graph | not yet exported; no fabricated graph is shown |
| training window/validation/sealed test | `reports/p1/*/summary.json` | not yet imported into viewer |
| live latest snapshot | observer/writer hook | not yet implemented |

The real view is explicitly `simulation_recorded`; it does not claim to be a
historical training-time recording.
