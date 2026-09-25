# TS1 Integration Map

## Current baseline

| Plan contract | Current implementation | Status |
|---|---|---|
| Frozen MaleCNS reservoir | `flydrone/brain.py` | Reuse after TS1 observation/clock contract is implemented |
| Existing PPO | `flydrone/ppo.py` | Reuse math only; option-duration discounting is still required |
| Existing read-only Viewer | `flyview/server.py`, `flyview/static/index.html` | Preserve for P1; add a versioned TS1 source later |
| SDK parser | `flydrone/tellosim/sdk/codec.py` | Spike implemented |
| Operation state | `flydrone/tellosim/sdk/operation.py` | Spike implemented; no transport yet |
| MuJoCo world | `flydrone/tellosim/physics/world.py` | Spike implemented; controller is not Tello-calibrated |
| TS1 observation/encoder | None | Not implemented |
| TS1 PPO runner/checkpoint | None | Not implemented |
| Three.js TelloSim page | None | Not implemented |

## Boundary

The TS1 spike uses the existing P0/P1 Python environment and adds only
`mujoco==3.3.7`. It does not import the old 5-action checkpoint, open a real
UDP socket, or claim hardware compatibility. The first physics controller is a
bounded body-axis-thrust surrogate and is only evidence that continuous rigid
body integration is viable.
