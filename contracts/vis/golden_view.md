# Golden recorded view contract v1

`schema_version: golden_view/1.0` is a versioned Viewer transition extension.
It is not the P1 `1.0.0` event schema (which fixes physics_dt_s at 0.02).
Golden uses the source simulator's recorded 1/120 s timestep.

- `seq` is a zero-based contiguous decision index. `source_epoch` is the
  original replay SHA256; `source_kind` is `simulation_recorded`.
- `payload.golden` is the unmodified source JSON record. Projection fields
  (`state_after`, `action_probabilities`, `reward_parts`, `readout_snapshot`)
  support the shared Viewer without changing source values.
- `decision_time_s` is the observation/neural/policy/action issue time;
  `sim_time_s` is command completion time. The final neural substep is
  explicitly indexed; unrecorded substeps are not synthesized.
- During playback the world follows recorded physics samples, while the
  decision panel labels pre-action inputs and post-action outcomes. Seeking
  selects the command-completion snapshot for the chosen decision.
- Missing observations, neural fields, anatomical regions, and topology are
  `not_recorded` in the UI. Null probabilities must not become numeric zeros.
- `view_manifest.json` binds checkpoint, graph, mapping, original replay,
  exported events hashes, and source path. Export rejects mismatched source
  replay/checkpoint hashes and never loads a model for inference.
- Registry updates preserve other entries. The bounded events endpoint
  returns `next_offset`; the Viewer loads all pages before playback.

Formal acceptance lives in the Golden handoff. `golden-validate` reports
browser checks as null until a replay-bound browser acceptance record exists.
Absence of evidence must never be reported as a passed browser interaction.
