# Fixed H1 run budget

- Seeds: 11, 22, 33; each 1280 actions (384+448+448), fixed final checkpoint.
- Train: 1800-second per-process collection wall guard; batch supervisor 1900 seconds.
- Formal evaluation: 100 validation + 300 sealed + 100 same-case navigation retention + 12 zero-feature ablation + 4 instruction counterfactuals per seed.
- Seed 11 additionally: one fixed validation scene, untrained vs trained (two episodes).
- Diagnostic baselines: rule 100 validation, uniform random 300 sealed.
- Evaluation batch supervisor: 3600 seconds. Failures preserve evidence and stop only owned child PIDs.
- Train/eval process IDs and exit cleanup recorded separately. Viewer is kept open for user testing.
