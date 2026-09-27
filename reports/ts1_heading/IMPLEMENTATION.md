# H1: specified heading and stable hold

Task: in-place heading control in the rigid-body simulator; random initial yaw and target yaw, at least 35 degrees apart. SDK action subset STOP/CW30/CCW30. Goal tolerance 16 degrees because the command quantization is 30 degrees. Success also requires position/height/linear speed and yaw rate constraints continuously for 2 seconds within 60 seconds.

A separate learned readout shares the same frozen real MaleCNS connectome. Existing C1 navigation weights and contracts stay unchanged. External task selection chooses the skill; no rule chooses deployed turn direction. Channels 6/7 explicitly become measured target-error sine/cosine in a separately versioned observation contract. No raw observation bypass to the policy. Heading training uses demonstrations plus DAgger; no PPO or trained critic claim. Stage warm starts/inference only; exact mid-rollout resume not implemented for H1.

Per seed 11/22/33: 384 teacher actions + two 448-action DAgger rounds. Four independent simulator/neural lanes. Final checkpoints locked before sealed evaluations; 100 validation + 300 sealed cases per seed, uniform-random 300-case reference. Gates: every seed >=90%, collision/bounds <=1%, improvement >=20 percentage points, both CW/CCW used. Same 100 navigation cases must exactly match prior C1 results with untouched original navigation weights. Four preselected instruction counterfactuals, 12-case zero-neural-feature ablation, and seed-11 first-validation-case untrained/trained replay comparison are diagnostics.

This is an independent heading skill, not joint position-and-heading navigation, full TS1 readiness or hardware readiness. Training/testing must report failures without retuning on sealed results.
