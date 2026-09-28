# V1R R1 development

New isolated version. Preserve all V1/J2R modules, checkpoints and reports.
Hypotheses: coarse single-pair neural encoding around the 10cm threshold, STOP-biased dense labels, and insufficient half-step-grid coverage.
Smooth multiscale altitude population encoder (0.08/0.24/0.72m) feeds the same 34 input channels and frozen full MaleCNS; only the 128 downstream neural features reach the actor. No action rule at inference.
New decision and settled-state labels, symmetric threshold-neighbourhood weighting, half-step-grid curriculum. Physics, safety mask, action duration, tolerances, disturbances and acceptance unchanged.
Formal fixed budget per seed: 768 teacher + 1024 + 1024 DAgger actions; 80 epochs per stage; seeds 11/22/33. No checkpoint selection. Final weights frozen together before sealed evaluation.
Ranges: validation213m, sealed214m, pairs215m, boundary216m, zero217m, development218m; training220m+seed*100000+stage*10000+lane*500+episode; smoke230m+same formula. Historical metadata scan found9461 distinct seeds, maximum201199999; all old source seed formulas below210m.
R1 must pass every unchanged gate before R2. Development/smoke is not readiness evidence.
