# C2 continuous XYZ / yaw / hold

Prerequisite V1R all gates passed. New modules and directories; historical V1/J2R/V1R remain immutable.
Three trained readouts share the frozen full MaleCNS state engine, with phase-specific encoders. Altitude -> navigation -> heading; measured drift recovery can return to altitude or navigation. No physical reset on handoff; brain reset preserves tick count and the next sample advances real physics by0.1s.
Each phase retains60s budget; new three-phase task explicitly has180s total instead of J1 two-phase120s. Action durations/tolerances unchanged; full XYZ/yaw/speed constraints held simultaneously2s. Vertical speed additionally required.
New ranges: validation243m, sealed244m, instruction245m, boundary246m, zero247m, development248m; formal250m+modelseed*100000+stage*10000+lane*500+episode; smoke260m.
Fixed formal budget per seed:1024 teacher+1536+1536 DAgger actions,60 optimizer epochs per stage. Warm start successful J2R/V1R, every fourth old sample with half weight for rehearsal. All three readouts must change. Three fixed final models frozen together; no heldout model selection.
Acceptance: per seed total>=90%, clean>=90%, other profiles>=85%, collisions<=1%, random advantage>=20pp, instruction8/8, boundary>=11/12, zero0/12, rule>=99/100, physics and clock continuity, actual navigation above and below1m. All failures preserved.

C2R preserves absolute simulator180s safety deadline. Propagate its closure as terminal failure at the last fresh sensor tick; close is idempotent. Failed C2 code/data/logs retained. No sealed C2 results existed. Fresh seeds313m..330m.
