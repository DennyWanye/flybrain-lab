# J2R development diagnosis

Only eight J2 validation cases were replayed with model seed 11. No sealed case was consumed for this diagnosis.

The trace reproduces 6 failures and 2 successes. High-confidence wrong decisions occur well inside the position/heading tolerances, including clean case 72 (about 0.064m from goal) and pose case 21 (about 6.3deg heading error). This excludes sensor noise as the sole explanation. It does not establish that smoothing would fix the reservoir representation.

J2 training labels only option boundaries of separate skill episodes and reports about 99% training accuracy. J2R adds dense supervision of actual intermediate neural states and collects training trajectories in the continuous joint task, including skill handoffs and recovery. This addresses a measured generalization gap; success remains to be evaluated.

Old J2 artifacts remain unchanged. New training seeds, development split and sealed split are disjoint. Teachers only annotate/drive training; inference remains learned argmax over frozen MaleCNS features. Physical task deadlines and tolerances are unchanged.
