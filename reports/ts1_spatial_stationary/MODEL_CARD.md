# C2U: stationary vertical measurement filtering

The three readout sets are parameter-identical transfers of the formally trained C2Q seeds11/22/33 (5632 real actions each, full MaleCNS). New training actions:0. This is an observation-based phase-manager repair, not new training or a single end-to-end controller.

A20sample measured height mean is used only when the complete window spans no more than2cm and all measured vertical speeds are at most0.08m/s. Otherwise current measured height is used. XY and all velocities stay raw. Raw26 policy inputs and all encoders/weights are unchanged. Final physical-state success, tolerances, continuous2second hold, action set and time limits are unchanged.

Separate rule-development16/16 and12focused software checks passed. Full128 development checks and8new neural development cases must pass before independent formal validation493m/sealed494m/pairs495m/boundary496m/zero497m. Formal rule minimum99/100, model boundaries11/12 and every other original gate remain fixed. No sealed-case tuning.

Real aircraft adapter remains disabled. All previous failed versions remain intact.
