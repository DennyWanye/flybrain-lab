# H1 browser evidence

Browser: Codex in-app browser, existing test tab 3, localhost:8765/tellosim.

- PASS: live heading-s11 observer loads actual MaleCNS graph provenance, 26-channel measured observation, CW/CCW/STOP mask and neural sample view.
- PASS: target direction arrow and measured target/actual/error/tolerance fields visible; DAgger frame showed target 29.3 degrees, measured 29.7 degrees, error -0.4 degrees; command cw 30 in progress. This is training evidence, not formal model acceptance.
- PASS: pause observer held clock at 18.00 seconds / tick 2160 across checks, while training continued. Resume advanced to a new episode at 7.20 seconds / tick 864 with 384 committed training actions.
- PASS: no page JavaScript errors in the inspected log.
- PASS: saved seed-22 sealed case 0 replay loaded with outcome success; play advanced from 0.24 seconds to 10.37 seconds.
- PASS: pause held 10.41 seconds / tick 1249 across independent checks.
- PASS: dragged timeline from 10.41 seconds to 25.24 seconds / tick 3029; measured heading error updated to -48.4 degrees and command cw 30.
- PASS: camera pan changed target from (0,0,1) to (-0.529,-0.265,1.219), without changing replay time 25.24 seconds. Reset restored target (0,0,1).
- PASS: rotate drag changed camera position to (4.372,-9.830,8.089), reset restored (7,-9,7).
- PASS: inspected browser error log remained empty.
- PASS: refreshed H1 summary shows YES and exact formal rates 289/300, 294/300, 297/300, no collision/bounds, navigation retention 94/100, 94/100, 99/100, all exact.
- PASS: only five current shortcuts are shown: three H1 models and seed-11 same-case before/after.
- PASS: trained same-case replay command jump shows ccw 30 at 16.60 seconds. End-key seek reaches 23.00 seconds / tick 2760, STOP, completed/ack_ok, success, 2.00/2s hold, 5.9-degree error.
- PASS: displayed normalized measured yaw -72.3 degrees agrees with recorded truth -72.30115038382638 degrees within 0.05-degree display rounding.
- PASS: final browser JavaScript error log empty. heading-result.png captures actual trained replay at its completed state; tab retained for user.

training-observation.png records the live training view. Recorded heading replay uses the latest measured transition; replay_state_test.cjs verifies newer sensor data is not overwritten by an older completed-option result.
