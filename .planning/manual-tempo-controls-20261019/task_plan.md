# Manual recording tempo controls

Goal: two explicit translation presets for normal transport versus precision; rotation remains precision; verify actual speed before user's new recordings. Existing approved records/checkpoints remain intact. No training, hidden policy action scaling or robot hardware.

1. Trace old .1 default/.3 max and recording label path; snapshot metadata/sourcepins.
2. Isolated realphysics calibration, identical saved physical states/controllers: .1/.3/.6 normalized input, six free directions and five carry directions across two saved holding states;10steps/branch. Record actual displacement and possible slips. Select normal preset only after inspection.
3. Red tests for normal/fine UI modes, stop held joystick before switching, immutable raw labels and unchanged rotation/gripper controls; implement bounded translation range.
4. Targeted tests/Ruff, true GUI manual-recording smoke into separate scratch root; open updated GUI with last saved layout paused, no policy motion. If active recording/run appears, preserve it and do not interrupt.
5. Document limitations, preserve historical source hash checks, commit/push task-only artifacts and verify remote diff restore tree/bytes.
