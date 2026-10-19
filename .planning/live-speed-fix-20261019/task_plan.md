# Live speed fix

Target: reduce GUI overhead without changing VLA action size, replan1, model or old source/data provenance; pause after physical stable completion. Success: no rerenders during unchanged pending/idle; whole-cycle pacing; precise timing; pause/reset/manual regression gates; real seeded same-state action/trajectory comparison and measured scheduler speed; upgraded paused GUI; verified remote backup.

1. Preserve simulation_lab.py SHA (strict teacher dependency). Add responsive subclass and switch GUI workbench inheritance only.
2. Red/green tests for idle/pending cameras, stale native edits/pause, cycle cadence, timing, stable60 completion; existing simulation/workbench/panel/correction checks.
3. Isolated port8756 same-model seed0 benchmark on saved scene2, warmup before both arms, fixed400steps; do not touch live8755 RNG. Realphysics/rendering, manual poll runner excludes nativeGUI/compositor; report that limit.
4. Save current owned GUI only if completed/not recording; reopen updated panel paused. Preserve records/old model/data.
5. Commit/push task-only code/evidence, fetch and reconstruct clean remote diff, compare tree/file hashes.

No training or action scaling. Model tiny-action behavior remains a separate limitation. Failed native subagent routes unsupported by account; execute directly.

Verified:121tests+5subtests, oldteacher8sourcepins intact; two400step loop studies; rawactions exact, physics max4.5e-11(notbitexact), ~37% loop-only wall reduction; actualGUI20steps~8.1Hz. Fresh final-gate realphysics synthetic60steps PASS; new GUI3638666 reset0paused,lastuserlayout restored. Same-day remote backup verification remains.
