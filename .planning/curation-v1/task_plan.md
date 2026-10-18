# Select useful continuous human correction segments

User corrected full-trajectory inclusion: raw recordings stay intact, but novice wandering and redundant commands should be excluded from training.

- [x] Inspect both camera timelines with commands and physics outcomes; select explicit contiguous ranges.
- [x] Produce candidate two-camera clips and open review page; user explicitly wants to check before deciding.
- [x] Implement and test optional curated export/validation without exporting the real candidate dataset.
- [x] User confirmed both placement clips; exported and strictly validated separate placement-only dataset39/4746.
- [x] User confirmed recovery candidates; merged with partial-phase gates, verified actual loader/dry-run, and selected41/4881 version in GUI.
- [x] Back up new v2 capability, approved selection, actual dataset manifest and checks; restore remote hashes.
- [x] Back up scoped code/manifests to writable remote and restore for hash comparison.

No training launch, no model-job cancellation, no original edits. Retain necessary settling and context; selected grasp/placement behavior does not prove generalized recovery.
