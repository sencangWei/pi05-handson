# Select useful continuous human correction segments

User corrected full-trajectory inclusion: raw recordings stay intact, but novice wandering and redundant commands should be excluded from training.

- [x] Inspect both camera timelines with commands and physics outcomes; select explicit contiguous ranges.
- [x] Produce candidate two-camera clips and open review page; user explicitly wants to check before deciding.
- [x] Implement and test optional curated export/validation without exporting the real candidate dataset.
- [ ] After user's clip feedback: export curated mixture, actual loader/dry-run and GUI selection.
- [ ] Back up scoped code/manifests to writable remote and restore for hash comparison.

No training launch, no model-job cancellation, no original edits. Retain necessary settling and context; selected grasp/placement behavior does not prove generalized recovery.
