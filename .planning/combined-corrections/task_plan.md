# Retain human corrections in expanded public training

User approved combining the 37 public demonstrations with existing approved human corrections.

- [x] Inspect approved recordings, public split, current exporter and GUI.
- [x] Export a separate immutable combined dataset; preserve originals and held-out data.
- [x] Validate trainer, actual data loader, dry-run and GUI selection without starting training.
- [ ] Commit and push only relevant changes; restore fetched artifact and compare hashes.

Keep original normalization, horizon 10, uniform frame sampling and seed 42. Start from release_focus_2000/1999 with 3000 new updates, batch 4, LR 1e-5; relearn approved human actions from data rather than inherit the previous failed mixed checkpoint. User clicks Start Training.

No claim that endpoint approval proves every action expert quality or that recovery generalizes.
