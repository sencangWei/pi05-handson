Experimental π0.5 LoRA corrective fine-tune (1000 additional optimizer updates).

Parent: ui_correct_1792406592268062907/1999. Candidate: ui_correct_1792434450272918886/999.
Dataset:37 public demonstrations plus10 reviewed human trajectories/clips,6126 frames total. New interventions:4 recordings,680 frames. Batch4,learning rate5e-6,seed42,horizon10,uniform sampling.

This release is an inference backup. It contains only LoRA tensors and normalization; restore requires the original official π0.5 JAX base. Adam state and training data are retained locally, not included here. Paired development evaluation is complete: fresh initial-layout controls passed4/4(candidate) vs2/4(parent); saved mid-trajectory takeover diagnostics passed0/8 for both. These small development sets and controller-reconstruction limitations do not establish independent generalization or production promotion. User reports fresh-start GUI tests succeeded. No additional weighted training was launched. Downloaded adapter restoration verified all71 full parameter tensor hashes and normalization.

Use scripts/restore_adapter.py and the official base to reconstruct params. See evidence/intervention-finetune-20261020 for the immutable case plan, source checksums and results.
