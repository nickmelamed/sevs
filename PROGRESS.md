# Progress

Current phase and what is left. Updated at the end of each phase.

## Now

- [ ] Review and merge the `fix/reliability-and-honesty` pull request.

## Next

- [ ] Run the other detectors (Faster R-CNN, RetinaNet) and VOC. Only SSDlite320 on 200 COCO val images has been run.
- [ ] Run SSDlite320 on all 5,000 val images to compare with its published 21.3 mAP. The 200-image subset gave 28.96.
- [ ] `ood_energy_mean` is about 0 for every detection from torchvision and Ultralytics, because the pseudo-logits are `log([p, 1 - p])`, whose log-sum-exp is 0. Decide how to score energy without logits.
- [ ] 95.6% of detections at `conf_threshold: 0.05` are incorrect, so meta-error AUPRC (0.992) is close to its base rate. Report the incorrect share beside AUPRC, or raise the threshold.
- [ ] Fill in the paper sections and wire `check_numbers.py` to the results tables once there are numbers.
- [ ] Decide whether `num_samples` should count the unmodified TTA view (SPEC 6.9).
- [ ] Add labeled out-of-distribution data if OOD AUROC is wanted (SPEC 6.10).
- [ ] Rerun the 379 mutants that crashed in mutmut (see below).
- [ ] Add temperature scaling or MC dropout back, with code and tests, if they are wanted (DECISIONS D15).

## Done

- [x] Phases 0 to 5: assessment, spec, agent tooling, style cleanup, tests, first pull request (merged).
- [x] Reliability phase: see the Unreleased section of CHANGELOG.md.
- [x] First real run: SSDlite320 on 200 COCO val images (seed 42). mAP50 0.417 and mAP50_95 0.290, identical to pycocotools on the same predictions. Meta-error test AUROC 0.902 on 30 test images. Results are in `results/runs/sevs_coco_torchvision/` (gitignored).
- [x] The dataset registry imported no adapter, so no real run could start. Fixed.
- [x] Mutation testing on `utils/` and `evaluation/`: 902 mutants killed and 13 survived. New tests kill all 13. 379 mutants crashed in mutmut's forked workers and have no verdict.

## Open questions for the owner

- The geometry helpers follow an assumed signature (SPEC 6.1).
- TTA numbers from before this work are not comparable to later ones, since the augmentations and box mapping changed.
- The `release` skill came with the installer and nothing uses it yet.
