# Progress

Current phase and what is left. Updated at the end of each phase.

## Now

- [ ] Review and merge the `fix/reliability-and-honesty` pull request.

## Next

- [ ] Run a real detector on a COCO subset with `eval_real_torchvision_coco.yaml` (`/prepare-data`, then `/run-eval`). Nothing has been run on real data yet.
- [ ] Fill in the paper sections and wire `check_numbers.py` to the results tables once there are numbers.
- [ ] Decide what to do with `configs/dataset`, `configs/model`, and `configs/probes`. Nothing loads them (SPEC 6.9).
- [ ] Decide whether `num_samples` should count the unmodified TTA view (SPEC 6.10).
- [ ] Add labeled out-of-distribution data if OOD AUROC is wanted (SPEC 6.11).
- [ ] Rerun the 379 mutants that crashed in mutmut (see below).
- [ ] Add temperature scaling or MC dropout back, with code and tests, if they are wanted (DECISIONS D15).

## Done

- [x] Phases 0 to 5: assessment, spec, agent tooling, style cleanup, tests, first pull request (merged).
- [x] Reliability phase: see the Unreleased section of CHANGELOG.md.
- [x] Mutation testing on `utils/` and `evaluation/`: 902 mutants killed and 13 survived. New tests kill all 13. 379 mutants crashed in mutmut's forked workers and have no verdict.

## Open questions for the owner

- Rename the 1.1 GB `sevs/` virtualenv in the repo root to `.venv`? It is gitignored. This needs a new virtualenv, since they cannot be moved.
- The geometry helpers follow an assumed signature (SPEC 6.1).
- TTA numbers from before this work are not comparable to later ones, since the augmentations and box mapping changed.
- The `release` skill came with the installer and nothing uses it yet.
