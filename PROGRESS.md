# Progress

Current phase and what is left. Updated at the end of each phase.

## Now

- [ ] Review and merge the `chore/agent-standards` PR.
- [ ] Decide on the open bugs under Next.

## Next

- [ ] Split the meta-error data by image, not by detection, and use `val_ratio` and `test_ratio` (SPEC 6.2).
- [ ] Map TTA detections back to the original frame before matching them to the reference box.
- [ ] Decide whether `match_detection_to_ground_truth` should consume ground-truth boxes (duplicate detections currently all count as correct).
- [ ] Seed PyTorch in `seed_everything` (SPEC 6.3).
- [ ] Record in the results row whether the detector and dataset were mock (SPEC 6.4).
- [ ] Decide how to report mAP, since the native AP is VOC-style and ignores crowd flags (SPEC 6.5).
- [ ] Implement or remove config keys with no code: temperature scaling, MC dropout, abstention, AUPRC, extra plots, `save_top_k_failures` (SPEC 6.6).
- [ ] Make `smoke-test` and `run` differ, or merge them.
- [ ] Add a real-detector run with the COCO subset and fill in the paper sections. Then wire `check_numbers.py` to the results tables.
- [ ] Run `mutmut` on `src/sevs/utils/` and `src/sevs/evaluation/` to see how well the tests catch bugs.

## Done

- [x] Phase 0: assessment. Retrofit chosen.
- [x] Phase 1: docs/SPEC.md and docs/DECISIONS.md.
- [x] Phase 2: agent tooling v2.0.1, CLAUDE.md, rules, protected paths, skills.
- [x] Phase 3: README style.
- [x] Phase 4: fixed `roc_auc` ties, TTA factors, and the missing geometry helpers. Cleared ruff and mypy. Added tests. The gate runs `make agent-check`.
- [x] Declared runtime and dev dependencies in `pyproject.toml`.

## Open questions for the owner

- The helpers `area` and `match_detection_to_ground_truth` follow an assumed signature, because the original code is gone. Confirm or correct it (SPEC 6.1).
- The repo root holds a 1.1 GB virtualenv named `sevs/`. It is gitignored. Move it outside the repo or rename it to `.venv`?
- TTA numbers from before this branch are not comparable to later ones, since the scale and brightness augmentations changed.
- `requirements.txt` is a full `pip freeze` including an editable git line for this repo. `pyproject.toml` lists no runtime dependencies. Which should be the source of truth?
- `data/splits/*.json` store an absolute `coco_root` from this machine.
- The `release` skill came with the installer and nothing uses it yet.
