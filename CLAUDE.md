# SEVS

SEVS (Self-Evaluating Vision System) wraps an object detector and estimates
when its detections are wrong. It measures calibration, probes stability under
perturbations and test-time augmentation, trains a logistic-regression
meta-model to predict incorrect detections, and scores images for OOD risk.
"Good" means every reported number is reproducible from a seeded config and
comes from a real detector on real data.

The full design lives in docs/SPEC.md. Read the relevant section before
changing anything it covers. Section 6 lists known gaps between intent and
code. Current status and next steps are in PROGRESS.md. Design decisions and
their reasons are in docs/DECISIONS.md.

## Non-negotiable rules

1. No test-set leakage. The meta-error predictor trains and evaluates on
   disjoint data, and any calibration fit uses only its calibration split.
2. Every run is seeded from `run.seed` and reproducible from its config.
3. Output from the mock detector or mock dataset never goes into a results
   table, README, or paper number as a real result.
4. mAP, ECE, Brier, and AUROC must match their standard definitions. Do not
   swap or redefine a metric silently, and do not call the native VOC-style
   AP "COCO mAP".
5. A config key that the code reads must do what its name says. Do not leave
   a key in a config that nothing implements without saying so.
6. Never weaken a test, lint rule, or check to make it pass.
7. Ask before changing anything in `.claude/protected-paths`, the evaluation
   design, dependencies, CI, or hooks.
8. If you find a real bug, stop and tell me. Do not fix it unprompted.

## Commands

```bash
python -m sevs.cli smoke-test --config configs/eval/eval_default.yaml
python -m sevs.cli run --config configs/eval/eval_default.yaml
make test         # pytest
make lint         # ruff
make typecheck    # mypy
make agent-check  # the fast checks the Stop hook runs
```

The repo root contains a virtualenv named `sevs/` (ignored by git) whose
`python3` has no pytest, ruff, or mypy. Call tools by explicit path or
activate a different environment.

## Where things live

- `src/sevs/`: library code. `tests/`: tests with synthetic fixtures.
- `configs/`: YAML for datasets, detectors, probes, and evaluation.
- `data/splits/`: committed subset manifests. Images are never committed.
- `results/runs/`: generated outputs, gitignored.
- `docs/SPEC.md`: design spec. `docs/DECISIONS.md`: decision log.
- `.claude/rules/`: path-scoped rules that load when you touch those files.
- `.claude/skills/`: procedures (`/finish-phase`, `/release`, `/run-eval`,
  `/prepare-data`, `/smoke-test`).

## How to work here

- Plan before multi-file changes. Write the plan down and wait for approval
  when the task spans several modules or touches the spec.
- A task is done when the Stop hook's checks pass and you have shown the
  output. Show evidence (commands and results), not claims.
- Commit in small atomic Conventional Commits (`type(scope): subject`). Code
  and its tests go in the same commit.
- You may branch and commit locally. Ask before pushing, opening or merging
  pull requests, tagging, or changing dependencies.
- If you repeat a multi-step procedure, or I give you the same instructions
  twice, propose a skill for it in `.claude/skills/` and wait for approval.
