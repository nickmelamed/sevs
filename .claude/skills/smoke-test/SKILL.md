---
name: smoke-test
description: Run the mock end-to-end pipeline and report whether it completes.
---

Run the pipeline on synthetic data to check it still works end to end.

1. Run `python -m sevs.cli smoke-test --config configs/eval/eval_default.yaml`.
   Use an interpreter with the dependencies installed.
2. Show the printed results row and any traceback in full.
3. Say plainly that the output comes from the mock detector. Never present
   these numbers as results.
4. If it fails, report the error and stop. Do not fix the cause unless the
   owner asks.
