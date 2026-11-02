---
name: smoke-test
description: Run the mock end-to-end pipeline and report whether it completes.
---

1. Run `python -m sevs.cli smoke-test --config configs/eval/eval_default.yaml`.
   Use an interpreter with the dependencies installed.
2. Show the printed results row and any traceback in full.
3. State that the output comes from the mock detector and is not a result.
4. If it fails, report the error and stop. Do not fix the cause unless the
   owner asks.
