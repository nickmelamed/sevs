---
name: run-eval
description: Run the SEVS pipeline on a named config with a real detector and report the results row. Only the owner starts this.
disable-model-invocation: true
---

Run an evaluation for the config the owner named, such as
`configs/eval/eval_real_torchvision_coco.yaml`. If none was given, ask.

1. Read the config. Stop and say so if `detector.family` or `dataset.name` is
   `mock`, since mock output must never be reported as a result.
2. Confirm the dataset root and any `subset_manifest` exist. If not, point the
   owner to `/prepare-data` and stop.
3. Note `run.seed` and `run.output_dir`. Do not overwrite an existing output
   directory without asking.
4. Run `python -m sevs.cli run --config <config>` with an interpreter that has
   the dependencies installed. The `sevs/` virtualenv has the pinned ones.
5. Show the printed results row and list the files written to the output
   directory.
6. Report `n_images` and `n_detections`, and say which metrics are NaN and
   why. A NaN meta-error AUROC usually means every detection had the same
   label.
7. Do not copy numbers into the README or paper. The owner decides that.
