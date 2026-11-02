---
name: prepare-data
description: Download COCO or VOC into data/external and rebuild a subset manifest in data/splits. Only the owner starts this.
disable-model-invocation: true
---

Prepare data for the dataset and size the owner named. If either is missing,
ask.

1. Check free disk space and tell the owner roughly how much the download
   needs before starting. COCO val2017 plus annotations is about 1 GB.
2. Download into `data/external/<dataset>/`. That path is gitignored. The
   repo's `scripts/download_coco.sh` is a placeholder, so ask the owner for
   the download source if it is still empty.
3. Check that the expected folders exist (`annotations/` and the split
   folder for COCO).
4. Build the manifest with
   `python scripts/prepare_coco_subset.py --coco-root data/external/coco
   --split val2017 --n <N> --out data/splits/<name>.json --seed 42`.
5. Show the manifest's image count and first few names. The manifest stores
   an absolute `coco_root`, so tell the owner if that path is machine
   specific.
6. `data/splits/` is protected. Ask before overwriting an existing manifest,
   and never commit images.
