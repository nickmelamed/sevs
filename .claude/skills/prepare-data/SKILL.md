---
name: prepare-data
description: Download COCO or VOC into data/external and rebuild a subset manifest in data/splits. Only the owner starts this.
disable-model-invocation: true
---

Prepare data for the dataset and size the owner named. If either is missing,
ask.

1. Check free disk space. COCO val2017 plus annotations is about 1 GB.
2. Download into `data/external/<dataset>/`. That path is gitignored. 
   `scripts/download_coco.sh` is empty. Ask the owner for the source.
3. Check that the expected folders exist (`annotations/` and the split
   folder for COCO).
4. Build the manifest with
   `python scripts/prepare_coco_subset.py --coco-root data/external/coco
   --split val2017 --n <N> --out data/splits/<name>.json --seed 42`.
5. Show the manifest's image count and first few names. The manifest stores
   an absolute `coco_root`. Flag it if the path is machine specific.
6. `data/splits/` is protected. Ask before overwriting an existing manifest,
   and never commit images.
