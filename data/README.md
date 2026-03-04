# Data

This repo assumes datasets are stored locally and are **not** committed.

## Supported (starter)
- COCO (recommended)
- Pascal VOC (small/simple)

### COCO suggested layout
data/external/coco/
  annotations/
  train2017/
  val2017/

You can create a small manifest for fast iteration:
```bash
python scripts/prepare_coco_subset.py --coco-root data/external/coco --split val2017 --n 200 --out data/splits/coco_val_small.json
```

Then set `subset_manifest` in `configs/eval/eval_default.yaml`.
