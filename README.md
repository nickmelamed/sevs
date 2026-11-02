# SEVS (Self-Evaluating Vision System) for object detection reliability

SEVS is a hardware-agnostic perception reliability project that turns an object detector into a self-evaluating vision system in four steps:

1. run a real detector
2. measure calibration and stability under perturbations
3. learn a meta-model that predicts which detections are likely wrong
4. optionally score image-level OOD risk

## What is implemented

- Real detector wrappers:
  - `torchvision` detection backends (`fasterrcnn_resnet50_fpn_v2`, `retinanet_resnet50_fpn_v2`, `ssdlite320_mobilenet_v3_large`)
  - optional `ultralytics` YOLO wrapper if you install `ultralytics`
  - `mock` detector, used only by `smoke-test`
- Real dataset adapters:
  - COCO via `torchvision.datasets.CocoDetection`
  - Pascal VOC via `torchvision.datasets.VOCDetection`
- Real evaluation:
  - mAP@0.50 and mAP@0.50:0.95 in `src/sevs/evaluation/detection_eval.py`, following COCOeval and tested against `pycocotools`
  - calibration metrics
  - meta-error AUROC and AUPRC on held-out images, and abstention coverage and risk
  - OOD summary scores (`max_softmax_prob`, `entropy`, `energy`)

## Quickstart

### Smoke test
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m sevs.cli smoke-test --config configs/eval/eval_default.yaml
```

### Run on COCO
1. Download COCO into `data/external/coco/`.
2. Build a subset manifest:
```bash
python scripts/prepare_coco_subset.py --coco-root data/external/coco --split val2017 --n 200 --out data/splits/coco_val_small.json
```
3. Copy `configs/eval/eval_real_torchvision_coco.yaml` and edit it:
   - point `subset_manifest` at your manifest
   - keep `detector.family: torchvision` or switch to `ultralytics`
4. Run:
```bash
python -m sevs.cli run --config configs/eval/eval_real_torchvision_coco.yaml
```
`run` refuses a config that uses the mock detector or dataset. `smoke-test` always uses the mock and writes to a `*_smoke` directory.

### Run on VOC
Copy `configs/eval/eval_real_torchvision_coco.yaml` and set:
- `dataset.name: voc`
- `dataset.root: data/external/voc`
- `dataset.year: "2007"`
- `dataset.split: test`

## Notes

- mAP is computed in `detection_eval.py`, not by calling `pycocotools`. `pycocotools` is still required to read COCO annotations, and the tests use it as the reference.
- Config files are checked against `src/sevs/utils/config_schema.py`, and unknown keys are an error.
- For laptop inference, `ssdlite320_mobilenet_v3_large` is the lightest torchvision option in this repo.



