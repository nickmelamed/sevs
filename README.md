# SEVS — Self-Evaluating Vision System (Object Detection Reliability)

SEVS is a **hardware-agnostic** research-grade project that turns an object detector into a **self-evaluating perception system**:

1) runs a base object detector  
2) estimates uncertainty and calibration quality  
3) probes **consistency under perturbations**  
4) trains a **meta error predictor** to estimate when detections are likely wrong

This repo is intentionally **lightweight** and **modular** so you can iterate quickly on a laptop.

---

## Quickstart

### 1) Create env + install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2) Run a smoke test (no dataset required)
```bash
python -m sevs.cli smoke-test --config configs/eval/eval_default.yaml
```

### 3) Run evaluation (requires dataset prep)
1) Download / prepare COCO (or VOC) as described in `data/README.md`.
2) Create a subset manifest (example script provided).
3) Run:
```bash
python -m sevs.cli run --config configs/eval/eval_default.yaml
```

Outputs land in `results/runs/<run.name>/`.

---

## What’s implemented in this starter repo

✅ Configuration-driven runs (YAML)  
✅ Perturbation framework + matching scaffolding  
✅ Reliability metrics helpers (ECE, Brier, AUROC helpers)  
✅ Report generator scaffold (tables + plots placeholders)  
✅ Paper outline in `reports/paper/sevs_paper.md`  

This starter repo does **not** ship a detector training pipeline by default. Instead, it provides:
- a clean interface (`src/sevs/models/detector.py`) to plug in YOLO/DETR/etc.
- a runnable end-to-end skeleton with mock detections so you can validate the plumbing

---

## Repo layout

See the full tree in the prompt response that generated this repo; the structure is mirrored here.

---

## Suggested next steps

1) Plug in a detector wrapper (e.g., Ultralytics YOLOv8) inside `src/sevs/models/detector.py`
2) Implement dataset adapters for COCO/VOC in `src/sevs/datasets/`
3) Turn on real matching + compute:
   - mAP
   - ECE + reliability diagram
   - error detection AUROC for meta model
4) Add an ablation runner and save a canonical results table in `results/artifacts/`

---

## License

MIT
