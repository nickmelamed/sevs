"""Check the native mAP against pycocotools on random detections."""
import numpy as np
import pytest

from sevs.evaluation.detection_eval import evaluate_detection_predictions
from sevs.models.detector import Detection

pytest.importorskip("pycocotools")
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402


def random_scene(seed, n_images=12, n_classes=3):
    rng = np.random.default_rng(seed)
    gts, crowd, preds = [], {}, []
    for i in range(n_images):
        iid = f"{i}"
        n_gt = int(rng.integers(0, 5))
        boxes, labels = [], []
        for _ in range(n_gt):
            x, y = rng.uniform(0, 200, 2)
            w, h = rng.uniform(10, 80, 2)
            boxes.append([x, y, x + w, y + h])
            labels.append(int(rng.integers(1, n_classes + 1)))
        gts.append((iid, np.array(boxes).reshape(-1, 4), np.array(labels, dtype=int)))
        if rng.random() < 0.4:
            x, y = rng.uniform(0, 200, 2)
            w, h = rng.uniform(60, 150, 2)
            crowd[iid] = (np.array([[x, y, x + w, y + h]]), np.array([int(rng.integers(1, n_classes + 1))]))
        dets = []
        for b, c in zip(boxes, labels):
            if rng.random() < 0.8:
                box = np.array(b) + rng.normal(0, 4, 4)
                box[2] = max(box[2], box[0] + 1)
                box[3] = max(box[3], box[1] + 1)
                dets.append(Detection(box, c, float(rng.uniform(0.2, 1.0))))
        for _ in range(int(rng.integers(0, 4))):
            x, y = rng.uniform(0, 200, 2)
            w, h = rng.uniform(10, 80, 2)
            dets.append(Detection(np.array([x, y, x + w, y + h]), int(rng.integers(1, n_classes + 1)), float(rng.uniform(0.05, 0.9))))
        preds.append((iid, dets))
    return gts, crowd, preds


def coco_stats(gts, crowd, preds, n_classes=3):
    images, anns, dts = [], [], []
    aid = 1
    for iid, boxes, labels in gts:
        images.append({"id": int(iid), "width": 400, "height": 400})
        for b, c in zip(boxes, labels):
            w, h = b[2] - b[0], b[3] - b[1]
            anns.append({"id": aid, "image_id": int(iid), "category_id": int(c), "bbox": [b[0], b[1], w, h], "area": w * h, "iscrowd": 0})
            aid += 1
        if iid in crowd:
            cb, cl = crowd[iid]
            w, h = cb[0][2] - cb[0][0], cb[0][3] - cb[0][1]
            anns.append({"id": aid, "image_id": int(iid), "category_id": int(cl[0]), "bbox": [cb[0][0], cb[0][1], w, h], "area": w * h, "iscrowd": 1})
            aid += 1
    for iid, dets in preds:
        for d in dets:
            b = d.box_xyxy
            dts.append({"image_id": int(iid), "category_id": d.cls, "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": d.conf})
    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns, "categories": [{"id": c} for c in range(1, n_classes + 1)]}
    gt.createIndex()
    dt = gt.loadRes(dts)
    ev = COCOeval(gt, dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return ev.stats[0], ev.stats[1]


@pytest.mark.parametrize("seed", range(8))
def test_matches_cocoeval(seed):
    gts, crowd, preds = random_scene(seed)
    ref_map, ref_map50 = coco_stats(gts, crowd, preds)
    out = evaluate_detection_predictions(preds, gts, crowd=crowd)
    assert out.map50_95 == pytest.approx(ref_map, abs=1e-9)
    assert out.map50 == pytest.approx(ref_map50, abs=1e-9)


def rounded(preds, digits):
    return [(i, [Detection(d.box_xyxy, d.cls, round(d.conf, digits)) for d in ds]) for i, ds in preds]


@pytest.mark.parametrize("seed", range(10))
def test_matches_cocoeval_with_tied_scores_and_many_images(seed):
    gts, crowd, preds = random_scene(seed, n_images=30)
    preds = rounded(preds, 1)
    ref_map, ref_map50 = coco_stats(gts, crowd, preds)
    out = evaluate_detection_predictions(preds, gts, crowd=crowd)
    assert out.map50_95 == pytest.approx(ref_map, abs=1e-9)
    assert out.map50 == pytest.approx(ref_map50, abs=1e-9)


@pytest.mark.parametrize("seed", range(10))
def test_matches_cocoeval_with_tied_scores_and_few_images(seed):
    gts, crowd, preds = random_scene(seed, n_images=9)
    preds = rounded(preds, 1)
    ref_map, ref_map50 = coco_stats(gts, crowd, preds)
    out = evaluate_detection_predictions(preds, gts, crowd=crowd)
    assert out.map50_95 == pytest.approx(ref_map, abs=1e-9)
    assert out.map50 == pytest.approx(ref_map50, abs=1e-9)


@pytest.mark.parametrize("seed", range(5))
def test_matches_cocoeval_when_an_image_has_more_than_100_detections_of_a_class(seed):
    rng = np.random.default_rng(seed)
    gts = [("0", np.array([[10.0, 10.0, 60.0, 60.0], [100.0, 100.0, 150.0, 150.0]]), np.array([1, 1]))]
    dets = [Detection(np.array([10.0, 10.0, 60.0, 60.0]) + rng.normal(0, 2, 4), 1, round(float(rng.uniform(0.1, 0.5)), 2)) for _ in range(120)]
    dets.append(Detection(np.array([100.0, 100.0, 150.0, 150.0]), 1, 0.05))
    preds = [("0", dets)]
    ref_map, ref_map50 = coco_stats(gts, {}, preds, n_classes=1)
    out = evaluate_detection_predictions(preds, gts)
    assert out.map50_95 == pytest.approx(ref_map, abs=1e-9)
    assert out.map50 == pytest.approx(ref_map50, abs=1e-9)
