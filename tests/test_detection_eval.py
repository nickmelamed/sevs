import numpy as np
from sevs.evaluation.detection_eval import evaluate_detection_predictions
from sevs.models.detector import Detection


def test_detection_eval_perfect_case():
    preds = [(
        'img1',
        [Detection(box_xyxy=np.array([0, 0, 10, 10], dtype=float), cls=1, conf=0.9)]
    )]
    gts = [('img1', np.array([[0, 0, 10, 10]], dtype=float), np.array([1], dtype=int))]
    out = evaluate_detection_predictions(preds, gts, iou_thresholds=[0.5])
    assert out.map50 == 1.0