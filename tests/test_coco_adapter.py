import json

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("pycocotools")
from sevs.datasets.coco import CocoDataset  # noqa: E402


@pytest.fixture
def coco_root(tmp_path):
    (tmp_path / "val2017").mkdir()
    (tmp_path / "annotations").mkdir()
    for name in ("a.jpg", "b.jpg"):
        Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8)).save(tmp_path / "val2017" / name)
    ann = {
        "images": [{"id": 1, "file_name": "a.jpg", "height": 40, "width": 60}, {"id": 2, "file_name": "b.jpg", "height": 40, "width": 60}],
        "categories": [{"id": 1, "name": "x"}, {"id": 2, "name": "y"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [5, 5, 10, 10], "area": 100, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [20, 10, 30, 20], "area": 600, "iscrowd": 1},
        ],
    }
    (tmp_path / "annotations" / "instances_val2017.json").write_text(json.dumps(ann))
    return tmp_path


def test_crowd_regions_are_returned_in_meta_not_as_ground_truth(coco_root):
    ds = CocoDataset({"root": str(coco_root), "split": "val2017"})
    first = next(iter(ds.iter_samples()))
    assert first.gt_boxes_xyxy.tolist() == [[5, 5, 15, 15]]
    assert first.gt_labels.tolist() == [1]
    assert first.meta["crowd_boxes_xyxy"].tolist() == [[20, 10, 50, 30]]
    assert first.meta["crowd_labels"].tolist() == [2]


def test_config_root_wins_over_the_manifest_root(coco_root, tmp_path):
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"images": ["a.jpg"], "coco_root": "/somewhere/else", "split": "val2017"}))
    ds = CocoDataset({"root": str(coco_root), "subset_manifest": str(manifest)})
    assert [s.meta["file_name"] for s in ds.iter_samples()] == ["a.jpg"]
