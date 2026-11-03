import subprocess
import sys

import pytest


@pytest.mark.parametrize("name, cls", [("coco", "CocoDataset"), ("voc", "VocDataset")])
def test_known_datasets_resolve_in_a_fresh_interpreter(name, cls):
    code = f"from sevs.datasets.registry import get_dataset; print(get_dataset({name!r}).__name__)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == cls


def test_unknown_dataset_names_the_available_ones():
    from sevs.datasets.registry import get_dataset

    with pytest.raises(KeyError, match="nope"):
        get_dataset("nope")
