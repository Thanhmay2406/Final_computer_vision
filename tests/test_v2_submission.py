"""Kaggle papermill must be able to select the serialized notebook kernel."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import nbformat
from scripts.submit_kaggle_v2 import build_submission


def test_submission_includes_kernel_and_is_private(tmp_path):
    import json
    build_submission(tmp_path,'thanhmay2406/test-v2',smoke=True)
    notebook=nbformat.read(tmp_path/'experiment.ipynb',4)
    assert notebook.metadata.kernelspec.name=='python3'
    compile(next(c.source for c in notebook.cells if c.cell_type=='code'),'cloud-wrapper','exec')
    meta=json.loads((tmp_path/'kernel-metadata.json').read_text())
    assert meta['is_private'] is True and meta['enable_gpu'] is True


def test_fetch_recognizes_sdk_int_enum_completion():
    from enum import IntEnum
    from types import SimpleNamespace
    from scripts.fetch_kaggle_v2 import kernel_state
    class Status(IntEnum):COMPLETE=2
    assert kernel_state(SimpleNamespace(status=Status.COMPLETE))=='COMPLETE'
