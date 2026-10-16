"""Candidate selection must use validation, verify provenance and preserve weights."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import torch
from handgesture.models import build_model, image_tensor, GesturePredictor
from handgesture.training import choose_cnn


def test_architectures_and_mobilenet_require_explicit_weights(tmp_path):
    torch.set_num_threads(1)
    for name,size in [('gap',128),('spatial',128),('mobilenet_v3_small',224)]:
        model=build_model(name,initialize_pretrained=False);model.eval()
        with torch.inference_mode():assert model(torch.zeros(2,3,size,size)).shape==(2,6)
        t=image_tensor(np.full((25,25,3),255,np.uint8),name)
        assert t.shape==(3,size,size)
    with pytest.raises(ValueError,match='weights'):
        build_model('mobilenet_v3_small',initialize_pretrained=True)


def test_candidate_choice_is_validation_only_and_rejects_mixed_protocols():
    rows=[{'model':'cnn','directory':'a','validation_metrics':{'macro_f1':.8},'validation_loss':.4,
           'protocol':{'manifest_sha256':'same','preprocessing_fingerprint':'fp','smoke_test':False}},
          {'model':'cnn','directory':'b','validation_metrics':{'macro_f1':.8},'validation_loss':.3,
           'protocol':{'manifest_sha256':'same','preprocessing_fingerprint':'fp','smoke_test':False}}]
    assert choose_cnn(rows)['directory']=='b'
    rows[1]['protocol']['preprocessing_fingerprint']='other'
    with pytest.raises(ValueError,match='protocol'):choose_cnn(rows)


def test_finalization_requires_all_three_distinct_cnn_architectures(tmp_path):
    from handgesture.training import require_cnn_candidates
    rows=[{'model':'cnn','config':{'architecture':'gap'},'directory':'a'},
          {'model':'cnn','config':{'architecture':'gap'},'directory':'a'},
          {'model':'cnn','config':{'architecture':'spatial'},'directory':'c'}]
    with pytest.raises(ValueError,match='three'):require_cnn_candidates(rows)
