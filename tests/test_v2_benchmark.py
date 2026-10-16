"""Deployment selection needs a completed comparable benchmark and real hand frames."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from handgesture.benchmark import select_deployment


def test_deployment_uses_validation_among_models_that_meet_real_time_target():
    quality={'rf':.93,'mlp':.94,'cnn':.96}
    rows=[dict(model='rf',fps=20,complete=True,hand_frames=10,median_ms=50),
          dict(model='mlp',fps=30,complete=True,hand_frames=10,median_ms=33),
          dict(model='cnn',fps=10,complete=True,hand_frames=10,median_ms=100)]
    assert select_deployment(rows,quality)['model']=='mlp'
    rows[1]['hand_frames']=0
    assert select_deployment(rows,quality)['model']=='rf'
    rows[0]['complete']=False
    assert select_deployment(rows,quality)['model'] is None


def test_mostly_empty_video_cannot_qualify_fast_no_hand_path():
    rows=[dict(model='cnn',fps=100,complete=True,hand_frames=1,measured_frames=300,median_ms=10)]
    assert select_deployment(rows,{'cnn':.99})['model'] is None


def test_deployment_rejects_checkpoint_different_from_benchmarked(tmp_path):
    from handgesture.benchmark import validate_deployment
    import json,hashlib,pytest
    config={'model':'mlp','preprocessing_fingerprint':'fp'}
    (tmp_path/'model_config.json').write_text(json.dumps(config))
    (tmp_path/'model.pt').write_bytes(b'A')
    report={'model':'mlp','target_met':True,'preprocessing_fingerprint':'fp',
            'artifact_sha256':{'model.pt':hashlib.sha256(b'A').hexdigest()}}
    validate_deployment(report,tmp_path)
    (tmp_path/'model.pt').write_bytes(b'B')
    with pytest.raises(ValueError,match='benchmark'):validate_deployment(report,tmp_path)
