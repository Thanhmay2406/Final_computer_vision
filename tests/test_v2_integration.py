"""Real tiny training round trips, candidate selection and immutable verification."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import hashlib
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torchvision.models import mobilenet_v3_small
from handgesture.data import prepare_dataset
from handgesture.preprocessing import HandPreprocessor,PreprocessingConfig
from handgesture.training import train_run,finalize,verify_outputs,regenerate_outputs
from handgesture.models import GesturePredictor


def fixture_dataset(tmp_path):
    source=tmp_path/'source';(source/'metadata').mkdir(parents=True);rows=[]
    rng=np.random.default_rng(42)
    for split in ['train','validation','test']:
        for c,g in enumerate(['fist','one','peace','three','four','palm']):
            name=f'{split}-{g}.png';Image.fromarray(rng.integers(0,256,(40,40,3),dtype=np.uint8)).save(source/name)
            rows.append(dict(sample_id=name,class_id=c,gesture=g,split=split,image_path=name))
    pd.DataFrame(rows).to_csv(source/'metadata/manifest.csv',index=False)
    points=np.zeros((21,3));points[:,0]=np.linspace(.2,.8,21);points[:,1]=np.linspace(.2,.8,21)
    processor=HandPreprocessor(PreprocessingConfig('fixture'),detector=lambda im,t:points)
    data=tmp_path/'derived';prepare_dataset(source,data,processor=processor,limit=2)
    return data


def test_train_finalize_reload_verify_and_tamper_detection(tmp_path):
    torch.set_num_threads(1);data=fixture_dataset(tmp_path)
    weights=tmp_path/'fixture.pth';torch.save(mobilenet_v3_small(weights=None).state_dict(),weights)
    for kind in ['rf','mlp']:train_run(data,tmp_path/kind,kind,smoke=True,device='cpu')
    candidates=[]
    for architecture in ['gap','spatial','mobilenet_v3_small']:
        out=tmp_path/architecture;candidates.append(out)
        train_run(data,out,'cnn',architecture,weights,smoke=True,device='cpu')
        assert not (out/'test_predictions.csv').exists()
    final=tmp_path/'final';finalize(data,tmp_path/'rf',tmp_path/'mlp',candidates,final)
    assert set(verify_outputs(final,data))=={'rf','mlp','cnn'}
    for kind in ['rf','mlp','cnn']:
        run=final/kind;before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in run.iterdir() if p.is_file()}
        if kind!='rf':assert 'phase' in pd.read_csv(run/'history.csv').columns
        predictor=GesturePredictor(run,PreprocessingConfig('fixture'))
        frame=pd.read_csv(data/'metadata/manifest.csv');row=frame.loc[frame.split.eq('test')].iloc[0]
        with Image.open(data/row.image_path) as im:roi=np.array(im)
        p=predictor.predict(row[[f'lm_{i}' for i in range(63)]].to_numpy(dtype=float),roi)
        saved=pd.read_csv(run/'test_predictions.csv').set_index('sample_id').loc[row.sample_id]
        np.testing.assert_allclose(p,saved[[f'prob_{i}' for i in range(6)]].to_numpy(dtype=float),atol=3e-6)
        assert before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in run.iterdir() if p.is_file()}
        with pytest.raises(ValueError,match='fingerprint'):GesturePredictor(run,PreprocessingConfig('wrong'))
    protected=[p for p in final.rglob('*') if p.is_file() and p.suffix in {'.pt','.joblib','.json'} or p.is_file() and p.name in ['test_predictions.csv','validation_predictions.csv','history.csv']]
    hashes={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    (final/'cnn/summary.csv').unlink()
    regenerate_outputs(final,data)
    assert hashes=={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    verify_outputs(final,data)
    path=final/'mlp/validation_predictions.csv';path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='hash'):verify_outputs(final,data)
