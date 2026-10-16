"""Derived datasets must retain splits and account for every rejected source row."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from PIL import Image
from handgesture.preprocessing import HandPreprocessor,PreprocessingConfig
from handgesture.data import prepare_dataset,load_dataset


def test_derived_data_preserves_ids_splits_and_rejection_log(tmp_path):
    source=tmp_path/'source';source.mkdir();rows=[]
    for split_index,split in enumerate(['train','validation','test']):
        for c,g in enumerate(['fist','one','peace','three','four','palm']):
            name=f'{split}-{g}.png';Image.new('RGB',(30,30),(c+1,split_index,0)).save(source/name)
            rows.append(dict(sample_id=name,class_id=c,gesture=g,split=split,image_path=name))
    Image.new('RGB',(30,30)).save(source/'bad.png')
    rows.append(dict(sample_id='bad',class_id=0,gesture='fist',split='train',image_path='bad.png'))
    (source/'metadata').mkdir();pd.DataFrame(rows).to_csv(source/'metadata/manifest.csv',index=False)
    points=np.zeros((21,3));points[:,0]=np.linspace(.2,.8,21);points[:,1]=np.linspace(.2,.8,21)
    processor=HandPreprocessor(PreprocessingConfig('test'),detector=lambda im,t: points if im.max() else None)
    out=tmp_path/'out';prepare_dataset(source,out,processor=processor)
    frame,report=load_dataset(out)
    assert len(frame)==18 and report['counts']['rejected']==1
    parallel=tmp_path/'parallel'
    prepare_dataset(source,parallel,processor=processor,workers=2)
    again,_=load_dataset(parallel)
    pd.testing.assert_frame_equal(frame,again)
    import shutil
    stage=tmp_path/'resume.staging';shutil.copytree(out,stage)
    with (stage/'progress.jsonl').open('ab') as log:log.write(b'{\"sample_id\":')
    resumed,_=prepare_dataset(source,tmp_path/'resume',processor=processor)
    pd.testing.assert_frame_equal(frame,resumed)
    original=pd.DataFrame(rows).set_index('sample_id')
    assert frame.set_index('sample_id')['split'].equals(original.loc[frame.sample_id,'split'])
    assert pd.read_csv(out/'metadata/rejections.csv').reason.tolist()==['no_hand']
    # Corruption must be detected before training.
    (out/frame.iloc[0].image_path).write_bytes(b'changed')
    import pytest
    with pytest.raises(ValueError,match='hash'):load_dataset(out)
