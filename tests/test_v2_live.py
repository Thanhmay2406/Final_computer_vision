"""Webcam behavior: no stale gestures, measured timings and resource cleanup."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
from handgesture.live import DecisionFilter,run_stream
from handgesture.preprocessing import HandPreprocessor,PreprocessingConfig


def test_filter_requires_consensus_and_immediately_clears_missing_hand():
    f=DecisionFilter();p=np.array([.9,.02,.02,.02,.02,.02])
    assert [f.update(p)['status'] for _ in range(5)]==['uncertain']*4+['recognized']
    assert f.update(None)['status']=='no_hand'
    assert f.update(p)['status']=='uncertain'
    assert f.update(np.full(6,1/6))['status']=='uncertain'


class Camera:
    def __init__(self,frames):self.frames=iter(frames);self.closed=False
    def read(self):return next(self.frames,(False,None))
    def release(self):self.closed=True


class Predictor:
    kind='rf';config={'preprocessing_fingerprint':PreprocessingConfig('test').fingerprint}
    def prepare_input(self,features,roi):return features
    def predict_input(self,value):return np.array([.9,.02,.02,.02,.02,.02])


def test_stream_times_frames_releases_camera_and_does_not_save_video(tmp_path):
    points=np.zeros((21,3));points[:,0]=np.linspace(.2,.8,21);points[:,1]=.5
    processor=HandPreprocessor(PreprocessingConfig('test'),detector=lambda im,t:points if im.max() else None)
    frames=[(True,np.full((30,30,3),255,np.uint8)) for _ in range(7)]+[(True,np.zeros((30,30,3),np.uint8))]
    camera=Camera(frames)
    result=run_stream(camera,processor,Predictor(),tmp_path,warmup=1,measured_frames=7,display=False)
    assert camera.closed and result['measured_frames']==7 and result['hand_frames']==6
    rows=pd.read_csv(tmp_path/'frames.csv')
    assert rows.iloc[-1].status=='no_hand' and result['fps']>0
    assert all(rows.total_ms>=rows.classifier_ms)
    assert not list(tmp_path.glob('*.mp4'))


def test_stream_releases_camera_on_detector_failure(tmp_path):
    camera=Camera([(True,np.zeros((30,30,3),np.uint8))])
    def fail(im,t):raise RuntimeError('detector failed')
    processor=HandPreprocessor(PreprocessingConfig('test'),detector=fail)
    with pytest.raises(RuntimeError,match='detector failed'):
        run_stream(camera,processor,Predictor(),tmp_path,display=False)
    assert camera.closed


def test_unexpected_camera_end_is_not_a_completed_live_session(tmp_path):
    camera=Camera([(True,np.zeros((30,30,3),np.uint8))])
    processor=HandPreprocessor(PreprocessingConfig('test'),detector=lambda im,t:None)
    result=run_stream(camera,processor,Predictor(),tmp_path,display=False)
    assert not result['complete'] and result['termination_reason']=='source_ended'


@pytest.mark.parametrize('gesture,number,rps',[
    ('fist','0','Búa'),('one','1','Ngoài bộ cử chỉ'),('peace','2','Kéo'),
    ('three','3','Ngoài bộ cử chỉ'),('four','4','Ngoài bộ cử chỉ'),('palm','5','Bao')])
def test_display_modes_keep_count_and_rps_separate(gesture,number,rps):
    from handgesture.live import format_display_label
    decision={'status':'recognized','label':gesture,'confidence':.9}
    assert format_display_label(decision,'count')==number
    assert format_display_label(decision,'rps')==rps
    assert decision['label']==gesture


@pytest.mark.parametrize('mode',['count','rps'])
@pytest.mark.parametrize('status',['no_hand','uncertain'])
def test_display_mode_never_exposes_a_stale_recognition(mode,status):
    from handgesture.live import format_display_label
    assert format_display_label({'status':status,'label':'peace','confidence':.1},mode)==status


def test_mode_switch_logs_the_mode_actually_rendered_and_keeps_raw_class(tmp_path,monkeypatch):
    import handgesture.live as live
    points=np.zeros((21,3));points[:,0]=np.linspace(.2,.8,21);points[:,1]=.5
    processor=HandPreprocessor(PreprocessingConfig('test'),detector=lambda im,t:points)
    camera=Camera([(True,np.full((80,120,3),255,np.uint8)) for _ in range(3)])
    class PeacePredictor(Predictor):
        def predict_input(self,value):return np.array([.02,.02,.9,.02,.02,.02])
    keys=iter([ord('m'),ord('m'),ord('q')]);rendered=[]
    real_render=live.render_frame
    def capture_render(*args,**kwargs):
        rendered.append(kwargs['display_mode'])
        return real_render(*args,**kwargs)
    monkeypatch.setattr(live,'render_frame',capture_render)
    monkeypatch.setattr(live.cv2,'imshow',lambda *args:None)
    monkeypatch.setattr(live.cv2,'destroyAllWindows',lambda:None)
    monkeypatch.setattr(live.cv2,'waitKey',lambda n:next(keys))
    result=run_stream(camera,processor,PeacePredictor(),tmp_path,display=True,
                      window=1,votes=1,display_mode='count')
    rows=pd.read_csv(tmp_path/'frames.csv')
    assert rendered==['count','rps','count']
    assert rows.display_mode.tolist()==rendered
    assert rows.display_label.astype(str).tolist()==['2','Kéo','2']
    assert rows.label.tolist()==['peace']*3
    assert result['mode_switches']==2 and result['display_mode_initial']=='count'
    assert result['display_mode_final']=='count' and result['termination_reason']=='user_exit'
    assert camera.closed


def test_invalid_display_mode_releases_resources(tmp_path):
    camera=Camera([])
    processor=HandPreprocessor(PreprocessingConfig('test'),detector=lambda im,t:None)
    with pytest.raises(ValueError,match='display mode'):
        run_stream(camera,processor,Predictor(),tmp_path,display=False,display_mode='both')
    assert camera.closed
