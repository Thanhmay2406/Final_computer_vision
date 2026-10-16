"""CPU webcam loop and comparable timing. Never writes camera images/video."""
from collections import Counter,deque
from functools import lru_cache
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont

from handgesture.data import GESTURES,write_json

CONNECTIONS=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),
             (10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
DISPLAY_MODES=('count','rps')
MODE_TITLES={'count':'Số đếm','rps':'Kéo–búa–bao'}
RPS_LABELS={'fist':'Búa','peace':'Kéo','palm':'Bao'}


def format_display_label(decision,mode):
    if mode not in DISPLAY_MODES:raise ValueError('Invalid display mode')
    if decision['status']!='recognized':return decision['status']
    label=decision['label']
    if mode=='count':return str(GESTURES.index(label))
    return RPS_LABELS.get(label,'Ngoài bộ cử chỉ')


@lru_cache(maxsize=4)
def overlay_font(size):
    # Matplotlib ships this Unicode font with the existing project dependencies.
    from matplotlib import get_data_path
    return ImageFont.truetype(str(Path(get_data_path())/'fonts/ttf/DejaVuSans.ttf'),size)


class DecisionFilter:
    def __init__(self,threshold=.7,window=7,votes=5):
        if not 0<=threshold<=1 or not 1<=votes<=window:
            raise ValueError('Invalid smoothing configuration')
        self.threshold=threshold;self.window=window;self.votes=votes
        self.history=deque(maxlen=window)

    def update(self,probabilities):
        if probabilities is None:
            self.history.clear()
            return {'status':'no_hand','label':None,'confidence':0.}
        p=np.asarray(probabilities)
        if p.shape!=(6,) or not np.isfinite(p).all():raise ValueError('Invalid probabilities')
        label=int(p.argmax());confidence=float(p[label])
        self.history.append(label if confidence>=self.threshold else None)
        count=Counter(x for x in self.history if x is not None)
        winner,count_value=count.most_common(1)[0] if count else (None,0)
        stable=confidence>=self.threshold and winner==label and count_value>=self.votes
        return {'status':'recognized' if stable else 'uncertain',
                'label':GESTURES[label] if stable else None,'confidence':confidence}


def render_frame(frame,hand,decision,fps,mirror=True,display_mode='count'):
    shown=frame.copy()
    if hand.landmarks is not None:
        xy=(hand.landmarks[:,:2]*[shown.shape[1],shown.shape[0]]).astype(int)
        for a,b in CONNECTIONS:cv2.line(shown,tuple(xy[a]),tuple(xy[b]),(0,255,0),1)
        for x,y in xy:cv2.circle(shown,(x,y),2,(0,180,255),-1)
    if hand.bbox is not None:
        left,top,right,bottom=hand.bbox
        cv2.rectangle(shown,(left,top),(right,bottom),(255,160,0),2)
    if mirror:shown=cv2.flip(shown,1)
    if hand.roi is not None:
        side=min(96,shown.shape[0],shown.shape[1])
        shown[:side,-side:]=cv2.cvtColor(cv2.resize(hand.roi,(side,side)),cv2.COLOR_RGB2BGR)
    label=format_display_label(decision,display_mode)
    # Pillow handles Vietnamese text; OpenCV's built-in fonts replace accents.
    canvas=Image.fromarray(cv2.cvtColor(shown,cv2.COLOR_BGR2RGB))
    draw=ImageDraw.Draw(canvas)
    panel_width=max(draw.textbbox((0,0),text,font=overlay_font(size))[2]
                    for text,size in [(MODE_TITLES[display_mode]+'   [m: đổi chế độ]',16),
                                      (label,26),
                                      (f"Confidence {decision['confidence']:.2f}   {fps:.1f} FPS",16)])+16
    draw.rectangle((0,0,min(panel_width,shown.shape[1]),88),fill=(15,20,28))
    draw.text((8,5),MODE_TITLES[display_mode]+'   [m: đổi chế độ]',font=overlay_font(16),fill=(200,210,225))
    draw.text((8,25),label,font=overlay_font(26),fill=(255,235,80))
    draw.text((8,63),f"Confidence {decision['confidence']:.2f}   {fps:.1f} FPS",
              font=overlay_font(16),fill=(235,240,245))
    return cv2.cvtColor(np.asarray(canvas),cv2.COLOR_RGB2BGR)


def run_stream(capture,processor,predictor,output_dir,warmup=0,measured_frames=None,
               display=True,threshold=.7,window=7,votes=5,mirror=True,display_mode='count'):
    if warmup<0 or (measured_frames is not None and measured_frames<1):raise ValueError('Invalid frame count')
    smoothing=DecisionFilter(threshold,window,votes)
    rows=[];index=0;previous_fps=0.;start_session=time.perf_counter()
    error=None;termination_reason='frame_limit'
    initial_mode=display_mode;mode_switches=0
    try:
        if display_mode not in DISPLAY_MODES:raise ValueError('Invalid display mode')
        while measured_frames is None or len(rows)<measured_frames:
            start=time.perf_counter();ok,frame=capture.read();after_capture=time.perf_counter()
            if not ok:
                termination_reason='source_ended'
                break
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            t=time.perf_counter()
            points=processor.detect(rgb,int((t-start_session)*1000))
            after_detect=time.perf_counter()
            hand=processor.transform(rgb,points)
            prepared=predictor.prepare_input(hand.features,hand.roi) if hand.status=='ok' else None
            after_preprocess=time.perf_counter()
            probabilities=predictor.predict_input(prepared) if prepared is not None else None
            after_classifier=time.perf_counter()
            decision=smoothing.update(probabilities)
            if hand.status=='invalid':decision['status']='uncertain'
            frame_mode=display_mode
            frame_label=format_display_label(decision,frame_mode)
            shown=render_frame(frame,hand,decision,previous_fps,mirror,display_mode=frame_mode)
            quit_requested=False
            if display:
                cv2.imshow('HandGesture - m: mode; q / Esc: quit',shown)
                key=cv2.waitKey(1)&0xff
                quit_requested=key in [ord('q'),27]
                if key in [ord('m'),ord('M')]:
                    display_mode='rps' if display_mode=='count' else 'count'
                    mode_switches+=1
            end=time.perf_counter();previous_fps=1/max(end-start,1e-9)
            if index>=warmup:
                rows.append({'frame':index,'timestamp_ms':(start-start_session)*1000,
                    'capture_ms':(after_capture-start)*1000,'detector_ms':(after_detect-t)*1000,
                    'preprocessing_ms':((t-after_capture)+(after_preprocess-after_detect))*1000,
                    'classifier_ms':(after_classifier-after_preprocess)*1000,
                    'render_ms':(end-after_classifier)*1000,'total_ms':(end-start)*1000,
                    'hand_detected':hand.status=='ok',**decision,
                    'display_mode':frame_mode,'display_label':frame_label,
                    'width':frame.shape[1],'height':frame.shape[0]})
            index+=1
            if quit_requested:
                termination_reason='user_exit'
                break
    except BaseException as exc:
        error=f'{type(exc).__name__}: {exc}'
        termination_reason='error'
        raise
    finally:
        capture.release();processor.close()
        if display:cv2.destroyAllWindows()
        output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(rows).to_csv(output_dir/'frames.csv',index=False)
        elapsed=sum(row['total_ms'] for row in rows)/1000
        hand_rows=[row for row in rows if row['hand_detected']]
        hand_seconds=sum(row['total_ms'] for row in hand_rows)/1000
        result={'measured_frames':len(rows),'requested_frames':measured_frames,'warmup':warmup,
                'complete':bool(rows) and (termination_reason=='user_exit' if measured_frames is None else len(rows)==measured_frames) and error is None,
                'termination_reason':termination_reason,
                'hand_frames':sum(row['hand_detected'] for row in rows),'fps':len(rows)/elapsed if elapsed else 0.,
                'hand_positive_fps':len(hand_rows)/hand_seconds if hand_seconds else 0.,
                'median_ms':float(np.median([r['total_ms'] for r in rows])) if rows else None,
                'p95_ms':float(np.percentile([r['total_ms'] for r in rows],95)) if rows else None,
                'error':error,'display':display,'model':predictor.kind,
                'display_mode_initial':initial_mode,'display_mode_final':display_mode,'mode_switches':mode_switches,
                'smoothing':{'threshold':threshold,'window':window,'votes':votes},
                'preprocessing_fingerprint':processor.config.fingerprint,
                'scope':'capture/decode + detector + preprocessing + classifier + overlay/render; display wait included when enabled'}
        write_json(output_dir/'summary.json',result)
    return result
