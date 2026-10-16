#!/usr/bin/env python3
"""Probe native detector and optional camera; saves diagnostics only, never images/video."""
from pathlib import Path
import argparse,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from dataclasses import asdict
from handgesture.preprocessing import HandPreprocessor,PreprocessingConfig,sha256
from handgesture.data import write_json

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--detector',type=Path,default=Path('assets/mediapipe/hand_landmarker.task'))
    p.add_argument('--camera',type=int)
    p.add_argument('--frames',type=int,default=30)
    p.add_argument('--output',type=Path,default=Path('docs/implementation/runtime_check.json'))
    a=p.parse_args()
    if a.frames<1:p.error('frames must be positive')
    config=PreprocessingConfig(sha256(a.detector));capture=None;times=[];hands=0;resolutions=set()
    processor=HandPreprocessor(config,a.detector,'IMAGE' if a.camera is None else 'VIDEO')
    try:
        if a.camera is not None:
            capture=cv2.VideoCapture(a.camera)
            if not capture.isOpened():raise RuntimeError('Cannot open camera')
            capture.set(cv2.CAP_PROP_FRAME_WIDTH,640);capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        for index in range(a.frames):
            if capture is None:image=np.zeros((480,640,3),dtype=np.uint8)
            else:
                ok,bgr=capture.read()
                if not ok:raise RuntimeError('Camera read failed')
                image=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            resolutions.add((image.shape[1],image.shape[0]));start=time.perf_counter()
            result=processor.process(image,int(start*1000));times.append((time.perf_counter()-start)*1000)
            hands+=int(result.status=='ok')
    finally:
        if capture is not None:capture.release()
        processor.close()
    import importlib.metadata
    report={'python':sys.version.split()[0],'versions':{n:importlib.metadata.version(n) for n in ['mediapipe','numpy','opencv-contrib-python']},
            'detector_initialized':True,'camera':a.camera,'frames':len(times),'resolutions':[list(r) for r in sorted(resolutions)],
            'hand_frames':hands,'detector_preprocessing_median_ms':float(np.median(times)),
            'preprocessing':asdict(config),'preprocessing_fingerprint':config.fingerprint,
            'scope':'native detector/environment probe only; no trained classifier, no recognition accuracy or end-to-end FPS claim',
            'camera_released':True,'saved_images_or_video':False}
    a.output.parent.mkdir(parents=True,exist_ok=True);write_json(a.output,report);print(json.dumps(report,indent=2))
