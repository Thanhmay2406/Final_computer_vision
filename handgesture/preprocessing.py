"""The same geometry is used for Kaggle images and unmirrored webcam frames."""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PreprocessingConfig:
    detector_sha256: str
    version: str = 'pixel_wrist_max3d_square_v2'
    detection_confidence: float = .5
    presence_confidence: float = .5
    tracking_confidence: float = .5
    roi_scale: float = 1.4
    epsilon: float = 1e-6
    num_hands: int = 1
    mirror_input: bool = False
    detector_input_padding: float = .5
    detector_runtime_version: str = '0.10.21'

    def __post_init__(self):
        if (self.version != 'pixel_wrist_max3d_square_v2' or self.roi_scale < 1
                or self.epsilon <= 0 or self.num_hands != 1 or self.mirror_input
                or not 0 <= self.detector_input_padding <= 1
                or any(not 0 <= v <= 1 for v in (self.detection_confidence,
                       self.presence_confidence, self.tracking_confidence))):
            raise ValueError('Invalid preprocessing configuration')

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True,
                                        allow_nan=False).encode()).hexdigest()

    def assert_matches(self, fingerprint):
        if fingerprint != self.fingerprint:
            raise ValueError('Preprocessing fingerprint mismatch; retrain with this pipeline')


def normalize_landmarks(landmarks, width, height, epsilon=1e-6):
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (21, 3) or not np.isfinite(points).all() or min(width, height) <= 0:
        raise ValueError('Invalid landmarks or image dimensions')
    pixel = points * np.array([width, height, width])
    relative = pixel - pixel[0]
    scale = np.linalg.norm(relative, axis=1).max()
    if not np.isfinite(scale) or scale <= epsilon:
        raise ValueError('Degenerate landmark scale')
    return (relative / scale).astype(np.float32).reshape(63)


def square_roi(image, landmarks, roi_scale=1.4):
    height, width = image.shape[:2]
    points = np.asarray(landmarks, dtype=float)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError('Invalid ROI landmarks')
    xy = points[:, :2] * [width, height]
    low, high = xy.min(axis=0), xy.max(axis=0)
    side = int(math.ceil(float((high-low).max()) * roi_scale))
    if side < 1 or side > 4 * max(width, height):
        raise ValueError('Invalid ROI extent')
    center = (low+high)/2
    left, top = np.floor(center-side/2).astype(int)
    right, bottom = left+side, top+side
    sx0, sy0, sx1, sy1 = max(0,left), max(0,top), min(width,right), min(height,bottom)
    if sx1 <= sx0 or sy1 <= sy0:
        raise ValueError('ROI outside image')
    roi = np.zeros((side,side,3), dtype=np.uint8)
    roi[sy0-top:sy1-top, sx0-left:sx1-left] = image[sy0:sy1,sx0:sx1]
    return roi, (int(left),int(top),int(right),int(bottom))


@dataclass
class HandResult:
    status: str
    features: object = None
    roi: object = None
    landmarks: object = None
    bbox: object = None
    reason: str = ''


class HandPreprocessor:
    def __init__(self, config, model_path=None, mode='IMAGE', detector=None):
        self.config = config
        self._landmarker = None
        self._last_timestamp = -1
        self.mode = mode
        if detector is not None:
            self._detect = detector
            return
        if mode not in {'IMAGE','VIDEO'}:
            raise ValueError('mode must be IMAGE or VIDEO')
        if sha256(model_path) != config.detector_sha256:
            raise ValueError('Detector SHA-256 mismatch')
        from importlib.metadata import version
        if version('mediapipe') != config.detector_runtime_version:
            raise ValueError('MediaPipe runtime differs from fingerprint; use the documented v2 environment')
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path),delegate=BaseOptions.Delegate.CPU),
            running_mode=getattr(vision.RunningMode,mode), num_hands=config.num_hands,
            min_hand_detection_confidence=config.detection_confidence,
            min_hand_presence_confidence=config.presence_confidence,
            min_tracking_confidence=config.tracking_confidence)
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._detect = self._mediapipe_detect

    def _mediapipe_detect(self, image, timestamp_ms):
        value = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(image))
        if self.mode == 'VIDEO':
            timestamp_ms = max(int(timestamp_ms), self._last_timestamp+1)
            self._last_timestamp = timestamp_ms
            result = self._landmarker.detect_for_video(value,timestamp_ms)
        else:
            result = self._landmarker.detect(value)
        if not result.hand_landmarks:
            return None
        return np.array([[p.x,p.y,p.z] for p in result.hand_landmarks[0]])

    def detect(self, image, timestamp_ms=0):
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError('Expected uint8 RGB image HxWx3')
        height,width=image.shape[:2]
        pad_x,pad_y=int(width*self.config.detector_input_padding),int(height*self.config.detector_input_padding)
        padded=np.pad(image,((pad_y,pad_y),(pad_x,pad_x),(0,0)))
        points=self._detect(padded,timestamp_ms)
        return None if points is None else self.map_detector_landmarks(points,width,height)

    def map_detector_landmarks(self,points,width,height):
        points=np.asarray(points,dtype=float).copy()
        if points.shape!=(21,3):raise ValueError('Invalid detector landmark shape')
        pad_x,pad_y=int(width*self.config.detector_input_padding),int(height*self.config.detector_input_padding)
        points[:,0]=(points[:,0]*(width+2*pad_x)-pad_x)/width
        points[:,1]=(points[:,1]*(height+2*pad_y)-pad_y)/height
        points[:,2]*=(width+2*pad_x)/width
        return points

    def transform(self, image, points):
        if points is None:
            return HandResult('no_hand', reason='no_hand')
        try:
            features = normalize_landmarks(points,image.shape[1],image.shape[0],self.config.epsilon)
            roi,bbox = square_roi(image,points,self.config.roi_scale)
        except ValueError as exc:
            return HandResult('invalid',reason=str(exc))
        return HandResult('ok',features,roi,np.asarray(points),bbox)

    def process(self, image, timestamp_ms=0):
        return self.transform(image,self.detect(image,timestamp_ms))

    def close(self):
        if self._landmarker is not None:
            self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()
