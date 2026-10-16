"""Model registry and inference; inference never downloads pretrained weights."""
from pathlib import Path
import json
import random

import joblib
import numpy as np
from PIL import Image,ImageEnhance,ImageOps
import torch
from torch import nn

from handgesture.data import MAPPING,LM_COLUMNS
from handgesture.preprocessing import sha256

ARCHITECTURES = ['gap','spatial','mobilenet_v3_small']
IMAGENET_MEAN = torch.tensor([.485,.456,.406])[:,None,None]
IMAGENET_STD = torch.tensor([.229,.224,.225])[:,None,None]


class GestureMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(63,128),nn.ReLU(),nn.Dropout(.2),
                                     nn.Linear(128,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,6))
    def forward(self,x): return self.network(x)


class GestureCNN(nn.Module):
    def __init__(self,spatial=False):
        super().__init__()
        blocks=[];channels=3
        for output in [32,64,128]:
            blocks += [nn.Conv2d(channels,output,3,padding=1),nn.BatchNorm2d(output),
                       nn.ReLU(),nn.MaxPool2d(2)]
            channels=output
        self.features=nn.Sequential(*blocks)
        self.classifier=(nn.Sequential(nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),
                         nn.Linear(2048,128),nn.ReLU(),nn.Dropout(.3),nn.Linear(128,6)) if spatial else
                         nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(.2),nn.Linear(128,6)))
    def forward(self,x): return self.classifier(self.features(x))


def build_model(architecture,initialize_pretrained=False,weights_path=None):
    if architecture=='mlp': return GestureMLP()
    if architecture in {'gap','spatial'}: return GestureCNN(architecture=='spatial')
    if architecture!='mobilenet_v3_small': raise ValueError(f'Unknown architecture: {architecture}')
    from torchvision.models import mobilenet_v3_small
    model=mobilenet_v3_small(weights=None)
    if initialize_pretrained:
        if weights_path is None or not Path(weights_path).is_file():
            raise ValueError('Pretrained MobileNet weights file is required; no random-init fallback')
        model.load_state_dict(torch.load(weights_path,map_location='cpu',weights_only=True))
    model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,6)
    return model


def image_tensor(image,architecture='gap',training=False):
    image=Image.fromarray(image) if isinstance(image,np.ndarray) else image.convert('RGB')
    image=ImageOps.exif_transpose(image).convert('RGB')
    size=224 if architecture=='mobilenet_v3_small' else 128
    image=image.resize((size,size),Image.Resampling.BILINEAR)
    if training:
        image=image.rotate(random.uniform(-10,10),resample=Image.Resampling.BILINEAR)
        image=ImageEnhance.Brightness(image).enhance(random.uniform(.85,1.15))
        image=ImageEnhance.Contrast(image).enhance(random.uniform(.85,1.15))
    tensor=torch.from_numpy(np.array(image,dtype=np.float32).transpose(2,0,1).copy())/255
    if architecture=='mobilenet_v3_small': tensor=(tensor-IMAGENET_MEAN)/IMAGENET_STD
    return tensor


class GesturePredictor:
    def __init__(self,directory,preprocessing_config=None,allow_legacy=False):
        self.directory=Path(directory)
        self.config=json.loads((self.directory/'model_config.json').read_text())
        self.kind=self.config['model']
        fingerprint=self.config.get('preprocessing_fingerprint')
        if fingerprint is None:
            if not allow_legacy: raise ValueError('Legacy preprocessing is unverified; use a v2 checkpoint for webcam')
            from scripts.kaggle_notebook_support import load_saved_model
            self.model,self.scaler,_=load_saved_model(self.kind,self.directory)
            self.architecture='gap' if self.kind=='cnn' else self.kind
            return
        if preprocessing_config is not None: preprocessing_config.assert_matches(fingerprint)
        if self.config.get('class_mapping')!=MAPPING:
            raise ValueError('Checkpoint class mapping mismatch')
        record_path=self.directory/('metrics.json' if (self.directory/'metrics.json').exists() else 'run.json')
        record=json.loads(record_path.read_text())
        for filename,digest in record['artifact_sha256'].items():
            if sha256(self.directory/filename)!=digest: raise ValueError(f'Artifact hash mismatch: {filename}')
        if record['config']!=self.config: raise ValueError('Config differs from run record')
        self.architecture=self.config['architecture']
        if self.kind=='rf':
            self.model=joblib.load(self.directory/'model.joblib');self.model.set_params(n_jobs=1)
        else:
            self.model=build_model(self.architecture)
            self.model.load_state_dict(torch.load(self.directory/'model.pt',map_location='cpu',weights_only=True))
            self.model.eval()
        self.scaler=joblib.load(self.directory/'scaler.joblib') if self.kind=='mlp' else None

    def predict(self,features=None,roi=None):
        return self.predict_input(self.prepare_input(features,roi))

    def prepare_input(self,features=None,roi=None):
        if self.kind in {'rf','mlp'}:
            return np.asarray(features,dtype=float).reshape(-1,63)
        return image_tensor(roi,self.architecture).unsqueeze(0)

    def predict_input(self,value):
        if self.kind=='rf': return self.model.predict_proba(value)[0]
        with torch.inference_mode():
            if self.kind=='mlp':
                x=self.scaler.transform(value).astype(np.float32)
                tensor=torch.from_numpy(x)
            else: tensor=value
            return self.model(tensor).softmax(1).numpy()[0]
