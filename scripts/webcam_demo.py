#!/usr/bin/env python3
"""Recognize six gestures on CPU; use only v2 models with matching preprocessing."""
from pathlib import Path
import argparse
import json
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import torch
from threadpoolctl import threadpool_limits
from handgesture.models import GesturePredictor
from handgesture.preprocessing import PreprocessingConfig,HandPreprocessor
from handgesture.live import run_stream,DISPLAY_MODES
from handgesture.data import write_json
from handgesture.training import environment
from handgesture.benchmark import validate_deployment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['rf','mlp','cnn'])
    parser.add_argument('--mode',choices=DISPLAY_MODES,default='count',
                        help='Display counting or rock-paper-scissors; press m to switch')
    parser.add_argument('--checkpoint-dir',type=Path,default=Path('results_v2/final'))
    parser.add_argument('--deployment',type=Path)
    parser.add_argument('--detector',type=Path,default=Path('assets/mediapipe/hand_landmarker.task'))
    parser.add_argument('--camera',type=int,default=0)
    parser.add_argument('--width',type=int,default=640);parser.add_argument('--height',type=int,default=480)
    parser.add_argument('--threshold',type=float,default=.7);parser.add_argument('--window',type=int,default=7)
    parser.add_argument('--votes',type=int,default=5);parser.add_argument('--frames',type=int)
    parser.add_argument('--warmup',type=int,default=30);parser.add_argument('--headless',action='store_true')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    model=args.model
    if model is None:
        if args.deployment is None:parser.error('Use --model until a completed benchmark selects --deployment')
        deployment=json.loads(args.deployment.read_text());model=deployment.get('model')
        if not deployment.get('target_met') or model not in ['rf','mlp','cnn']:
            parser.error('Deployment report has no eligible model')
    folder=args.checkpoint_dir
    if not (folder/'model_config.json').exists():folder=folder/model
    if args.deployment is not None:
        validate_deployment(json.loads(args.deployment.read_text()),folder)
    config=json.loads((folder/'model_config.json').read_text())
    if config['model']!=model:parser.error('Checkpoint model does not match --model')
    if config.get('smoke_test'):parser.error('Smoke checkpoint cannot be used as a final webcam model')
    if 'preprocessing_fingerprint' not in config:parser.error('Legacy checkpoint has unverified preprocessing; train v2 first')
    processing=PreprocessingConfig(**config['preprocessing'])
    predictor=GesturePredictor(folder,processing)
    # Instantiate detector before opening the camera; native initialization failures cannot leak a camera handle.
    processor=HandPreprocessor(processing,args.detector,'VIDEO')
    camera=cv2.VideoCapture(args.camera)
    if not camera.isOpened():camera.release();processor.close();parser.error('Cannot open camera')
    camera.set(cv2.CAP_PROP_FRAME_WIDTH,args.width);camera.set(cv2.CAP_PROP_FRAME_HEIGHT,args.height)
    output=args.output_dir or Path('results_v2/webcam')/time.strftime('%Y%m%d-%H%M%S')
    if output.exists() and any(output.iterdir()):camera.release();processor.close();parser.error('Output directory is not empty')
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        result=run_stream(camera,processor,predictor,output,args.warmup,args.frames,not args.headless,
                          args.threshold,args.window,args.votes,display_mode=args.mode)
    write_json(output/'protocol.json',{'environment':environment('cpu'),'camera':args.camera,
             'requested_resolution':[args.width,args.height],'model_config':config,
             'display_mode_initial':args.mode,'mode_switch_key':'m',
             'scope':'live camera timing, no labeled accuracy evaluation; no video saved'})
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
