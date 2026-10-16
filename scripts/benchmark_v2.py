#!/usr/bin/env python3
"""Measure classifier latency and optionally the same video through all three models."""
from pathlib import Path
import sys,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from handgesture.benchmark import benchmark_classifiers,benchmark_video

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root',type=Path,default=Path('results_v2/final'))
    p.add_argument('--data-root',type=Path,default=Path('data_v2'))
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--video',type=Path)
    p.add_argument('--display',action='store_true')
    p.add_argument('--detector',type=Path,default=Path('assets/mediapipe/hand_landmarker.task'))
    a=p.parse_args()
    benchmark_classifiers(a.results_root,a.data_root,a.output_dir/'classifier')
    if a.video:benchmark_video(a.results_root,a.video,a.detector,a.output_dir/'video',a.display)
