#!/usr/bin/env python3
"""Create reproducible paired ROI/landmark data without changing legacy data."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from handgesture.data import prepare_dataset

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,default=Path('data'))
    p.add_argument('--output-root',type=Path,default=Path('data_v2'))
    p.add_argument('--detector',type=Path,default=Path('assets/mediapipe/hand_landmarker.task'))
    p.add_argument('--limit-per-class-split',type=int)
    p.add_argument('--workers',type=int,default=0)
    a=p.parse_args()
    _,report=prepare_dataset(a.source_root,a.output_root,a.detector,limit=a.limit_per_class_split,workers=a.workers)
    print(report['counts'])
