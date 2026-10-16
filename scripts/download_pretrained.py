#!/usr/bin/env python3
"""Download and verify official ImageNet weights once, never during inference."""
from pathlib import Path
import argparse,hashlib,urllib.request
URL='https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth'
EXPECTED='047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f'
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('assets/pretrained/mobilenet_v3_small-047dcff4.pth'))
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    if not a.output.exists():
        stage=a.output.with_suffix('.download');urllib.request.urlretrieve(URL,stage)
        if hashlib.sha256(stage.read_bytes()).hexdigest()!=EXPECTED:stage.unlink();raise ValueError('Weights hash mismatch')
        stage.rename(a.output)
    if hashlib.sha256(a.output.read_bytes()).hexdigest()!=EXPECTED:raise ValueError('Weights hash mismatch')
    print(a.output,EXPECTED)
