#!/usr/bin/env python3
"""Fetch completed private v2 archives, verify hashes and refuse existing destinations."""
from pathlib import Path
import argparse,hashlib,json,zipfile


def kernel_state(response):
    value=response.status
    return getattr(value,'name',str(value)).upper().split('.')[-1]


def extract_archive(archive,destination):
    destination=Path(destination).resolve()
    if destination.exists():raise ValueError(f'Refusing to overwrite {destination}')
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if not (destination/info.filename).resolve().is_relative_to(destination):
                raise ValueError('Archive member escapes destination')
        z.extractall(destination)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kernel',default='thanhmay2406/hand-gesture-v2-full-20261015')
    p.add_argument('--output-root',type=Path,default=Path('results_v2/kaggle_20261015'))
    p.add_argument('--data-root',type=Path,default=Path('data_v2_kaggle'))
    a=p.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate();status=api.kernels_status(a.kernel)
    state=kernel_state(status)
    print('Kernel status:',state)
    if state!='COMPLETE':return
    downloads=a.output_root/'downloads';downloads.mkdir(parents=True,exist_ok=True)
    token=None
    while True:
        _,token=api.kernels_output(a.kernel,str(downloads),file_pattern=r'^(data_v2\.zip|results_v2\.zip|export_manifest\.json)$',
                                   force=False,page_token=token,page_size=1000)
        if not token:break
    manifest=json.loads((downloads/'export_manifest.json').read_text())
    for name,digest in manifest['archives'].items():
        if hashlib.sha256((downloads/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'Archive hash mismatch: {name}')
    extract_archive(downloads/'data_v2.zip',a.data_root)
    extract_archive(downloads/'results_v2.zip',a.output_root/'experiment')
    # Import after downloading so status-only operation needs no scientific packages.
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from handgesture.training import verify_outputs
    verify_outputs(a.output_root/'experiment/final',a.data_root)
    print('Verified:',a.output_root/'experiment/final','dataset:',a.data_root)

if __name__=='__main__':main()
