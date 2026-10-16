#!/usr/bin/env python3
"""Run validation-only training, finalization or read-only verification."""
from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from handgesture.training import train_run,finalize,verify_outputs,regenerate_outputs

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    t=sub.add_parser('train');t.add_argument('--model',choices=['rf','mlp','cnn'],required=True)
    t.add_argument('--architecture',choices=['gap','spatial','mobilenet_v3_small'])
    t.add_argument('--data-root',type=Path,default=Path('data_v2'));t.add_argument('--output-dir',type=Path,required=True)
    t.add_argument('--weights',type=Path);t.add_argument('--device',default='auto');t.add_argument('--smoke',action='store_true')
    t.add_argument('--workers',type=int,default=0)
    f=sub.add_parser('finalize');f.add_argument('--data-root',type=Path,default=Path('data_v2'))
    f.add_argument('--rf-dir',type=Path,required=True);f.add_argument('--mlp-dir',type=Path,required=True)
    f.add_argument('--cnn-dirs',type=Path,nargs=3,required=True);f.add_argument('--output-dir',type=Path,required=True)
    v=sub.add_parser('verify');v.add_argument('--results-root',type=Path,required=True);v.add_argument('--data-root',type=Path)
    r=sub.add_parser('report');r.add_argument('--results-root',type=Path,required=True);r.add_argument('--data-root',type=Path,required=True)
    a=p.parse_args()
    if a.action=='train':train_run(a.data_root,a.output_dir,a.model,a.architecture,a.weights,a.smoke,a.device,a.workers)
    elif a.action=='finalize':finalize(a.data_root,a.rf_dir,a.mlp_dir,a.cnn_dirs,a.output_dir)
    elif a.action=='report':regenerate_outputs(a.results_root,a.data_root);print('Regenerated derived reports; preserved measured artifacts')
    else:verify_outputs(a.results_root,a.data_root);print('Verified all three models and validation/test predictions')
