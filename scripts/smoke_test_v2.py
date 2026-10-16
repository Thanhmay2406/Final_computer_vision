#!/usr/bin/env python3
"""Execute all v2 notebooks on synthetic images; no camera, network or native detector.

The MediaPipe boundary is injected; all geometry, file IO, learning, selection,
artifact validation, model round trips and classifier benchmarking are real.
"""
from pathlib import Path
import argparse,json,sys,tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision.models import mobilenet_v3_small
from handgesture.training import verify_outputs
from handgesture.benchmark import benchmark_classifiers
ROOT=Path(__file__).resolve().parents[1]


def main():
    # Keep the CPU smoke workload bounded, including fixture weight initialization.
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path)
    p.add_argument('--notebook',help='Execute one named notebook, e.g. 07_interactive_workflow.ipynb')
    args=p.parse_args();output=(args.output_dir or Path(tempfile.mkdtemp(prefix='handgesture-v2-smoke-'))).resolve()
    output.mkdir(parents=True,exist_ok=True)
    source=output/'source';(source/'metadata').mkdir(parents=True)
    rng=np.random.default_rng(42);rows=[]
    for split,count in [('train',8),('validation',4),('test',4)]:
        for c,g in enumerate(['fist','one','peace','three','four','palm']):
            for i in range(count):
                name=f'{split}-{g}-{i}.png';Image.fromarray(rng.integers(0,256,(40,40,3),dtype=np.uint8)).save(source/name)
                rows.append(dict(sample_id=name,class_id=c,gesture=g,split=split,image_path=name))
    pd.DataFrame(rows).to_csv(source/'metadata/manifest.csv',index=False)
    weights=output/'fixture_weights.pth';torch.save(mobilenet_v3_small(weights=None).state_dict(),weights)
    with tempfile.TemporaryDirectory(prefix='v2-kernel-') as kernel_dir:
        spec=Path(kernel_dir)/'smoke';spec.mkdir()
        (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
                                                   'display_name':'V2 smoke','language':'python'}))
        manager=KernelSpecManager(kernel_dirs=[kernel_dir])
        class Kernel(AsyncKernelManager):
            def __init__(self,**kwargs):super().__init__(kernel_spec_manager=manager,**kwargs)
        work=output/'work'
        paths=sorted((ROOT/'notebooks/kaggle_v2').glob('*.ipynb'))
        if args.notebook:
            paths=[p for p in paths if p.name==args.notebook]
            if not paths:raise ValueError('Notebook not found')
        for path in paths:
            notebook=nbformat.read(path,4)
            config=next(c for c in notebook.cells if c.cell_type=='code')
            original=source if path.name.startswith(('00','06','07')) else work/'data_v2'
            config.source+=f"\nDATA_ROOT=Path({str(original)!r})\nOUTPUT_ROOT=Path({str(work)!r})\nSMOKE_TEST=True\nNUM_WORKERS=0\nDEVICE='cpu'\nWEIGHTS_PATH=Path({str(weights)!r})\n"
            if path.name.startswith(('00','06','07')):
                config.source+=f"\nDETECTOR_PATH=Path({str(ROOT/'assets/mediapipe/hand_landmarker.task')!r})\n"
                # Inject only the external native detection boundary after modules are embedded.
                inject_at=next((i for i,c in enumerate(notebook.cells) if c.metadata.get('step')=='preprocessing'),len(notebook.cells)-1)
                notebook.cells.insert(inject_at,nbformat.v4.new_code_cell('''import numpy as np
import handgesture.data as _data
from handgesture.preprocessing import HandPreprocessor as _RealPreprocessor
_points=np.zeros((21,3));_points[:,0]=np.linspace(.2,.8,21);_points[:,1]=np.linspace(.2,.8,21)
def _factory(config,model_path=None,mode='IMAGE'):
    return _RealPreprocessor(config,detector=lambda image,timestamp:_points)
_data.HandPreprocessor=_factory
'''))
            if path.name.startswith('04'):
                config.source+=f"\nRF_DIR=Path({str(work/'runs/rf')!r})\nMLP_DIR=Path({str(work/'runs/mlp')!r})\nCNN_DIRS={[str(work/'candidates'/a) for a in ['gap','spatial','mobilenet_v3_small']]!r}\n"
            if path.name.startswith('05'):
                config.source+=f"\nRESULTS_ROOT=Path({str(work/'final')!r})\n"
            if path.name.startswith('06'):
                config.source+=f"\nOUTPUT_ROOT=Path({str(output/'full')!r})\n"
            if path.name.startswith('07'):
                config.source+=f"\nOUTPUT_ROOT=Path({str(output/'manual')!r})\nREQUIRE_CUDA=False\nEVALUATE_TEST=True\nINSTALL_DEPENDENCIES=False\n"
            notebook.cells.insert(1,nbformat.v4.new_code_cell('import torch\ntorch.set_num_threads(1)'))
            print('Executing',path.name,flush=True)
            NotebookClient(notebook,timeout=600,kernel_name='smoke',kernel_manager_class=Kernel,
                           resources={'metadata':{'path':str(ROOT)}}).execute()
            nbformat.write(notebook,output/(path.stem+'_SMOKE.ipynb'))
    evaluated=output/'manual' if args.notebook and args.notebook.startswith('07') else work
    results=verify_outputs(evaluated/'final',evaluated/'data_v2')
    assert all(r['metrics']['protocol']['smoke_test'] for r in results.values())
    benchmark_classifiers(evaluated/'final',evaluated/'data_v2',output/'classifier')
    print(f'{len(paths)} notebooks, final outputs and classifier benchmark passed. SMOKE ONLY:',output)

if __name__=='__main__':main()
