"""Run all notebooks in exact ~/.venv, saving distinctly labelled smoke outputs outside source data."""
from pathlib import Path
import argparse
import json
import tempfile
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager
from jupyter_client import AsyncKernelManager
import importlib.util
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    output=(args.output_dir or Path(tempfile.mkdtemp(prefix='hand-gesture-smoke-'))).resolve()
    output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='hand-gesture-kernel-') as tmp:
        kernel=Path(tmp)/'venv';kernel.mkdir()
        (kernel/'kernel.json').write_text(json.dumps({'argv':[str(Path.home()/'.venv/bin/python'),'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Project venv','language':'python'}))
        manager=KernelSpecManager(kernel_dirs=[tmp])
        class VenvKernelManager(AsyncKernelManager):
            def __init__(self, **kwargs):
                super().__init__(kernel_spec_manager=manager, **kwargs)
        for name in ['01_random_forest','02_mlp','03_cnn','04_compare_models']:
            notebook=nbformat.read(ROOT/'notebooks/kaggle'/f'{name}.ipynb',as_version=4)
            config=next(cell for cell in notebook.cells if cell.cell_type=='code')
            config.source+=f"\nDATA_ROOT = {str(ROOT/'data')!r}\nOUTPUT_ROOT = Path({str(output)!r})\n"
            if name.startswith('04'):
                config.source+="\nRESULT_DIRS = {name: OUTPUT_ROOT/name for name in ['rf','mlp','cnn']}\n"
                # Exercise the optional model/dataset branch; unit tests cover default results-only mode.
                config.source+='\nRUN_BENCHMARK = True\n'
            else:
                config.source+='\nSMOKE_TEST = True\nNUM_WORKERS = 0\nDEVICE = "cpu"\n'
                # Limit local CPU threads; preserves default notebook behaviour on Kaggle.
                if not name.startswith('01'):config.source+='\nimport torch\ntorch.set_num_threads(1)\n'
            print('Executing',name,flush=True)
            NotebookClient(notebook,timeout=600,kernel_name='venv',kernel_manager_class=VenvKernelManager,
                resources={'metadata':{'path':str(ROOT)}}).execute()
            nbformat.write(notebook,output/f'{name}_SMOKE.ipynb')
    spec=importlib.util.spec_from_file_location('support',ROOT/'scripts/kaggle_notebook_support.py')
    support=importlib.util.module_from_spec(spec);spec.loader.exec_module(support)
    full,_=support.load_dataset(ROOT/'data')
    small=support.smoke_subset(full,{'train':8,'validation':4,'test':4},42)
    test=small.loc[small.split.eq('test')].reset_index(drop=True)
    torch.set_num_threads(1)
    for name in ['rf','mlp','cnn']:
        folder=output/name
        predictions=__import__('pandas').read_csv(folder/'test_predictions.csv')
        model,scaler,config=support.load_saved_model(name,folder)
        if name=='rf':prob=model.predict_proba(test[support.LM_COLUMNS].to_numpy())
        elif name=='mlp':
            assert np.allclose(scaler.mean_,small.loc[small.split.eq('train'),support.LM_COLUMNS].mean().to_numpy())
            with torch.inference_mode():prob=model(torch.from_numpy(scaler.transform(test[support.LM_COLUMNS].to_numpy()).astype(np.float32))).softmax(1).numpy()
        else:
            dataset=support.GestureImages(test,ROOT/'data',False)
            with torch.inference_mode():prob=model(torch.stack([dataset[i][0] for i in range(len(dataset))])).softmax(1).numpy()
        assert predictions.sample_id.tolist()==test.sample_id.tolist()
        assert np.allclose(prob,predictions[[f'prob_{i}' for i in range(6)]].to_numpy(),atol=1e-6)
        metrics=json.loads((folder/'metrics.json').read_text())
        assert metrics['protocol']['smoke_test'] is True
        for artifact in ['summary.csv','per_class_metrics.csv','history.png','report.md']:
            assert (folder/artifact).is_file(), (name,artifact)
        summary=__import__('pandas').read_csv(folder/'summary.csv')
        assert summary.split.tolist()==['validation','test']
        assert summary.sample_count.tolist()==[24,24]
        history=__import__('pandas').read_csv(folder/'history.csv')
        assert history.columns.tolist()==support.HISTORY_COLUMNS
        if name=='rf':
            assert len(history)==1
            assert history[['epoch','learning_rate','train_loss','validation_loss']].isna().all().all()
        if name in {'mlp','cnn'}:
            assert config['max_epochs']==2
            assert config['learning_rate']==(3e-4 if name=='cnn' else 1e-3)
            assert config['scheduler'] is not None if name=='cnn' else config['scheduler'] is None
            history=__import__('pandas').read_csv(folder/'history.csv')
            assert len(history)==2 and np.allclose(history.learning_rate,config['learning_rate'])
    assert (output/'comparison/report.md').is_file()
    assert (output/'comparison/per_class_metrics.csv').is_file()
    assert (output/'comparison/benchmark_protocol.json').is_file()
    comparison=__import__('pandas').read_csv(output/'comparison/comparison.csv')
    assert comparison.cpu_latency_median_ms.notna().all()
    print('All four notebooks and saved-model prediction round trips passed. SMOKE ONLY:',output)

if __name__=='__main__':main()
