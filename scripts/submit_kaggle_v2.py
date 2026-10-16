#!/usr/bin/env python3
"""Submit a private GPU experiment using an isolated Python3.11 runtime.

Requires an authenticated Kaggle API and the two named private/public inputs.
No credentials are copied into the notebook. Use --smoke for a small cloud check.
"""
from pathlib import Path
import argparse,json,tempfile
from datetime import datetime,timezone
import nbformat
ROOT=Path(__file__).resolve().parents[1]


def build_submission(folder,kernel,smoke=False):
    source=nbformat.read(ROOT/'notebooks/kaggle_v2/06_full_experiment.ipynb',4)
    program='\n\n'.join(c.source for c in source.cells if c.cell_type=='code')
    if smoke:program=program.replace('SMOKE_TEST = False','SMOKE_TEST = True')
    requirement=(ROOT/'requirements/kaggle.txt').read_text()
    runtime_source=(ROOT/'handgesture/runtime.py').read_text()
    cell=f'''from pathlib import Path
import sys,os,time,types,json
_runtime=types.ModuleType('hg_runtime')
exec(compile({runtime_source!r},'handgesture/runtime.py','exec'),_runtime.__dict__)
run_logged,stage,progress=_runtime.run_logged,_runtime.stage,_runtime.progress
# One absolute deadline includes package installation, compute and export.
deadline=time.monotonic()+21600
os.environ['HG_KAGGLE']='1'
os.environ['HG_REQUIRE_CUDA']='1'
os.environ['HG_PROGRESS_PATH']='/kaggle/working/progress.json'
os.environ['PYTHONUNBUFFERED']='1'
log=Path('/kaggle/working/execution.log')
os.environ['HG_EVENT_LOG_PATH']=str(log)
os.environ['HG_EVENT_LOG_OWNER_PID']=str(os.getpid())
def run(command):
    run_logged(command,log,deadline)
try:
    with stage('install_uv'):
        run([sys.executable,'-m','pip','install','uv'])
    runtime=Path('/tmp/handgesture-python311')
    with stage('create_runtime'):
        run([sys.executable,'-m','uv','venv','--python','3.11',str(runtime)])
    python=runtime/'bin/python'
    requirements=Path('/tmp/handgesture-kaggle.txt');requirements.write_text({requirement!r})
    with stage('install_dependencies'):
        run([sys.executable,'-m','uv','pip','install','--python',str(python),'-r',str(requirements)])
    with stage('install_torch'):
        run([sys.executable,'-m','uv','pip','install','--python',str(python),
             'torch==2.11.0','torchvision==0.26.0','--index-url','https://download.pytorch.org/whl/cu128'])
    program=Path('/kaggle/working/experiment_v2.py');program.write_text({program!r})
    with stage('environment'):
        offset=log.stat().st_size if log.exists() else 0
        run([sys.executable,'-m','uv','pip','freeze','--python',str(python)])
        with log.open('rb') as stream:
            stream.seek(offset)
            Path('/kaggle/working/environment.freeze.txt').write_bytes(stream.read())
    # Neural-network training validates CUDA in the isolated runtime before preprocessing.
    run([str(python),'-u',str(program)])
except BaseException as exc:
    progress(status='timeout' if isinstance(exc,TimeoutError) else 'error',
             error=f'{{type(exc).__name__}}: {{exc}}')
    raise
'''
    notebook=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('# Private reproducible v2 experiment\nPython3.11 / MediaPipe0.10.21 / CUDA PyTorch2.11.'),nbformat.v4.new_code_cell(cell)],
            metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    nbformat.write(notebook,folder/'experiment.ipynb')
    owner,slug=kernel.split('/')
    metadata={'id':kernel,'title':slug.replace('-',' ').title(),'code_file':'experiment.ipynb',
              'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,
              'dataset_sources':['thanhmay2406/hand-gesture','thanhmay2406/hand-gesture-v2-runtime-assets-20261015'],
              'kernel_sources':[],'competition_sources':[]}
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kernel',default='thanhmay2406/hand-gesture-v2-full-20261015')
    p.add_argument('--smoke',action='store_true');p.add_argument('--build-only',type=Path)
    p.add_argument('--run-record',type=Path,help='Separate submission metadata file; smoke defaults to kaggle_smoke_run.json')
    a=p.parse_args();folder=a.build_only or Path(tempfile.mkdtemp(prefix='hg-v2-kaggle-'))
    folder.mkdir(parents=True,exist_ok=True);build_submission(folder,a.kernel,a.smoke)
    if a.build_only:print(folder);return
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate();result=api.kernels_push(str(folder),timeout='21600',acc='nvidiaTeslaT4')
    if getattr(result,'error',None):raise RuntimeError(result.error)
    # Response can contain fields beyond URL; never persist authentication.
    status={'kernel':a.kernel,'url':getattr(result,'url',None),'status':'submitted','smoke_test':a.smoke,
            'version_number':getattr(result,'version_number',None),'kernel_id':getattr(result,'kernel_id',None),
            'submission_dir':str(folder),'runtime':'Python3.11 / mediapipe0.10.21 / torch2.11 CUDA12.8',
            'note':'No quality/FPS claims until downloaded outputs verify and CPU benchmark completes.'}
    status['submitted_at_utc']=datetime.now(timezone.utc).isoformat()
    record=a.run_record or ROOT/'docs/implementation'/('kaggle_smoke_run.json' if a.smoke else 'kaggle_run.json')
    record.parent.mkdir(parents=True,exist_ok=True)
    record.write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status,indent=2))

if __name__=='__main__':main()
