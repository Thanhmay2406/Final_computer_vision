#!/usr/bin/env python3
"""Build independent Kaggle notebooks from the exact local module sources."""
from pathlib import Path
import json
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
SOURCES=[('handgesture.runtime','handgesture/runtime.py'),
         ('handgesture.preprocessing','handgesture/preprocessing.py'),
         ('handgesture.data','handgesture/data.py'),
         ('scripts.kaggle_notebook_support','scripts/kaggle_notebook_support.py'),
         ('handgesture.models','handgesture/models.py'),
         ('handgesture.training','handgesture/training.py')]


def bootstrap(training=True):
    cells=[nbf.v4.new_code_cell('''import sys, types, os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for package in ['handgesture','scripts']:
    module=types.ModuleType(package);module.__path__=[];sys.modules[package]=module
''')]
    for name,path in SOURCES if training else SOURCES[:3]:
        source=(ROOT/path).read_text()
        code=f"# Shared local source: {path}\n_module = types.ModuleType({name!r})\n_module.__file__ = {path!r}\nsys.modules[{name!r}] = _module\n_source = {source!r}\nexec(compile(_source, {path!r}, 'exec'), _module.__dict__)"
        cells.append(nbf.v4.new_code_cell(code))
    return cells

RESOLVE='''from pathlib import Path
import json, os
SEARCH_ROOT = Path('/kaggle/input')
def input_files(name):
    matches=[]
    for directory,dirs,files in os.walk(SEARCH_ROOT):
        dirs[:]=[d for d in dirs if d not in {'images','__pycache__','.git'}]
        if name in files:matches.append(Path(directory)/name)
    return matches
def find_dataset(derived):
    candidates=[]
    for p in input_files('preparation_report.json'):
        if p.parent.name=='metadata' and ('preprocessing' in json.loads(p.read_text()))==derived:
            candidates.append(p.parent.parent)
    if len(candidates)!=1:raise ValueError(f'Set DATA_ROOT explicitly; candidates: {candidates}')
    return candidates[0]
def find_asset(name):
    matches=input_files(name)
    if len(matches)!=1:raise ValueError(f'Expected one {name}; got {matches}')
    return matches[0]
'''
CONFIG='''DATA_ROOT = None
OUTPUT_ROOT = Path('/kaggle/working/hand_gesture_v2')
DEVICE = 'auto'
SMOKE_TEST = False
NUM_WORKERS = 0
WEIGHTS_PATH = None  # MobileNet IMAGENET1K_V1 .pth; no download/fallback during training
import os
if Path('/kaggle').exists():
    os.environ['HG_KAGGLE']='1'
    os.environ['HG_REQUIRE_CUDA']='1'
    os.environ.setdefault('HG_PROGRESS_PATH','/kaggle/working/progress.json')
'''
INTRO='''Giữ nguyên ID/split nguồn. Data v2 phải có fingerprint preprocessing.
Train và chọn CNN chỉ dùng train/validation. Notebook finalize mới đánh giá test.
Output cũ là baseline lịch sử, không ghi đè. `SMOKE_TEST=True` không phải kết quả đầy đủ.
Gắn dataset/Notebook Output phù hợp và runtime assets (detector + pretrained weights).
'''


def write(name,cells):
    target=ROOT/'notebooks/kaggle_v2'/name;target.parent.mkdir(exist_ok=True)
    notebook=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{
        'display_name':'Python 3','language':'python','name':'python3'}})
    nbf.validate(notebook);nbf.write(notebook,target)


def build():
    write('00_prepare_data.ipynb',[nbf.v4.new_markdown_cell('# Chuẩn bị data v2\n\n'+INTRO),
        nbf.v4.new_code_cell(RESOLVE+CONFIG+"\nDETECTOR_PATH = None\n"),*bootstrap(False),
        nbf.v4.new_code_cell('''from handgesture.data import prepare_dataset
from handgesture.runtime import stage
DATA_ROOT = Path(DATA_ROOT) if DATA_ROOT else find_dataset(False)
DETECTOR_PATH = Path(DETECTOR_PATH) if DETECTOR_PATH else find_asset('hand_landmarker.task')
with stage('preprocessing'):
    frame, report = prepare_dataset(DATA_ROOT, OUTPUT_ROOT/'data_v2', DETECTOR_PATH,
                                     limit=8 if SMOKE_TEST else None, workers=NUM_WORKERS)
print(report['counts'], report['split_counts'])
''')])
    for filename,kind in [('01_random_forest.ipynb','rf'),('02_mlp.ipynb','mlp'),('03_cnn_candidates.ipynb','cnn')]:
        run='''from handgesture.training import train_run
from handgesture.runtime import stage
DATA_ROOT = Path(DATA_ROOT) if DATA_ROOT else find_dataset(True)
'''
        if kind=='cnn':
            run+='''WEIGHTS_PATH = Path(WEIGHTS_PATH) if WEIGHTS_PATH else find_asset('mobilenet_v3_small-047dcff4.pth')
for architecture in ['gap','spatial','mobilenet_v3_small']:
    with stage('training',model=architecture):
        train_run(DATA_ROOT, OUTPUT_ROOT/'candidates'/architecture, 'cnn', architecture,
                  weights_path=WEIGHTS_PATH, smoke=SMOKE_TEST, device=DEVICE, workers=NUM_WORKERS)
'''
        else:run+=f"with stage('training',model={kind!r}):\n    train_run(DATA_ROOT, OUTPUT_ROOT/'runs'/{kind!r}, {kind!r}, smoke=SMOKE_TEST, device=DEVICE, workers=NUM_WORKERS)\n"
        write(filename,[nbf.v4.new_markdown_cell(f'# Huấn luyện {kind.upper()} — validation only\n\n'+INTRO),
                        nbf.v4.new_code_cell(RESOLVE+CONFIG),*bootstrap(),nbf.v4.new_code_cell(run)])
    write('04_finalize_models.ipynb',[nbf.v4.new_markdown_cell('# Chọn CNN và đánh giá test\n\n'+INTRO),
        nbf.v4.new_code_cell(RESOLVE+CONFIG+'''\nRF_DIR = None
MLP_DIR = None
CNN_DIRS = None
'''),*bootstrap(),nbf.v4.new_code_cell('''from handgesture.training import finalize
from handgesture.runtime import stage
DATA_ROOT = Path(DATA_ROOT) if DATA_ROOT else find_dataset(True)
def run_directories(kind):
    return [p.parent for p in SEARCH_ROOT.rglob('run.json')
            if json.loads(p.read_text())['model']==kind]
rf=Path(RF_DIR) if RF_DIR else run_directories('rf')
mlp=Path(MLP_DIR) if MLP_DIR else run_directories('mlp')
if isinstance(rf,list):
    if len(rf)!=1:raise ValueError('Set RF_DIR')
    rf=rf[0]
if isinstance(mlp,list):
    if len(mlp)!=1:raise ValueError('Set MLP_DIR')
    mlp=mlp[0]
cnn=CNN_DIRS or run_directories('cnn')
if len(cnn)!=3:raise ValueError('Select exactly three CNN candidate directories')
with stage('finalize'):
    finalize(DATA_ROOT,rf,mlp,cnn,OUTPUT_ROOT/'final')
''')])
    write('05_compare_models.ipynb',[nbf.v4.new_markdown_cell('# Kiểm chứng và so sánh final outputs\n\n'+INTRO),
        nbf.v4.new_code_cell(RESOLVE+CONFIG+"\nRESULTS_ROOT = None\n"),*bootstrap(),
        nbf.v4.new_code_cell('''from handgesture.training import verify_outputs
from scripts.kaggle_notebook_support import export_comparison
if RESULTS_ROOT is None:
    matches=[p.parent for p in SEARCH_ROOT.rglob('selection.json')]
    if len(matches)!=1:raise ValueError('Set RESULTS_ROOT')
    RESULTS_ROOT=matches[0]
results=verify_outputs(RESULTS_ROOT,DATA_ROOT)
export_comparison(results,OUTPUT_ROOT/'comparison')
''')])
    # One optional end-to-end notebook for a private Save & Run All on Kaggle.
    write('06_full_experiment.ipynb',[nbf.v4.new_markdown_cell('# Thực nghiệm v2 đầy đủ\n\n'+INTRO),
        nbf.v4.new_code_cell(RESOLVE+CONFIG+"\nDETECTOR_PATH = None\n"),*bootstrap(),nbf.v4.new_code_cell('''from handgesture.data import prepare_dataset
from handgesture.training import train_run,finalize
from handgesture.runtime import stage, progress, require_cuda
import os, torch
if os.environ.get('HG_REQUIRE_CUDA')=='1':
    with stage('gpu_check'): require_cuda()
torch.set_num_threads(2)
source=Path(DATA_ROOT) if DATA_ROOT else find_dataset(False)
detector=Path(DETECTOR_PATH) if DETECTOR_PATH else find_asset('hand_landmarker.task')
weights=Path(WEIGHTS_PATH) if WEIGHTS_PATH else find_asset('mobilenet_v3_small-047dcff4.pth')
data=OUTPUT_ROOT/'data_v2'
with stage('preprocessing'):
    prepare_dataset(source,data,detector,limit=8 if SMOKE_TEST else None, workers=NUM_WORKERS)
for name in ['rf','mlp']:
    with stage('training',model=name):
        train_run(data,OUTPUT_ROOT/'runs'/name,name,smoke=SMOKE_TEST,device=DEVICE,workers=NUM_WORKERS)
for architecture in ['gap','spatial','mobilenet_v3_small']:
    with stage('training',model=architecture):
        train_run(data,OUTPUT_ROOT/'candidates'/architecture,'cnn',architecture,weights,
                  smoke=SMOKE_TEST,device=DEVICE,workers=NUM_WORKERS)
with stage('finalize'):
    finalize(data,OUTPUT_ROOT/'runs/rf',OUTPUT_ROOT/'runs/mlp',
         [OUTPUT_ROOT/'candidates'/a for a in ['gap','spatial','mobilenet_v3_small']],OUTPUT_ROOT/'final')
# Archive outputs for reliable downloading; no changes to measured artifacts.
import zipfile,hashlib
from handgesture.data import write_json
archives={}
progress(stage='archive',model=None,status='running',completed=0,total=2)
for name,folder in [('data_v2.zip',data),('results_v2.zip',OUTPUT_ROOT)]:
    target=OUTPUT_ROOT.parent/name
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for path in sorted(folder.rglob('*')):
            relative=path.relative_to(folder)
            if path.is_file() and not (name=='results_v2.zip' and relative.parts[0]=='data_v2'):
                z.write(path,str(relative))
    with target.open('rb') as stream:
        archives[name]=hashlib.file_digest(stream,'sha256').hexdigest()
    progress(stage='archive',archive=name,completed=len(archives),total=2)
write_json(OUTPUT_ROOT.parent/'export_manifest.json',{'archives':archives,'smoke_test':SMOKE_TEST})
progress(stage='experiment',model=None,status='complete',completed=2,total=2)
''')])

if __name__=='__main__':
    build()
    from build_interactive_notebook import build_interactive
    build_interactive()
    print('Built seven legacy v2 notebooks and one interactive workflow notebook')
