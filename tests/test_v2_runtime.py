"""Observe progress and terminate real subprocess groups, including silent children."""
import importlib
import json
import os
from pathlib import Path
import sys
import time

import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def runtime():
    return importlib.import_module('handgesture.runtime')


def test_runtime_module_exists():
    assert (Path(__file__).resolve().parents[1]/'handgesture/runtime.py').exists()


def test_progress_preserves_stage_and_records_failure(tmp_path,monkeypatch):
    r=runtime();path=tmp_path/'progress.json';monkeypatch.setenv('HG_PROGRESS_PATH',str(path))
    with pytest.raises(ValueError,match='bad input'):
        with r.stage('training',model='gap'):
            r.progress(completed=2,total=10,epoch=1)
            state=json.loads(path.read_text())
            assert state['stage']=='training' and state['model']=='gap'
            assert state['completed']==2 and state['last_progress_at_utc']
            raise ValueError('bad input')
    state=json.loads(path.read_text())
    assert state['status']=='error' and 'bad input' in state['error']
    assert not list(tmp_path.glob('*.tmp'))


def test_logged_command_streams_before_exit_and_preserves_failure(tmp_path,capsys):
    r=runtime();log=tmp_path/'run.log'
    import threading
    output=[]
    def run():
        try:r.run_logged([sys.executable,'-u','-c',"import time;print('first',flush=True);time.sleep(1);raise RuntimeError('broken')"],log,time.monotonic()+5)
        except Exception as exc:output.append(exc)
    t=threading.Thread(target=run);t.start()
    deadline=time.monotonic()+3
    while time.monotonic()<deadline and (not log.exists() or 'first' not in log.read_text()):time.sleep(.02)
    assert t.is_alive() and 'first' in log.read_text()
    t.join(5)
    assert output and output[0].returncode!=0
    assert 'broken' in log.read_text() and 'first' in capsys.readouterr().out


def test_deadline_kills_silent_grandchild_even_when_parent_exits(tmp_path):
    r=runtime();pidfile=tmp_path/'pid'
    child="import os,time;from pathlib import Path;Path("+repr(str(pidfile))+").write_text(str(os.getpid()));time.sleep(60)"
    parent='import subprocess,sys;subprocess.Popen([sys.executable,"-c",'+repr(child)+'])'
    start=time.monotonic()
    with pytest.raises(TimeoutError):r.run_logged([sys.executable,'-c',parent],tmp_path/'log',start+.8)
    assert time.monotonic()-start<4
    pid=int(pidfile.read_text())
    # A killed orphan may briefly remain a zombie until the system reaps it.
    status=Path(f'/proc/{pid}/stat')
    assert not status.exists() or status.read_text().split()[2]=='Z'


def test_kaggle_rejects_parallel_workers(monkeypatch):
    r=runtime();monkeypatch.setenv('HG_KAGGLE','1')
    with pytest.raises(ValueError,match='workers=0'):r.validate_workers(2)
    r.validate_workers(0)


def test_gpu_preflight_fails_without_cuda(monkeypatch):
    r=runtime();import torch
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='CUDA'):r.require_cuda()


def test_kaggle_entrypoints_reject_workers_before_reading_data(monkeypatch,tmp_path):
    monkeypatch.setenv('HG_KAGGLE','1')
    from handgesture.data import prepare_dataset
    from handgesture.training import train_run
    with pytest.raises(ValueError,match='workers=0'):
        prepare_dataset(tmp_path/'missing',tmp_path/'out',workers=2)
    with pytest.raises(ValueError,match='workers=0'):
        train_run(tmp_path/'missing',tmp_path/'out','cnn',workers=2)


def test_generated_notebooks_default_to_serial_and_embed_runtime(tmp_path):
    import nbformat
    from scripts.submit_kaggle_v2 import build_submission
    build_submission(tmp_path,'owner/test',smoke=True)
    text='\n'.join(c.source for c in nbformat.read(tmp_path/'experiment.ipynb',4).cells)
    assert 'NUM_WORKERS = 0' in text
    assert 'run_logged' in text and 'HG_REQUIRE_CUDA' in text
    assert 'deadline' in text and '21600' in text


def test_wrapper_stage_events_are_also_saved_to_execution_log(tmp_path,monkeypatch):
    r=runtime();events=tmp_path/'execution.log'
    monkeypatch.setenv('HG_EVENT_LOG_PATH',str(events))
    monkeypatch.setenv('HG_EVENT_LOG_OWNER_PID',str(os.getpid()))
    with r.stage('install_dependencies'):pass
    rows=[json.loads(line) for line in events.read_text().splitlines()]
    assert [r['status'] for r in rows]==['running','complete']
    assert all(r['stage']=='install_dependencies' for r in rows)


def test_child_progress_does_not_double_write_parent_tee_log(tmp_path,monkeypatch):
    r=runtime();events=tmp_path/'execution.log'
    monkeypatch.setenv('HG_EVENT_LOG_PATH',str(events))
    monkeypatch.setenv('HG_EVENT_LOG_OWNER_PID','999999999')
    r.progress(stage='child')
    assert not events.exists()
