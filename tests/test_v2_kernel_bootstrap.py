"""Native Jupyter kernel switching retains the connection and removes cross-version paths."""
from pathlib import Path
import importlib
import json
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def test_restart_spec_rejects_unprepared_kernel(tmp_path):
    module=importlib.import_module('handgesture.kernel_bootstrap')
    connection=tmp_path/'connection.json';connection.write_text('{}')
    with pytest.raises(ValueError,match='setup'):
        module.native_restart_spec(tmp_path/'runtime',connection)


def test_restart_spec_retains_connection_and_removes_host_python_paths(tmp_path,monkeypatch):
    module=importlib.import_module('handgesture.kernel_bootstrap')
    runtime=tmp_path/'runtime';(runtime/'bin').mkdir(parents=True)
    python=runtime/'bin/python';python.write_text('fixture executable')
    (runtime/'native_ready.json').write_text(json.dumps({'python':str(python)}))
    connection=tmp_path/'connection.json';connection.write_text('{}')
    monkeypatch.setenv('PYTHONHOME','/wrong/python313')
    monkeypatch.setenv('PYTHONPATH','/wrong/python313/site-packages')
    args,env=module.native_restart_spec(runtime,connection)
    assert args==[str(python),'-m','ipykernel_launcher','-f',str(connection)]
    assert 'PYTHONHOME' not in env and 'PYTHONPATH' not in env
    assert env['VIRTUAL_ENV']==str(runtime) and env['PATH'].split(':')[0]==str(runtime/'bin')
    assert connection.read_text()=='{}'


def test_manual_notebook_setup_is_explicit_and_keeps_native_training():
    import nbformat
    path=Path(__file__).resolve().parents[1]/'notebooks/kaggle_v2/07_interactive_workflow.ipynb'
    n=nbformat.read(path,4)
    config=next(c.source for c in n.cells if c.metadata.get('step')=='configuration')
    assert 'PREPARE_COMPATIBLE_KERNEL = False' in config
    assert 'SWITCH_COMPATIBLE_KERNEL = False' in config
    assert any(c.metadata.get('step')=='kernel_setup' for c in n.cells)
    assert any(c.metadata.get('step')=='kernel_switch' for c in n.cells)
