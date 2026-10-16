"""Explicit setup and in-place restart of a native Jupyter kernel on Kaggle."""
from pathlib import Path
import json
import os
import shlex
import sys


def prepare_native_kernel(runtime,requirements,ipython):
    """Installation only. Training stays in the notebook kernel after restart."""
    runtime=Path(runtime).resolve();python=runtime/'bin/python'
    ipython.run_line_magic('pip','install uv')
    if ipython.user_ns.get('_exit_code',0)!=0:
        raise RuntimeError('uv installation failed; inspect the pip output')
    def command(args):
        value=shlex.join([str(a) for a in args])
        print('SETUP:',value,flush=True)
        ipython.system(value)
        if ipython.user_ns.get('_exit_code')!=0:
            raise RuntimeError('Setup failed; inspect the command output above')
    if not python.is_file():
        command([sys.executable,'-m','uv','venv','--python','3.11',runtime])
    packages=[*requirements,'ipykernel==6.29.5']
    reqfile=runtime/'requirements.kaggle.txt';reqfile.write_text(chr(10).join(packages)+chr(10))
    command([sys.executable,'-m','uv','pip','install','--python',python,'-r',reqfile])
    command([sys.executable,'-m','uv','pip','install','--python',python,
             'torch==2.11.0','torchvision==0.26.0','--index-url','https://download.pytorch.org/whl/cu128'])
    check=("import sys,torch,mediapipe,importlib.metadata as m; "
           "assert sys.version_info[:2]==(3,11); "
           "assert all(m.version(p.split('==')[0])==p.split('==')[1] for p in "+repr(packages)+"); "
           "assert torch.cuda.is_available(); x=torch.ones((8,8),device='cuda'); "
           "assert bool(torch.all(x@x==8)); torch.cuda.synchronize(); "
           "print('NATIVE_READY',sys.version,torch.__version__,mediapipe.__version__,torch.cuda.get_device_name())")
    command([python,'-u','-c',check])
    (runtime/'native_ready.json').write_text(json.dumps({'python':str(python),'requirements':packages},indent=2))
    print('Setup complete. Run the separate kernel-switch cell, then rerun configuration/dependencies.',flush=True)


def native_restart_spec(runtime,connection_file):
    runtime=Path(runtime).resolve();python=runtime/'bin/python';marker=runtime/'native_ready.json'
    if not marker.is_file() or not python.is_file():
        raise ValueError('Run the native Python setup cell successfully before switching kernel')
    ready=json.loads(marker.read_text())
    if ready.get('python')!=str(python):raise ValueError('Native setup marker does not match Python path')
    connection=Path(connection_file).resolve()
    if not connection.is_file():raise ValueError('Current Jupyter connection file is missing')
    env=os.environ.copy()
    env.pop('PYTHONHOME',None);env.pop('PYTHONPATH',None)
    env['VIRTUAL_ENV']=str(runtime)
    env['PATH']=str(runtime/'bin')+os.pathsep+env.get('PATH','')
    return [str(python),'-m','ipykernel_launcher','-f',str(connection)],env


def switch_native_kernel(runtime):
    from ipykernel.connect import get_connection_file
    args,env=native_restart_spec(runtime,get_connection_file())
    print('Switching THIS notebook kernel to Python 3.11. Rerun from the configuration cell after reconnection.',flush=True)
    sys.stdout.flush();sys.stderr.flush()
    os.execve(args[0],args,env)
