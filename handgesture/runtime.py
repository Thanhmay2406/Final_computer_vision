"""Small shared runtime controls for observable, bounded Kaggle execution."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import codecs
import json
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import time


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def progress(**fields):
    """Only actual milestones call this; a watchdog heartbeat is not progress."""
    path = os.environ.get('HG_PROGRESS_PATH')
    value = {}
    if path and Path(path).exists():
        value = json.loads(Path(path).read_text())
    value.update(fields)
    value.update(last_progress_at_utc=utc_now(), pid=os.getpid())
    if path:
        atomic_json(path, value)
    line=json.dumps(value, ensure_ascii=False, allow_nan=False)
    event_log=os.environ.get('HG_EVENT_LOG_PATH')
    if event_log and os.environ.get('HG_EVENT_LOG_OWNER_PID')==str(os.getpid()):
        event_path=Path(event_log);event_path.parent.mkdir(parents=True,exist_ok=True)
        with event_path.open('a') as stream:stream.write(line+'\n')
    print(line, flush=True)
    return value


@contextmanager
def stage(name, model=None):
    start = time.monotonic()
    progress(stage=name, model=model, status='running', completed=0, total=None,
             epoch=None, error=None, started_at_utc=utc_now())
    try:
        yield
    except BaseException as exc:
        progress(stage=name, model=model, status='timeout' if isinstance(exc,TimeoutError) else 'error',
                 error=f'{type(exc).__name__}: {exc}', elapsed_seconds=time.monotonic()-start)
        raise
    else:
        progress(stage=name, model=model, status='complete', elapsed_seconds=time.monotonic()-start)


def validate_workers(workers):
    if workers < 0:
        raise ValueError('workers must be nonnegative')
    if os.environ.get('HG_KAGGLE') == '1' and workers:
        raise ValueError('Kaggle runtime requires workers=0; parallel workers are not supported safely')


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for Kaggle neural-network training; refusing CPU fallback')
    # Test allocation, kernel execution and synchronization, not only driver discovery.
    x = torch.ones((8,8), device='cuda')
    result = x @ x
    torch.cuda.synchronize()
    if not bool(torch.all(result == 8).item()):
        raise RuntimeError('CUDA compute preflight failed')
    progress(gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda)


def _stop_group(process):
    """Kill descendants too, even if the group leader has already exited."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    time.sleep(.2)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_logged(command, log_path, deadline, env=None):
    """Tee combined output immediately; deadline is absolute monotonic time."""
    if time.monotonic() >= deadline:
        raise TimeoutError('Experiment deadline exceeded before command start')
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               start_new_session=True, env=env)
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    try:
        with log_path.open('a', buffering=1) as log, selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map() or process.poll() is None:
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Experiment deadline exceeded; terminating subprocess group')
                for key, _ in selector.select(min(.1,remaining)):
                    chunk = os.read(key.fileobj.fileno(),65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    value = decoder.decode(chunk)
                    sys.stdout.write(value);sys.stdout.flush()
                    log.write(value);log.flush()
            tail = decoder.decode(b'',final=True)
            if tail:
                sys.stdout.write(tail);sys.stdout.flush();log.write(tail)
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    except BaseException:
        _stop_group(process)
        raise
    finally:
        process.stdout.close()
