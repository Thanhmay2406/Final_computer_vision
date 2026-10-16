#!/usr/bin/env python3
"""Bounded remote monitoring; remote execution and local liveness are separate."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.fetch_kaggle_v2 import kernel_state
from handgesture.runtime import atomic_json, utc_now, _stop_group, run_logged
ROOT=Path(__file__).resolve().parents[1]
CANCELLED={'CANCELLED','CANCELED','CANCEL_ACKNOWLEDGED','CANCEL_REQUESTED'}
TERMINAL={'ready','error','cancelled','connection_error','retrieval_error','stopped','timeout'}


class WatcherInterrupted(Exception):
    pass


class LogStreamer:
    """A silent SSE connection can only occupy its own short-lived process."""
    def __init__(self,command,log_path,timeout_seconds=60):
        self.command=command;self.log_path=Path(log_path);self.timeout=timeout_seconds
        self.process=None;self.started=0;self.last_timed_out_pid=None;self._log=None
    def tick(self):
        if self.process is not None:
            if self.process.poll() is None and time.monotonic()-self.started<self.timeout:
                return
            if self.process.poll() is None:self.last_timed_out_pid=self.process.pid
            self.close()
        self.log_path.parent.mkdir(parents=True,exist_ok=True)
        self._log=self.log_path.open('a',buffering=1)
        self.process=subprocess.Popen(self.command,stdout=self._log,stderr=subprocess.STDOUT,
                                      start_new_session=True)
        self.started=time.monotonic()
    def close(self):
        if self.process is not None:
            _stop_group(self.process);self.process=None
        if self._log is not None:self._log.close();self._log=None


def watcher_health(state_path,max_age_seconds=180):
    state=json.loads(Path(state_path).read_text())
    heartbeat=state.get('heartbeat_at_utc',state.get('updated_at_utc'))
    age=(datetime.now(timezone.utc)-datetime.fromisoformat(heartbeat)).total_seconds() if heartbeat else None
    pid=state.get('pid');alive=False
    if isinstance(pid,int) and pid>0:
        try:os.kill(pid,0);alive=True
        except ProcessLookupError:pass
        except PermissionError:alive=True
    status=state.get('watcher_state',state.get('state'))
    health='finished' if status in TERMINAL else 'dead' if not alive else 'stale' if age is None or age>max_age_seconds else 'alive'
    return {**state,'health':health,'process_alive':alive,'heartbeat_age_seconds':age}


def monitor(api,kernel,state_path,on_complete,poll_seconds=60,timeout_seconds=25200,
            retry_base_seconds=1,max_retries=5,streamer=None):
    state_path=Path(state_path);start=time.monotonic();lock=threading.RLock();done=threading.Event()
    state={'kernel':kernel,'pid':os.getpid(),'watcher_started_at_utc':utc_now(),
           'remote_state':None,'last_successful_poll_at_utc':None,'retry_count':0,'error':None}
    def save(status=None,**extra):
        with lock:
            if status is not None:state.update(state=status,watcher_state=status)
            state.update(extra,heartbeat_at_utc=utc_now(),updated_at_utc=utc_now())
            atomic_json(state_path,state)
    def heartbeat():
        while not done.wait(10):save()
    def pause(seconds):
        # Keep streaming timeout enforcement independent of long polling intervals.
        end=min(time.monotonic()+seconds,start+timeout_seconds)
        while time.monotonic()<end:
            if streamer:streamer.tick()
            time.sleep(min(1,max(0,end-time.monotonic())))
    save('monitoring')
    pulse=threading.Thread(target=heartbeat,daemon=True);pulse.start()
    try:
        while time.monotonic()-start<timeout_seconds:
            if streamer:streamer.tick()
            try:
                remote=kernel_state(api.kernels_status(kernel))
            except (OSError,TimeoutError) as exc:
                retries=state['retry_count']
                if retries>=max_retries:
                    save('connection_error',error=f'{type(exc).__name__}: {exc}')
                    return 'connection_error'
                retries+=1
                save('retrying',retry_count=retries,error=f'{type(exc).__name__}: {exc}')
                pause(min(60,retry_base_seconds*2**(retries-1)))
                continue
            save('monitoring',remote_state=remote,last_successful_poll_at_utc=utc_now(),retry_count=0,error=None)
            if remote=='COMPLETE':
                save('retrieving')
                try:on_complete()
                except WatcherInterrupted:raise
                except Exception as exc:
                    save('retrieval_error',error=f'{type(exc).__name__}: {exc}')
                    raise
                save('ready',note='Full outputs verified and classifier benchmark completed; webcam/video evaluation remains separate')
                return 'ready'
            # CANCEL_REQUESTED is not yet proof that compute has stopped.
            if remote=='ERROR' or remote in CANCELLED-{'CANCEL_REQUESTED'}:
                status='error' if remote=='ERROR' else 'cancelled'
                save(status)
                try:
                    logs=api.kernels_logs(kernel)
                    state_path.with_suffix('.log').write_text(logs if isinstance(logs,str) else json.dumps(logs))
                except Exception as exc:save(status,log_error=f'{type(exc).__name__}: {exc}')
                return status
            pause(poll_seconds)
        save('timeout',error='Local monitoring deadline expired; remote run was not cancelled')
        return 'timeout'
    except WatcherInterrupted as exc:
        save('stopped',error=str(exc));return 'stopped'
    except BaseException as exc:
        if state.get('watcher_state')!='retrieval_error':save('error',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        done.set();pulse.join(timeout=1)
        if streamer:streamer.close()


class BoundedKaggleAPI:
    """The SDK has no per-call timeout; isolate network calls from the watcher."""
    def __init__(self,timeout_seconds=30):self.timeout=timeout_seconds
    def _query(self,kernel,kind):
        with tempfile.TemporaryDirectory(prefix='hg-kaggle-query-') as folder:
            result=Path(folder)/'response.json'
            try:
                completed=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--kernel',kernel,
                    '--query',kind,'--response',str(result)],capture_output=True,text=True,timeout=self.timeout)
            except subprocess.TimeoutExpired as exc:raise TimeoutError(f'Kaggle {kind} query exceeded {self.timeout}s') from exc
            if completed.returncode or not result.exists():raise OSError(f'Kaggle {kind} query failed: {completed.stderr[-1500:]}')
            value=json.loads(result.read_text())
            return value
    def kernels_status(self,kernel):return SimpleNamespace(**self._query(kernel,'status'))
    def kernels_logs(self,kernel):return self._query(kernel,'logs')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kernel',default='thanhmay2406/hand-gesture-v2-full-20261015')
    p.add_argument('--output-root',type=Path,default=ROOT/'results_v2/kaggle_20261015')
    p.add_argument('--data-root',type=Path,default=ROOT/'data_v2_kaggle')
    p.add_argument('--state',type=Path,default=ROOT/'docs/implementation/watcher_state.json')
    p.add_argument('--check',action='store_true',help='Read PID/heartbeat health without contacting Kaggle')
    p.add_argument('--stream-log',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--query',choices=['status','logs'],help=argparse.SUPPRESS)
    p.add_argument('--response',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.check:
        print(json.dumps(watcher_health(a.state),indent=2));return
    if a.stream_log or a.query:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi();api.authenticate()
        if a.stream_log:
            for event in api.kernels_logs_stream(a.kernel):
                print(event.get('data',''),end='',flush=True)
        else:
            value=api.kernels_status(a.kernel) if a.query=='status' else api.kernels_logs(a.kernel)
            if a.query=='status':value={'status':kernel_state(value)}
            atomic_json(a.response,value)
        return
    def interrupted(signum,frame):raise WatcherInterrupted(signal.Signals(signum).name)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    watcher_deadline=time.monotonic()+25200
    retrieval_log=a.state.with_name(a.state.stem+'_retrieval.log')
    def complete():
        run_logged([sys.executable,str(ROOT/'scripts/fetch_kaggle_v2.py'),
             '--kernel',a.kernel,'--output-root',str(a.output_root),'--data-root',str(a.data_root)],retrieval_log,watcher_deadline)
        run_logged([sys.executable,str(ROOT/'scripts/benchmark_v2.py'),
             '--results-root',str(a.output_root/'experiment/final'),'--data-root',str(a.data_root),
             '--output-dir',str(a.output_root/'cpu_benchmark')],retrieval_log,watcher_deadline)
    stream=LogStreamer([sys.executable,'-u',str(Path(__file__).resolve()),'--kernel',a.kernel,'--stream-log'],
                       a.state.with_name(a.state.stem+'_stream.log'))
    result=monitor(BoundedKaggleAPI(),a.kernel,a.state,complete,streamer=stream,
                   timeout_seconds=max(0,watcher_deadline-time.monotonic()))
    print('Watcher:',result,flush=True)
    if result!='ready':raise SystemExit(1)


if __name__=='__main__':main()
