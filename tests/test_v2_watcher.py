"""Cloud output retrieval is triggered only on a successful terminal state."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
from scripts.watch_kaggle_v2 import monitor


class API:
    def __init__(self,states):self.states=iter(states)
    def kernels_status(self,kernel):return SimpleNamespace(status=next(self.states))
    def kernels_logs(self,kernel):return 'remote failure details'


def test_monitor_fetches_only_after_complete_and_persists_state(tmp_path):
    output=tmp_path/'done'
    result=monitor(API(['RUNNING','COMPLETE']),'test/kernel',tmp_path/'state.json',
                   lambda:output.write_text('verified'),poll_seconds=0)
    assert result=='ready' and output.read_text()=='verified'
    assert 'ready' in (tmp_path/'state.json').read_text()


def test_remote_error_does_not_publish_ready_outputs(tmp_path):
    output=tmp_path/'done'
    result=monitor(API(['ERROR']),'test/kernel',tmp_path/'state.json',
                   lambda:output.write_text('invalid'),poll_seconds=0)
    assert result=='error' and not output.exists()
    assert 'remote failure details' in (tmp_path/'state.log').read_text()


def test_network_retry_recovers_and_retains_last_success(tmp_path):
    import json
    class Flaky(API):
        def __init__(self):self.calls=0
        def kernels_status(self,kernel):
            self.calls+=1
            if self.calls<3:raise OSError('offline')
            return SimpleNamespace(status='COMPLETE')
    api=Flaky();path=tmp_path/'state.json'
    result=monitor(api,'test/kernel',path,lambda:None,poll_seconds=0,retry_base_seconds=0)
    state=json.loads(path.read_text())
    assert result=='ready' and api.calls==3
    assert state['remote_state']=='COMPLETE' and state['watcher_state']=='ready'
    assert state['heartbeat_at_utc'] and state['last_successful_poll_at_utc']


def test_network_retry_exhaustion_records_error(tmp_path):
    import json
    class Offline(API):
        calls=0
        def kernels_status(self,kernel):self.calls+=1;raise OSError('offline')
    api=Offline([]);path=tmp_path/'state.json'
    result=monitor(api,'test/kernel',path,lambda:None,poll_seconds=0,retry_base_seconds=0)
    assert result=='connection_error' and api.calls==6
    state=json.loads(path.read_text())
    assert state['remote_state'] is None and state['retry_count']==5
    assert 'offline' in state['error']


def test_cancel_acknowledged_never_fetches_even_if_logs_fail(tmp_path):
    import json
    class Cancelled(API):
        def kernels_logs(self,kernel):raise OSError('log unavailable')
    path=tmp_path/'state.json'
    def unexpected():raise AssertionError('must not fetch')
    assert monitor(Cancelled(['CANCEL_ACKNOWLEDGED']),'t/k',path,unexpected,poll_seconds=0)=='cancelled'
    state=json.loads(path.read_text())
    assert state['remote_state']=='CANCEL_ACKNOWLEDGED' and 'log unavailable' in state['log_error']


def test_callback_failure_never_publishes_ready(tmp_path):
    import json,pytest
    path=tmp_path/'state.json'
    def failed():raise ValueError('artifact hash mismatch')
    with pytest.raises(ValueError,match='hash'):
        monitor(API(['COMPLETE']),'t/k',path,failed,poll_seconds=0)
    state=json.loads(path.read_text())
    assert state['watcher_state']=='retrieval_error' and state['remote_state']=='COMPLETE'


def test_timeout_and_interruption_are_saved(tmp_path):
    import json
    from scripts.watch_kaggle_v2 import WatcherInterrupted
    path=tmp_path/'state.json'
    assert monitor(API([]),'t/k',path,lambda:None,timeout_seconds=0)=='timeout'
    assert json.loads(path.read_text())['watcher_state']=='timeout'
    class Interrupted(API):
        def kernels_status(self,kernel):raise WatcherInterrupted('SIGTERM')
    assert monitor(Interrupted([]),'t/k',path,lambda:None,poll_seconds=0)=='stopped'
    assert json.loads(path.read_text())['watcher_state']=='stopped'


def test_health_detects_dead_pid_and_stale_heartbeat(tmp_path):
    import json,os
    from scripts.watch_kaggle_v2 import watcher_health
    path=tmp_path/'state.json'
    path.write_text(json.dumps({'pid':os.getpid(),'heartbeat_at_utc':'2000-01-01T00:00:00+00:00','watcher_state':'monitoring'}))
    assert watcher_health(path)['health']=='stale'
    path.write_text(json.dumps({'pid':999999999,'heartbeat_at_utc':'2000-01-01T00:00:00+00:00','watcher_state':'monitoring'}))
    assert watcher_health(path)['health']=='dead'


def test_silent_log_stream_cannot_block_monitor(tmp_path):
    import sys,time
    from scripts.watch_kaggle_v2 import LogStreamer
    stream=LogStreamer([sys.executable,'-c','import time;time.sleep(60)'],tmp_path/'stream.log',timeout_seconds=.1)
    stream.tick();pid=stream.process.pid
    time.sleep(.15);stream.tick()
    assert stream.last_timed_out_pid==pid
    stream.close()


def test_bounded_adapter_response_can_be_consumed_by_monitor(tmp_path,monkeypatch):
    from scripts.watch_kaggle_v2 import BoundedKaggleAPI
    api=BoundedKaggleAPI()
    monkeypatch.setattr(api,'_query',lambda kernel,kind:{'status':'COMPLETE'})
    assert monitor(api,'t/k',tmp_path/'state.json',lambda:None,poll_seconds=0)=='ready'
