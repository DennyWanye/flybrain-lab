import sys,os,time,json,subprocess,threading,queue
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from flydrone.shadow.__main__ import fixture
from flydrone.shadow.recording import verify_shadow
OUT=ROOT/'reports/shadow_v1/live-stdin-probe';OUT.mkdir(exist_ok=False)
profile=ROOT/'reports/shadow_v1/cli-probe/profile.json'
processes=[]
for mode in ('valid','stall','backlog'):
    target=OUT/mode
    cmd=[sys.executable,'-m','flydrone.shadow','shadow','--root',str(ROOT),'--input','-','--profile',str(profile),'--output',str(target),'--idle-timeout-s','1']
    with (OUT/f'{mode}.stdout.log').open('x') as stdout,(OUT/f'{mode}.stderr.log').open('x') as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.PIPE,stdout=stdout,stderr=subprocess.PIPE,text=True)
        ready=queue.Queue();entry={'pid':proc.pid,'cmd':cmd,'cwd':str(ROOT)};processes.append(entry)
        (OUT/'processes.json').write_text(json.dumps(processes,indent=2))
        def consume():
            for line in proc.stderr:
                log.write(line);log.flush()
                if 'SHADOW_INPUT_READY' in line:ready.put(True)
            ready.put(False)
        reader=threading.Thread(target=consume,daemon=True);reader.start()
        try:
            assert ready.get(timeout=30), 'process exited before ready'
            start=time.monotonic_ns()+50_000_000
            rows=list(fixture())[:3 if mode=='valid' else 1]
            for i,packet in enumerate(rows):
                tick=start+i*100_000_000
                time.sleep(max(0,(tick-time.monotonic_ns())/1e9))
                if mode=='backlog':tick-=1_000_000_000
                packet['time_ns']=tick
                for key in ('pose','state'):packet[key]['captured_ns']=packet[key]['received_ns']=tick
                proc.stdin.write(json.dumps(packet)+'\n');proc.stdin.flush()
            if mode!='stall':proc.stdin.close()
            code=proc.wait(timeout=10)
            reader.join(timeout=2)
            assert not reader.is_alive()
            if mode=='valid':
                assert code==0
                result=verify_shadow(target/'shadow')
                assert result['samples']==3 and result['blocked_samples']==0
            else:
                assert code==1
                result=json.loads((target/'shadow/result.json').read_text())
                assert result['completed'] is False
                assert result['samples']==(1 if mode=='stall' else 0)
            entry.update(returncode=code,exited=not Path(f'/proc/{proc.pid}').exists(),samples=result['samples'])
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            if not proc.stdin.closed:proc.stdin.close()
            reader.join(timeout=2)
            proc.stderr.close()
    (OUT/'processes.json').write_text(json.dumps(processes,indent=2))
report={'live_normalized_stdin_pass':True,'wall_backlog_stopped_before_neural_step':True,'stall_timeout_stopped_session':True,'hardware_data_used':False,'processes_exited':all(x['exited'] for x in processes),'REAL_FLIGHT_READY':False}
(OUT/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
