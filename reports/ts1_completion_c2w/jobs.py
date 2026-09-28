from pathlib import Path
import subprocess,sys,os,json,time,signal
root=Path.cwd();out=root/'reports/ts1_completion_c2w';job=sys.argv[1]
commands={'regression':[sys.executable,'-m','pytest','tests/flydrone','-q','--junitxml=reports/ts1_completion_c2w/regression.xml'],'observer':[sys.executable,'-u','-m','reports.ts1_completion_c2w.observer_audit'],'package':[sys.executable,'-B','-u','-m','reports.ts1_completion_c2w.package_delivery']}
cmd=commands[job];log=(out/(job+'.log')).open('x');p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);started=time.monotonic();peak=0
(out/(job+'-process.json')).write_text(json.dumps({'pid':p.pid,'parent_pid':os.getpid(),'command':cmd,'cwd':str(root)},indent=2));print(json.dumps({'job':job,'pid':p.pid}),flush=True)
try:
 while p.poll() is None:
  try:peak=max(peak,sum(int(x.split()[1]) for x in Path(f'/proc/{p.pid}/smaps_rollup').read_text().splitlines() if x.startswith(('Private_Clean:','Private_Dirty:'))))
  except (FileNotFoundError,ProcessLookupError):pass
  if time.monotonic()-started>3600:raise TimeoutError(job)
  time.sleep(.5)
finally:
 if p.poll() is None:
  assert os.getpgid(p.pid)==p.pid and os.readlink(f'/proc/{p.pid}/cwd')==str(root)
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
 log.close();(out/(job+'-cleanup.json')).write_text(json.dumps({'pid':p.pid,'returncode':p.returncode,'absent':not Path(f'/proc/{p.pid}').exists(),'sampled_peak_private_kib':peak,'elapsed_s':time.monotonic()-started},indent=2))
print(json.dumps({'job':job,'returncode':p.returncode}),flush=True);sys.exit(p.returncode)
