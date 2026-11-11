import pathlib,subprocess,os,json,re
out=pathlib.Path('/home/mixer/hrx-validation/qwen-bench-sync-tests')
binary='/home/mixer/hrx-validation/qwen-bench-sync-candidate/bin/gufo'
model='/persist/models/store/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf'
runs=[]
for runtime in ['hip','hrx']:
 base=os.environ.copy()
 if runtime=='hrx':base['LD_LIBRARY_PATH']='/nix/store/2z5yrfk8swk40hnxbyriz6klhxml0gc9-hrx-system-0-unstable-2026-11-11/lib'
 for fail in [0,2,3,4,5]:
  env=base.copy();env['LD_PRELOAD']='/home/mixer/hrx-validation/qwen_sync_fault.so';env['GUFO_SYNC_FAIL_AT']=str(fail)
  cmd=[binary,'bench','-m',model,'-p','32','-n','0','-r','1']
  r=subprocess.run(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
  tag=f'{runtime}-fail-{fail}';(out/(tag+'.log')).write_text(r.stdout)
  passed=(r.returncode==0 and 'HIP error:' not in r.stdout and '|            pp32 |' in r.stdout) if not fail else (r.returncode==1 and 'HIP error: invalid argument' in r.stdout and '|            pp32 |' not in r.stdout and f'call={fail} real=0 injected=1' in r.stdout)
  runs.append({'tag':tag,'command':cmd,'returncode':r.returncode,'passed':passed,'LD_LIBRARY_PATH':env.get('LD_LIBRARY_PATH')});(out/'tests.json').write_text(json.dumps(runs,indent=2)+'\n');print(tag,r.returncode,passed,flush=True)
  if not passed:raise SystemExit('failed '+tag)
 cmd=[binary,'bench','-m',model,'-p','32','-n','8','-d','0,32','-r','1','--verbose','--validate-prefill','32']
 r=subprocess.run(cmd,env=base,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=90)
 tag=runtime+'-depth-decode';(out/(tag+'.log')).write_text(r.stdout)
 passed=r.returncode==0 and 'HIP error:' not in r.stdout and len(re.findall(r'\[QwenBenchTrace\]',r.stdout))==2 and 'finite=yes' in r.stdout
 runs.append({'tag':tag,'command':cmd,'returncode':r.returncode,'passed':passed,'LD_LIBRARY_PATH':base.get('LD_LIBRARY_PATH')});(out/'tests.json').write_text(json.dumps(runs,indent=2)+'\n');print(tag,r.returncode,passed,flush=True)
 if not passed:raise SystemExit('failed '+tag)
