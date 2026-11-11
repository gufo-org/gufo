import os,select,pathlib,subprocess,json
out=pathlib.Path('/home/mixer/hrx-validation/qwen-bench-sampling-tests');rows=[]
code='''import importlib.util,sys
spec=importlib.util.spec_from_file_location('qwen_cli', '/home/mixer/gufo-qwen-bench-sync/tests/models/qwen27b/cli_test.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
try:
 m.check_bench(sys.argv[1],'/persist/models/store/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf','/persist/models/store/Qwen3.8-27B-DFlash2/Qwen3.8-27B-DFlash2-Q8_0.gguf')
except AssertionError as e:
 print(str(e));sys.exit(1)
'''
for tag,binary in [('baseline','/home/mixer/hrx-validation/qwen-bench-sync-candidate/bin/gufo'),('hip','/home/mixer/hrx-validation/qwen-bench-sampling-candidate/bin/gufo'),('hrx','/home/mixer/hrx-validation/qwen-bench-sampling-hrx/gufo')]:
 r=subprocess.run(['python3','-c',code,binary],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=240)
 (out/('regression-'+tag+'.log')).write_text(r.stdout)
 passed=(r.returncode==1 and 'did not reject sampling controls' in r.stdout) if tag=='baseline' else r.returncode==0
 rows.append({'tag':tag,'returncode':r.returncode,'expected_failure':tag=='baseline','passed':passed});(out/'regression.json').write_text(json.dumps(rows,indent=2)+'\n');print(tag,r.returncode,passed,flush=True)
 if not passed:raise SystemExit('failed '+tag)
