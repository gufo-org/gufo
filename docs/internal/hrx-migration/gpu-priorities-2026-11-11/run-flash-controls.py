import datetime,json,os,pathlib,subprocess,time
out=pathlib.Path('/home/mixer/hrx-validation/flash-next-hip-hrx-current');out.mkdir(exist_ok=True)
model='/persist/models/store/Qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf'
temp=pathlib.Path('/sys/class/drm/card1/device/hwmon')
def temperature():
 return int(next(temp.glob('hwmon*/temp1_input')).read_text())
def competitors():
 rows=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
 return [r for r in rows if any(x in r for x in ('/bin/gufo serve ','/bin/gufo bench ','/bin/llama-server ','/bin/loom-serve '))]
for label in ['hip','hrx']:
 if competitors(): raise SystemExit('competing model process: '+str(competitors()))
 while temperature()>38000:time.sleep(5)
 time.sleep(60)
 if competitors(): raise SystemExit('competing model process: '+str(competitors()))
 profile=pathlib.Path('/sys/firmware/acpi/platform_profile').read_text().strip()
 cmd=['systemd-run','--user','--scope','-q','-p','MemoryMax=118G','-p','TasksMax=256','--',('/nix/store/5mv5s936ivprp0skxmk156rmnbmb0rwi-gufo-0-unstable-9612db2/bin/gufo' if label == 'hip' else '/home/mixer/hrx-validation/flash-next-candidate/bin/gufo'),'bench','-m',model,'-p','512,2048','-n','128','-r','3','--seed','1','--validate-prefill','512']
 meta={'label':label,'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'start_temperature_millic':temperature(),'profile':profile,'command':cmd}
 (out/f'{label}-metadata.json').write_text(json.dumps(meta,indent=2))
 print('starting '+label,flush=True)
 with (out/f'{label}.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 meta.update(returncode=r.returncode,end_temperature_millic=temperature(),end_profile=pathlib.Path('/sys/firmware/acpi/platform_profile').read_text().strip())
 (out/f'{label}-metadata.json').write_text(json.dumps(meta,indent=2))
 print('finished '+label+' exit '+str(r.returncode),flush=True)
 if r.returncode:raise SystemExit(r.returncode)
