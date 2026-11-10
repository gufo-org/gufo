import os,json,subprocess,re,collections,sys
SP=sys.argv[1]
hip=json.load(open(f'{SP}/prof/hip.json'))['kernels']
keys=[json.loads(l) for l in open(f'{SP}/prof/hrx-dispatch.jsonl')]
keys=[r for r in keys if r['type']=='dispatch_group']
dem=subprocess.run([os.environ.get('CXXFILT','c++filt')],input='\n'.join(r['key'] for r in keys),capture_output=True,text=True).stdout.split('\n')
def norm(s):
  if s.startswith('void '): s=s[5:]
  s=s.replace('(anonymous namespace)::','')
  s=re.sub(r'\(.*$','',s).replace('gufo::','')
  s=re.sub(r'<.*$','',s).split('::')[-1]
  return s
hrx=collections.defaultdict(lambda:[0,0.0])
for r,d in zip(keys,dem):
  n=norm(d); hrx[n][0]+=r['count']; hrx[n][1]+=r['total_ns']/1e6
hipn=collections.defaultdict(lambda:[0,0.0])
for k,v in hip.items():
  hipn[norm(k)][0]+=v['calls']; hipn[norm(k)][1]+=v['ms']
rows=[];used=set()
for hk,(hc,hms) in hipn.items():
  m=[x for x in hrx if x not in used and x==hk]
  if len(m)==1: used.add(m[0]); rows.append((hk,hc,hms,*hrx[m[0]]))
  else: rows.append((hk,hc,hms,None,None))
rows.sort(key=lambda r:-r[2])
print(f"{'kernel':64s} {'calls':>6s} {'hip ms':>8s} {'hrx ms':>8s} {'d ms':>7s} {'d%':>6s}")
for k,hc,hms,xc,xms in rows[:int(sys.argv[2]) if len(sys.argv)>2 else 25]:
  if xms is None: print(f"{k[:64]:64s} {hc:6d} {hms:8.1f}      n/a"); continue
  print(f"{k[:64]:64s} {hc:6d} {hms:8.1f} {xms:8.1f} {xms-hms:7.1f} {100*(xms/hms-1):6.1f} {'' if hc==xc else 'calls '+str(xc)}")
m=[r for r in rows if r[4] is not None]
print('matched: hip %.1f hrx %.1f | unmatched hip %.1f (%d) hrx %.1f (%d)'%(sum(r[2] for r in m),sum(r[4] for r in m),sum(r[2] for r in rows if r[4] is None),sum(1 for r in rows if r[4] is None),sum(v[1] for k,v in hrx.items() if k not in used),sum(1 for k in hrx if k not in used)))
