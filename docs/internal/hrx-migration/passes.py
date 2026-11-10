import os,json,sys,collections,re,subprocess,sqlite3
CF=os.environ.get('CXXFILT','c++filt')
def base(s):
  s=s[5:] if s.startswith('void ') else s; s=s.replace('(anonymous namespace)::',''); s=re.sub(r'\(.*','',s); return re.sub(r'<.*','',s).split('::')[-1]
def load(path):
  if path.endswith('.db'):
    c=sqlite3.connect(path)
    rows=c.execute("SELECT ks.display_name, kd.start, kd.end, kd.queue_id FROM rocpd_kernel_dispatch kd JOIN rocpd_info_kernel_symbol ks ON ks.id=kd.kernel_id ORDER BY kd.start").fetchall()
    return [(base(r[0]),r[1],r[2],r[3]) for r in rows]
  ns=10.019036169
  ev=[json.loads(l) for l in open(path)]; ev=[e for e in ev if e.get('record_type')=='dispatch_event']
  keys=sorted({e['key'] for e in ev})
  dem=dict(zip(keys,subprocess.run([CF],input='\n'.join(keys),capture_output=True,text=True).stdout.split('\n')))
  r=[(base(dem[e['key']]),e['start_tick']*ns,e['end_tick']*ns,e['queue_ordinal']) for e in ev]
  r.sort(key=lambda x:x[1]); return r
rows=load(sys.argv[1])
# a pass = from an EmbedKernel to the dispatch before the next EmbedKernel; keep passes with > 4000 dispatches (pp2048)
idx=[i for i,r in enumerate(rows) if r[0]=='EmbedKernel']+[len(rows)]
out=[]
for a,b in zip(idx,idx[1:]):
  p=rows[a:b]
  if len(p)<3000: continue
  # trim trailing tail after big gap (>5 ms) i.e. decode/next phase
  q=[p[0]];ce=p[0][2]
  for r in p[1:]:
    if r[1]-ce>5e6: break
    q.append(r); ce=max(ce,r[2])
  p=q
  span=(max(r[2] for r in p)-p[0][1])/1e6; k=sum(r[2]-r[1] for r in p)/1e6
  busy=0;cs,ce=p[0][1],p[0][2];gaps=[]
  for r in p[1:]:
    if r[1]>ce: busy+=ce-cs; gaps.append(r[1]-ce); cs,ce=r[1],r[2]
    else: ce=max(ce,r[2])
  busy+=ce-cs
  fam=collections.Counter(); 
  for r in p: fam[r[0]]+=(r[2]-r[1])/1e6
  out.append(dict(n=len(p),span=span,kernel=k,busy=busy/1e6,idle=span-busy/1e6,ngaps=len(gaps),gap_med=sorted(gaps)[len(gaps)//2]/1e3 if gaps else 0,gap_sum_small=sum(g for g in gaps if g<2e4)/1e6,gap_sum_big=sum(g for g in gaps if g>=2e4)/1e6,fam=fam))
for i,o in enumerate(out):
  print(f"pass {i}: n={o['n']} span={o['span']:.1f} kernel={o['kernel']:.1f} busy={o['busy']:.1f} idle={o['idle']:.1f} (gaps {o['ngaps']} med {o['gap_med']:.2f}us small {o['gap_sum_small']:.1f} big {o['gap_sum_big']:.1f})")
json.dump([dict(o,fam=dict(o['fam'])) for o in out],open(sys.argv[2],'w'))
