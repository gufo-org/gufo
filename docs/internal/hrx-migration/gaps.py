import os,json,sys,collections,re,subprocess
ev=[json.loads(l) for l in open(sys.argv[1])]
ev=[e for e in ev if e.get('record_type')=='dispatch_event']
ns=10.019036169
ev.sort(key=lambda e:e['start_tick'])
keys=sorted({e['key'] for e in ev})
dem=dict(zip(keys,subprocess.run([os.environ.get('CXXFILT','c++filt')],input='\n'.join(keys),capture_output=True,text=True).stdout.split('\n')))
def n(k):
  s=dem[k]; s=s[5:] if s.startswith('void ') else s; s=s.replace('(anonymous namespace)::',''); s=re.sub(r'\(.*','',s); return re.sub(r'<.*','',s).split('::')[-1]
# global idle gaps (union)
ce=ev[0]['end_tick']; last=ev[0]; gaps=[]
for e in ev[1:]:
  if e['start_tick']>ce: gaps.append(((e['start_tick']-ce)*ns/1e3,last,e))
  if e['end_tick']>ce: ce=e['end_tick']; last=e
big=[g for g in gaps if g[0]>20]
print('idle gaps >20us: n=%d sum=%.1f ms; all gaps sum=%.1f ms n=%d'%(len(big),sum(g[0] for g in big)/1e3,sum(g[0] for g in gaps)/1e3,len(gaps)))
same=sum(g[0] for g in big if g[1]['queue_ordinal']==g[2]['queue_ordinal'])/1e3
print('big gaps same queue %.1f ms, cross-queue %.1f ms'%(same,sum(g[0] for g in big)/1e3-same))
c=collections.defaultdict(lambda:[0,0.0])
for g in big:
  k=(n(g[1]['key']),g[1]['queue_ordinal'],n(g[2]['key']),g[2]['queue_ordinal']); c[k][0]+=1; c[k][1]+=g[0]
for k,v in sorted(c.items(),key=lambda x:-x[1][1])[:15]:
  print('%6.1f ms n=%4d mean=%6.1fus  %s(q%d) -> %s(q%d)'%(v[1]/1e3,v[0],v[1]/v[0],*k))
# histogram of big gaps
h=collections.Counter(min(int(g[0]//50)*50,1000) for g in big)
print('hist us:',sorted(h.items()))
