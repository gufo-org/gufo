import os,json,sys,collections
ev=[json.loads(l) for l in open(sys.argv[1])]
ev=[e for e in ev if e.get('record_type')=='dispatch_event']
ns=10.019036169
ev.sort(key=lambda e:e['start_tick'])
print('streams',collections.Counter(e['stream_id'] for e in ev))
# split passes on gaps > 50 ms
passes=[[ev[0]]]
for a,b in zip(ev,ev[1:]):
  if (b['start_tick']-a['end_tick'])*ns>50e6: passes.append([])
  passes[-1].append(b)
for i,p in enumerate(passes):
  span=(max(e['end_tick'] for e in p)-p[0]['start_tick'])*ns/1e6
  ktime=sum(e['duration_ticks'] for e in p)*ns/1e6
  # union and gaps
  busy=0;cur_s,cur_e=p[0]['start_tick'],p[0]['end_tick'];gaps=[];ov=0
  for e in p[1:]:
    if e['start_tick']>cur_e: busy+=cur_e-cur_s; gaps.append(e['start_tick']-cur_e); cur_s,cur_e=e['start_tick'],e['end_tick']
    else: ov+=min(cur_e,e['end_tick'])-e['start_tick']; cur_e=max(cur_e,e['end_tick'])
  busy+=cur_e-cur_s
  if ktime<5: continue
  g=sorted(gaps)
  print(f"pass {i}: n={len(p)} span={span:.1f} ms kernel={ktime:.1f} busy={busy*ns/1e6:.1f} idle={span-busy*ns/1e6:.1f} overlap={ov*ns/1e6:.1f} gaps n={len(g)} median={g[len(g)//2]*ns/1e3:.2f}us p90={g[int(len(g)*.9)]*ns/1e3:.2f}us sum>20us={sum(x for x in g if x*ns>2e4)*ns/1e6:.1f}ms")
