import os,sqlite3,sys,collections
c=sqlite3.connect(sys.argv[1])
cols=[r[1] for r in c.execute("PRAGMA table_info(rocpd_kernel_dispatch)")]
qcol='queue_id' if 'queue_id' in cols else None
scol='stream_id' if 'stream_id' in cols else None
rows=c.execute(f"SELECT ks.display_name, kd.start, kd.end, {qcol or 0}, {scol or 0} FROM rocpd_kernel_dispatch kd JOIN rocpd_info_kernel_symbol ks ON ks.id=kd.kernel_id ORDER BY kd.start").fetchall()
print('queues',collections.Counter(r[3] for r in rows),'streams',collections.Counter(r[4] for r in rows))
passes=[[rows[0]]]
for a,b in zip(rows,rows[1:]):
  if b[1]-max(x[2] for x in passes[-1][-50:])>50e6: passes.append([])
  passes[-1].append(b)
for i,p in enumerate(passes):
  span=(max(r[2] for r in p)-p[0][1])/1e6
  k=sum(r[2]-r[1] for r in p)/1e6
  kr=sum(r[2]-r[1] for r in p if 'rocclr' in r[0])/1e6
  busy=0;cs,ce=p[0][1],p[0][2];gaps=[];ov=0
  for r in p[1:]:
    if r[1]>ce: busy+=ce-cs; gaps.append(r[1]-ce); cs,ce=r[1],r[2]
    else: ov+=min(ce,r[2])-r[1]; ce=max(ce,r[2])
  busy+=ce-cs
  if k<5: continue
  g=sorted(gaps)
  nq=collections.Counter(r[3] for r in p)
  print(f"pass {i}: n={len(p)} (rocclr {sum(1 for r in p if 'rocclr' in r[0])}, {kr:.1f} ms) span={span:.1f} kernel={k:.1f} busy={busy/1e6:.1f} idle={span-busy/1e6:.1f} overlap={ov/1e6:.1f} gaps median={g[len(g)//2]/1e3:.2f}us p90={g[int(len(g)*.9)]/1e3:.2f}us sum>20us={sum(x for x in g if x>2e4)/1e6:.1f}ms queues={dict(nq)}")
