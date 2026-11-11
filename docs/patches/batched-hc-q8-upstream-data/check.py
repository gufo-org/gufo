from pathlib import Path
import hashlib,json,zipfile
root=Path(__file__).resolve().parent; m=json.loads((root/'archive.json').read_text());p=root/'acquisition.zip'
assert hashlib.sha256(p.read_bytes()).hexdigest()==m['sha256']
with zipfile.ZipFile(p)as z:
 assert set(z.namelist())==set(m['entries'])
 for n,e in m['entries'].items():
  data=z.read(n);assert len(data)==e['bytes'] and hashlib.sha256(data).hexdigest()==e['sha256'],n
 q=json.loads(z.read('upstream-batched-hc-q8-v2/analysis.json'));assert q['status']=='pass'and q['records']==104 and q['complete_materialized_state_pairs']==32
 prior=json.loads(z.read('upstream-batched-hc-q8-v1/tp1-result.json'));assert prior['status']=='probe-failed'
print('PASS archive integrity and recorded coverage; not a GPU rerun or independent mathematical check')
