"""Paired bootstrap of rounded cumulative means from historical KL logs.

The equal-sized scored chunks each have 1023 positions. For cumulative
means c_i, recover chunk i as i*c_i - (i-1)*c_(i-1). Resample whole paired
chunks, not tokens. Intervals cover this slice under chunk exchangeability;
they do not establish broader task-quality equivalence.
"""

import re, numpy as np
from pathlib import Path
root=Path(__file__).resolve().parent/'source'
def chunks(name, col):
    a=[]
    for s in (root/name).read_text().splitlines():
        if re.match(r'^\s+\d+\s+\d+\.\d+\s+±',s):
            fields=re.findall(r'[-+]?\d+(?:\.\d+)?',s)
            assert int(fields[0]) == len(a)+1, (name, fields[0])
            a.append(float(fields[col]))
    assert len(a)==24
    return np.diff(np.r_[0,np.arange(1,25)*a])
rng=np.random.default_rng(731)
idx=rng.integers(0,24,size=(100000,24))
for col, lab in [(5,'KL Q5 minus Q6'),(3,'NLL Q5 minus Q6')]:
    d=chunks('kl-attnQ5K.log',col)-chunks('kl-denseQ6K.log',col)
    print(lab,'mean',d.mean(),'positive chunks',sum(d>0),'paired chunk bootstrap 95%',np.quantile(d[idx].mean(axis=1),[.025,.975]))
for name in ['kl-attnQ5K.log','kl-denseQ6K.log']:
    d=chunks(name,3)
    print(name,'NLL vs base mean',d.mean(),'paired chunk bootstrap95',np.quantile(d[idx].mean(axis=1),[.025,.975]))
