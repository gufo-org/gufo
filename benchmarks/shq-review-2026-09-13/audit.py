import sys, json, re, math, glob
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
from collections import defaultdict
sys.path.insert(0, str(ROOT/'llama.cpp/gguf-py'))
from gguf import GGUFReader
import numpy as np
sys.path.insert(0, str(ROOT/'tools'))
from gufo.shq import quantize_shq6
for g in [32,64]:
    p=quantize_shq6(np.ones((16,64),np.float32),g)
    print('SHQ6',g,{k:len(p[k]) for k in ['weight','scale','zero']},sum(len(p[k]) for k in ['weight','scale','zero'])*8/1024)
paths=glob.glob('/persist/models/*.gguf')+glob.glob('/persist/models/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/*Q4_K_XL.gguf')
seen=set()
for path in paths:
    if 'DFlash' in path or path.split('/')[-1] in seen: continue
    seen.add(path.split('/')[-1]); r=GGUFReader(path)
    rows=[(t.name,int(t.n_bytes),int(np.prod(t.shape)),t.tensor_type.name) for t in r.tensors]
    print('\nARTIFACT',path, 'bytes',sum(b for _,b,_,_ in rows))
    isds='DeepSeek' in path
    def factor(n):
        if n=='token_embd.weight': return 1/129280 if isds else 1/248320
        if '_exps.weight' in n: return 6/256
        if 'tid2eid' in n: return 1/129280
        if not isds and n.startswith('blk.64.'): return 0
        return 1
    base=sum(b*factor(n) for n,b,p,t in rows)
    print('one-pass active bytes',base,'GB',base/1e9)
    if isds and 'support' not in path:
        fam=defaultdict(float)
        for n,b,p,t in rows:
            key='expert' if '_exps' in n else 'embedding' if n=='token_embd.weight' else 'Q8 dense' if t=='Q8_0' else t
            fam[key]+=b*factor(n)
        print('families',dict(fam))
        selected=lambda n: bool(re.search(r'\.(attn_q_b|attn_output_a|attn_output_b|ffn_(gate|up|down)_shexp)\.weight$',n)) and '.indexer.' not in n
        for b in [4.5,5.5,6.5625,4.3125,5.25,5.328125,6.25]:
            total=sum((p*b/8 if selected(n) else p*6.5625/8 if n=='output.weight' else sz)*factor(n) for n,sz,p,t in rows)
            print('dense selected tier',b,total/1e9,170e9/total,base/total-1)
        totalexp=sum(p for n,sz,p,t in rows if '_exps' in n)
        print('expert params',totalexp,'GiB per bpw',totalexp/8/2**30)
        att=lambda n: bool(re.search(r'\.(attn_q_a|attn_q_b|attn_kv|attn_output_a|attn_output_b|ffn_(gate|up|down)_shexp)\.weight$',n))
        aux=lambda n: 'compressor' in n or '.indexer.' in n or n=='output.weight'
        protected=lambda n: 'gate_inp' in n or 'hc_' in n or 'tid2eid' in n or 'norm' in n or 'sinks' in n or 'bias' in n or n=='token_embd.weight'
        for lab, a, x, eb in [('attSHQ6',6.25,None,2.25),('attSHQ4',4.3125,None,2.25),('plusaux8',4.3125,8.25,2.25),('dense4',4.3125,4.3125,2.25),('dense4exp3',4.3125,4.3125,3),('dense4exp4.25',4.3125,4.3125,4.25)]:
            total=0; size=0
            for n,sz,p,t in rows:
                b=p*eb/8 if '_exps' in n else sz if protected(n) else p*x/8 if x is not None and aux(n) else p*a/8 if att(n) else sz
                total+=b*factor(n); size+=b
            print('scenario',lab,total/1e9,170e9/total,241e9/total,'resident',size/2**30)
        for b in [2.0625,2.25,2.4,2.625,3,3.25,2+3/32+1/3,4.25]:
            print('expert size',b,totalexp*b/8/2**30)
        for b in [4.3125,6.25]:
            dense=sum(p*b/8 if (t in ['Q8_0','F16'] and n!='token_embd.weight' and p>100000) else sz for n,sz,p,t in rows if '_exps' not in n)
            print('dense resident illustrative',b,dense/2**30)
    if not isds:
        up=lambda n: any(x in n for x in ['ffn_down.weight','ssm_out.weight','attn_output.weight']) or n in ['output.weight','token_embd.weight']
        for b6 in [6.25,6.5625]:
            size=sum(p*(b6 if up(n) else 4.3125)/8 if len(next(t.shape for t in r.tensors if t.name==n))>=2 else sz for n,sz,p,t in rows)
            traffic=sum((p*(b6 if up(n) else 4.3125)/8 if len(next(t.shape for t in r.tensors if t.name==n))>=2 else sz)*factor(n) for n,sz,p,t in rows)
            print('Qwen recipe',b6,'bpw',size*8/sum(p for n,b,p,t in rows),'GiB',size/2**30,'activeGB',traffic/1e9,'relative gain',base/traffic-1)
