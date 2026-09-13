import urllib.request, json, struct
root='https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/resolve/7872f01b1d1fe23eabc4c98b48bffcef5a386062/'
def get(name, start=None, end=None):
    h={} if start is None else {'Range':f'bytes={start}-{end}'}
    with urllib.request.urlopen(urllib.request.Request(root+name,headers=h),timeout=40) as r:
        if start is not None:
            assert r.status == 206, (name, r.status)
            assert r.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
        body = r.read() if start is None else r.read(end-start+1)
        if start is not None:
            assert len(body) == end-start+1, (name, len(body))
        return body
idx=json.loads(get('model.safetensors.index.json'))
print('metadata',idx.get('metadata'))
print('shards',len(set(idx['weight_map'].values())))
names=['layers.4.attn.indexer.wq_b.weight','layers.4.attn.indexer.wq_b.scale','layers.5.ffn.shared_experts.w1.weight','layers.5.ffn.shared_experts.w1.scale','layers.5.attn.wq_a.weight','layers.5.attn.wkv.weight','layers.5.attn.wo_a.weight','layers.5.attn.wo_b.weight','layers.5.ffn.gate.weight','embed.weight','head.weight']
names += ['layers.5.ffn.experts.0.w1.weight', 'layers.5.ffn.experts.0.w1.scale',
          'layers.5.attn.wq_b.weight', 'layers.5.attn.wq_b.scale', 'head.scale']
cache={}
for n in names:
    shard=idx['weight_map'].get(n)
    if shard is None: print(n,'not present'); continue
    if shard not in cache:
        # Distinct query strings avoid intermediary caches confusing range requests.
        hlen=struct.unpack('<Q',get(shard+'?headerlen=1',0,7))[0]
        cache[shard]=json.loads(get(shard+'?headerbody=1',8,7+hlen))
    print(n,shard,cache[shard][n])
