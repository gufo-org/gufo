import sys, json, numpy as np, collections
sys.path.insert(0,'/home/mixer/gufo/llama.cpp/gguf-py')
from gguf import GGUFReader
from gguf.quants import dequantize
S='/tmp/claude-1000/-home-mixer-gufo/0f0c8823-bf21-4871-9575-7e0e56a78514/scratchpad/v0731'
E2M1=np.array([0,0.5,1,1.5,2,3,4,6,-0,-0.5,-1,-1.5,-2,-3,-4,-6],dtype=np.float32)
def load_fp4(name, N, K):
    q=np.fromfile(f'{S}/{name}.weight.bin',dtype=np.uint8).reshape(N, K//2)
    lo=(q & 15).astype(np.int32); hi=(q>>4).astype(np.int32)
    codes=np.empty((N,K),dtype=np.int32); codes[:,0::2]=lo; codes[:,1::2]=hi
    e=np.fromfile(f'{S}/{name}.scale.bin',dtype=np.uint8).reshape(N, K//32).astype(np.int32)
    scale=np.ldexp(np.float32(1.0), e-127).astype(np.float32)
    vals=E2M1[codes]*np.repeat(scale,32,axis=1)
    return codes, e, vals
def load_fp8(name, N, K):
    b=np.fromfile(f'{S}/{name}.weight.bin',dtype=np.uint8).reshape(N,K)
    b=b.astype(np.int32); s=b>>7; ex=(b>>3)&15; m=b&7
    v=np.where(ex==0, np.ldexp(m/8.0, -6), np.ldexp(1+m/8.0, ex-7)).astype(np.float32)
    v=np.where(s==1,-v,v); v=np.where((ex==15)&(m==7), np.nan, v)
    e=np.fromfile(f'{S}/{name}.scale.bin',dtype=np.uint8).reshape(N//128, K//128).astype(np.int32)
    scale=np.ldexp(np.float32(1.0), e-127).astype(np.float32)
    return v*np.repeat(np.repeat(scale,128,axis=0),128,axis=1)
def rel(W, Wd):
    err=Wd-W; return float(np.sqrt((err**2).mean())/np.sqrt((W**2).mean())), float(np.abs(err).mean()/np.abs(W).mean())
def q2_sym(v):   # 2-bit symmetric uniform levels {-3,-1,1,3}*s per block of 32, s grid-searched
    B=v.reshape(-1,32); amax=np.abs(B).max(1,keepdims=True)+1e-30; best=None
    for f in np.linspace(0.15,0.5,29):
        s=amax*f; q=np.clip(np.round((B/s-1)/2),-2,1)*2+1; d=q*s; err=((d-B)**2).sum(1)
        if best is None: best=err; out=d
        else: m=err<best; best=np.where(m,err,best); out=np.where(m[:,None],d,out)
    return out.reshape(v.shape)
def q2_asym(v):  # Q2_K-like: {0,1,2,3}*s + min per block of 16
    B=v.reshape(-1,16); mn=B.min(1,keepdims=True); mx=B.max(1,keepdims=True); s=(mx-mn)/3+1e-30
    q=np.clip(np.round((B-mn)/s),0,3); return (q*s+mn).reshape(v.shape)
def q3_sym(v):
    B=v.reshape(-1,32); amax=np.abs(B).max(1,keepdims=True)+1e-30; best=None
    for f in np.linspace(0.1,0.35,26):
        s=amax*f; q=np.clip(np.round((B/s-1)/2),-4,3)*2+1; d=q*s; err=((d-B)**2).sum(1)
        if best is None: best=err; out=d
        else: m=err<best; best=np.where(m,err,best); out=np.where(m[:,None],d,out)
    return out.reshape(v.shape)
def kmeans_block(v, k, iters=8):  # per-block optimal k-level LUT (lower bound), blocks of 32
    B=v.reshape(-1,32); n=B.shape[0]
    # init centroids by quantiles
    qs=np.quantile(B, np.linspace(0.05,0.95,k), axis=1).T  # [n,k]
    for _ in range(iters):
        idx=np.abs(B[:,:,None]-qs[:,None,:]).argmin(2)
        for j in range(k):
            m=(idx==j); cnt=m.sum(1); sm=(B*m).sum(1); qs[:,j]=np.where(cnt>0, sm/np.maximum(cnt,1), qs[:,j])
    idx=np.abs(B[:,:,None]-qs[:,None,:]).argmin(2)
    return np.take_along_axis(qs, idx, 1).reshape(v.shape)
def two_table(v, tables):  # IQ2_KS-like: per block of 32, pick best of 2 fixed signed tables scaled by block scale s (grid)
    B=v.reshape(-1,32); amax=np.abs(B).max(1,keepdims=True)+1e-30; best=None
    for T in tables:
        T=np.array(T,dtype=np.float32)
        for f in np.linspace(0.1,1.0,19):
            s=amax*f; idx=np.abs(B[:,:,None]/s[:,:,None]-T[None,None,:]).argmin(2); d=T[idx]*s; err=((d-B)**2).sum(1)
            if best is None: best=err; out=d
            else: m=err<best; best=np.where(m,err,best); out=np.where(m[:,None],d,out)
    return out.reshape(v.shape)
res={}
for name,N,K in [('layers.5.ffn.experts.0.w1',2048,4096),('layers.5.ffn.experts.7.w1',2048,4096),('layers.20.ffn.experts.3.w1',2048,4096),('layers.5.ffn.experts.0.w2',4096,2048),('layers.20.ffn.experts.3.w2',4096,2048)]:
    codes,e,vals=load_fp4(name,N,K)
    hist=np.bincount(codes.ravel(),minlength=16); p=hist/hist.sum(); H=float(-(p[p>0]*np.log2(p[p>0])).sum())
    mag=np.bincount((codes&7).ravel(),minlength=8)/codes.size
    r={"shape":[N,K],"code_entropy_bits":round(H,3),"magnitude_fraction":{str(E2M1[i]):round(float(mag[i]),4) for i in range(8)},"exp_range":[int(e.min()-127),int(e.max()-127)],"zero_fraction":round(float(mag[0]),4)}
    for lab,fn in [("q2_sym_g32",q2_sym),("q2_asym_g16",q2_asym),("q3_sym_g32",q3_sym),("lut2_kmeans_g32",lambda v:kmeans_block(v,4)),("lut3_kmeans_g32",lambda v:kmeans_block(v,8)),
                   ("lut2_2tables_g32",lambda v:two_table(v,[[-1,-0.25,0.25,1],[-1,-0.5,0,0.5]]))]:
        rr,rm=rel(vals,fn(vals)); r[lab]={"rel_rmse":round(rr,4),"rel_mae":round(rm,4)}
    res[name]=r; print(name, json.dumps(r), flush=True)
# compare current GGUF expert quant against the official FP4 weights
ds=GGUFReader('/persist/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf')
def gg(name):
    t=[t for t in ds.tensors if t.name==name][0]; return np.asarray(dequantize(t.data,t.tensor_type),dtype=np.float32), t.tensor_type.name
for lay,ex,w,gname in [(5,0,'w1','blk.5.ffn_gate_exps.weight'),(5,7,'w1','blk.5.ffn_gate_exps.weight'),(20,3,'w1','blk.20.ffn_gate_exps.weight'),(5,0,'w2','blk.5.ffn_down_exps.weight'),(20,3,'w2','blk.20.ffn_down_exps.weight')]:
    name=f'layers.{lay}.ffn.experts.{ex}.{w}'; N,K=(2048,4096) if w=='w1' else (4096,2048)
    codes,e,vals=load_fp4(name,N,K)
    G,ty=gg(gname); Ge=G[ex]  # shape [N,K]?
    if Ge.shape!=vals.shape: Ge=Ge.T
    rr,rm=rel(vals,Ge); c=float(np.corrcoef(vals.ravel()[:2_000_000],Ge.ravel()[:2_000_000])[0,1])
    res[name][f"gguf_{ty}"]={"rel_rmse":round(rr,4),"rel_mae":round(rm,4),"corr":round(c,4)}
    print(name, ty, res[name][f"gguf_{ty}"], flush=True)
# attention wq_b FP8 -> SHQ4/SHQ6 and GGUF Q8_0
sys.path.insert(0,'/home/mixer/gufo/tools'); from gufo import shq
for name,N,K,gname in [('layers.5.attn.wq_b',32768,1024,'blk.5.attn_q_b.weight'),('layers.5.attn.wo_b',4096,8192,'blk.5.attn_output_b.weight'),('layers.5.ffn.shared_experts.w1',2048,4096,'blk.5.ffn_gate_shexp.weight')]:
    W=load_fp8(name,N,K); r={"shape":[N,K],"nan":int(np.isnan(W).sum())}; W=np.nan_to_num(W)
    G,ty=gg(gname)
    if G.shape!=W.shape: G=G.T
    r[f"gguf_{ty}"]=dict(zip(("rel_rmse","rel_mae"),[round(x,4) for x in rel(W,G)]))
    for fmt,G_,sym in [("SHQ4-U4Z-G64",64,False),("SHQ4-S4-G64",64,True),("SHQ4-U4Z-G32",32,False)]:
        p=shq.quantize_shq4(W,group_size=G_,symmetric=sym); Wd=shq.dequant_shq4(p)[:N,:K]; r[fmt]=dict(zip(("rel_rmse","rel_mae"),[round(x,4) for x in rel(W,Wd)]))
    p=shq.quantize_shq6(W,group_size=64); Wd=shq.dequant_shq6(p)[:N,:K]; r["SHQ6-G64"]=dict(zip(("rel_rmse","rel_mae"),[round(x,4) for x in rel(W,Wd)]))
    res[name]=r; print(name, json.dumps(r), flush=True)
json.dump(res, open(f'{S}/fp4_study_0731.json','w'), indent=1)
print("FP4_DONE")

