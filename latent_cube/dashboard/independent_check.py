#!/usr/bin/env python3
"""Independent check by Claude of Codex's dashboard numbers: re-read raw tensors from S3 with separate code
and compare. Read-only. Written without using build_data.py."""
import boto3, gzip, json, math, random, struct, requests
import numpy as np
s3=boto3.client("s3",region_name="ap-south-1"); B="posttrain-math-459653581741-ap-south-1"
man=json.load(open("site/manifest.json")); print("manifest keys",list(man.keys()))
caps=man["rows"]; print("n",len(caps),"example",json.dumps(caps[0])[:300])
def hdr(key):
    n=struct.unpack("<Q",s3.get_object(Bucket=B,Key=key,Range="bytes=0-7")["Body"].read())[0]
    return json.loads(s3.get_object(Bucket=B,Key=key,Range=f"bytes=8-{7+n}")["Body"].read()),8+n
def row(key,h,base,name,idx,dt):
    m=h[name]; sh=m["shape"]; w=int(np.prod(sh[len(idx):]))
    off=0; mul=w
    for i,s in reversed(list(zip(idx,sh[:len(idx)]))): off+=i*mul; mul*=s
    sz={"F16":2,"F32":4}[m["dtype"]]; a=base+m["data_offsets"][0]+off*sz
    b=s3.get_object(Bucket=B,Key=key,Range=f"bytes={a}-{a+w*sz-1}")["Body"].read()
    return np.frombuffer(b,dtype={"F16":"<f2","F32":"<f4"}[m["dtype"]]).astype(np.float64)
# Embedding rows + norm weight from HF, by range
REV="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"; U=f"https://huggingface.co/Qwen/Qwen3.5-4B/resolve/{REV}/"
idx=requests.get(U+"model.safetensors.index.json").json()["weight_map"]
def hf(name,start=None,count=None):
    f=idx[name]; n=struct.unpack("<Q",requests.get(U+f,headers={"Range":"bytes=0-7"}).content)[0]
    h=json.loads(requests.get(U+f,headers={"Range":f"bytes=8-{7+n}"}).content); m=h[name]; base=8+n+m["data_offsets"][0]
    w=m["shape"][-1] if start is not None else int(np.prod(m["shape"]))
    a=base+(start or 0)*w*2; bb=requests.get(U+f,headers={"Range":f"bytes={a}-{a+w*2*(count or 1)-1}"}).content
    v=np.frombuffer(bb,dtype="<u2")
    return v
def bf16(u): # bfloat16 -> float32: bf16 is the top 16 bits of float32
    return np.frombuffer((u.astype(np.uint32)<<16).tobytes(),dtype="<f4").astype(np.float64)
w=bf16(hf("model.language_model.norm.weight")); print("norm w",w.shape)
E={}
def erow(t):
    if t not in E: E[t]=bf16(hf("model.language_model.embed_tokens.weight",t,1))
    return E[t]
random.seed(20261031); picks=random.sample(range(len(caps)),6)
tot=0; bad=[]
def cmp(label,a,b,tol):
    global tot; tot+=1; d=abs(a-b)
    if d>tol: bad.append((label,a,b)); print("MISMATCH",label,a,b)
for pi in picks:
    c=caps[pi]; file=c.get("file") or c.get("path") or f"prompts/{pi}.bin"
    d=json.loads(gzip.decompress(open("site/"+file,"rb").read()))
    key=d["source_key"].replace(".json",".safetensors"); meta=json.load(s3.get_object(Bucket=B,Key=d["source_key"])["Body"])
    h,base=hdr(key); h.pop("__metadata__",None); T=meta["saved"]["R1"][1]
    print("---",meta["id"],meta["variant"],"T",T,"file",file)
    # reply preserved exactly
    cmp("free_reply",0 if d["meta"]["free_reply"]==meta["free_reply"] else 1,0,0)
    cmp("tokens",0 if d["meta"]["tokens"]==meta["tokens"] else 1,0,0)
    # norms at random layer/position
    for _ in range(4):
        li=random.randrange(32); pi2=random.randrange(len(d["norm_positions"])); p=d["norm_positions"][pi2]
        for nm in ("R1","R2","R5"):
            x=row(key,h,base,nm,(li,p),"F16"); l2=float(np.linalg.norm(x)); rms=float(np.sqrt((x**2).mean()))
            v=d["norms"][nm][li][pi2]
            if _==0 and nm=="R1": print(f"   norm convention: page {v} | L2 {l2:.3f} | RMS {rms:.3f}")
            cmp(f"{nm} norm L{li} p{p}",v,l2,0.002) if abs(v-l2)<abs(v-rms) else cmp(f"{nm} rms L{li} p{p}",v,rms,0.002)
    # readout positions
    cmp("positions",0 if d["lens_positions"]==[meta["positions"][k] for k in ("instr_end","prompt_end","reply_end")] or len(d["lens_positions"])==T else 1,0,0)
    # MLP top-20 at one cell
    li=random.randrange(32); ri=random.randrange(len(d["mlp_positions"])); p=d["mlp_positions"][ri]
    x=row(key,h,base,"R4",(li,p),"F16"); order=np.argsort(-np.abs(x))[:20]
    page=d["mlp"][li][ri]["top"]
    cmp("mlp top values",float(np.abs(np.abs(x[order])-np.abs(np.array([q[1] for q in page]))).max()),0,1e-4)
    cmp("mlp top ids overlap",20-len(set(order.tolist())&{q[0] for q in page}),0,1)
    # attention row
    la=random.randrange(8); hd=random.randrange(16); qi=random.randrange(len(d["attention_queries"])); q=d["attention_queries"][qi]
    x=row(key,h,base,"attn",(la,hd,q),"F16"); page=np.array(d["attention"][la][hd][qi])
    cmp("attention row",float(np.abs(x[:len(page)]-page).max()),0,1e-5)
    # logit lens: layer 31 top-5 vs saved logits row; mid layer via probability RATIOS from E rows
    ri=random.randrange(len(d["lens_positions"])); p=d["lens_positions"][ri]
    lg=row(key,h,base,"logits",(p,),"F16"); pr=np.exp(lg-lg.max()); pr/=pr.sum(); o=np.argsort(-pr)[:5]
    top=d["lens"][31][ri]["top"]
    print(f"   L31 top saved {o.tolist()} page {[t[0] for t in top]}")
    cmp("L31 top1",0 if int(o[0])==top[0][0] else 1,0,0); cmp("L31 p1",float(pr[o[0]]),top[0][2],0.02)
    li=random.randrange(1,31); r1=row(key,h,base,"R1",(li,p),"F16")
    hv=r1/math.sqrt((r1**2).mean()+1e-6)*(1+w)
    top=d["lens"][li][ri]["top"]; lgs=[float(hv@erow(t[0])) for t in top[:3]]
    for a,b,ta,tb in [(lgs[0],lgs[1],top[0],top[1]),(lgs[0],lgs[2],top[0],top[2])]:
        if ta[2]>1e-5 and tb[2]>1e-5: cmp(f"lens ratio L{li} p{p}",a-b,math.log(ta[2]/tb[2]),0.02)
    print(f"   L{li} p{p} tokens {[t[1] for t in top[:3]]} probs {[t[2] for t in top[:3]]} my logit gaps {lgs[0]-lgs[1]:.4f} page {math.log(top[0][2]/top[1][2]):.4f}")
print("COMPARED",tot,"MISMATCHES",len(bad)); [print(b) for b in bad]
