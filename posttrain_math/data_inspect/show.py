import json,sys
split,start,n=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
rows=[json.loads(l) for l in open("ifc4/sample.jsonl") if split in l[-120:]]
for i,r in enumerate(rows[start:start+n],start):
    msgs=r["messages"]; sysm=next((m["content"] for m in msgs if m["role"]=="system"),"")
    u=next(m["content"] for m in msgs if m["role"]=="user")
    nt=sum(m["role"]=="user" for m in msgs)
    print(f"[{i}] turns={nt} sys={len(sysm or '')}ch | {' '.join(str(u).split())[:230]}")
