# Random-offset HTTP range sampling of the IFC-v4 JSONL files (no full download).
import json, random, urllib.request, sys
BASE="https://huggingface.co/datasets/nvidia/Nemotron-SFT-Instruction-Following-Chat-v4/resolve/main/data/"
FILES={"chat.jsonl":24704029665,"instruction_following_for_response_length_control.jsonl":1754980945,"multi_turn_instruction_following.jsonl":54283343615}
random.seed(20261018)
out=open("ifc4/sample.jsonl","w")
for fn,size in FILES.items():
    n=0
    for k in range(40):
        off=random.randrange(0,size-4_000_000)
        req=urllib.request.Request(BASE+fn,headers={"Range":f"bytes={off}-{off+3_000_000}"})
        data=urllib.request.urlopen(req,timeout=60).read().decode("utf-8","ignore")
        lines=data.split("\n")[1:-1]   # drop partial first/last
        for ln in lines[:5]:
            try: r=json.loads(ln)
            except Exception: continue
            r["_split"]=fn; out.write(json.dumps(r)+"\n"); n+=1
    print(fn,"rows",n,flush=True)
