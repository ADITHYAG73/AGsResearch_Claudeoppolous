#!/usr/bin/env python3
"""site/ -> site_pub/ for claude.ai: .bin is not a served type, so each .bin becomes base64 text (.txt) and the page's
loader decodes it. Bytes are identical after decoding (asserted). Prints publish batches: <=255 files, <=60 MB."""
import base64, json, os, shutil
SRC, DST = "site", "site_pub"
shutil.rmtree(DST, ignore_errors=True); os.makedirs(DST + "/prompts"); os.makedirs(DST + "/across")
for root, _, files in os.walk(SRC):
    for f in files:
        p = os.path.join(root, f); rel = os.path.relpath(p, SRC)
        if f.endswith(".bin"):
            raw = open(p, "rb").read(); t = base64.b64encode(raw)
            assert base64.b64decode(t) == raw
            open(os.path.join(DST, rel[:-4] + ".txt"), "wb").write(t)
        elif f != "index.html":
            shutil.copy(p, os.path.join(DST, rel))
h = open(SRC + "/index.html", encoding="utf-8").read()
a = "const r=await fetch(path);"; assert h.count(a) == 1
h = h.replace(a, "const r=await fetch(path.replace(/\\.bin$/,'.txt'));")
b = "r.body.pipeThrough(new DecompressionStream('gzip'))"; assert h.count(b) == 1
h = h.replace(b, "new Blob([Uint8Array.from(atob((await r.text()).trim()),c=>c.charCodeAt(0))]).stream().pipeThrough(new DecompressionStream('gzip'))")
open(DST + "/index.html", "w", encoding="utf-8").write(h)
fs = sorted(os.path.relpath(os.path.join(r, f), DST) for r, _, ff in os.walk(DST) for f in ff if f != "index.html")
fs.sort(key=lambda x: (not x.endswith(".json"), not x.startswith("across"), x))
batches, cur, size = [], [], 0
for f in fs:
    s = os.path.getsize(os.path.join(DST, f))
    if len(cur) >= 250 or size + s > 60_000_000: batches.append(cur); cur, size = [], 0
    cur.append(f); size += s
batches.append(cur)
json.dump(batches, open("publish_batches.json", "w"))
print("files", len(fs), "total MB", round(sum(os.path.getsize(os.path.join(DST, f)) for f in fs) / 1e6, 1), "batches", [len(b) for b in batches])
