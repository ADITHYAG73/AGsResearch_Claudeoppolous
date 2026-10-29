#!/usr/bin/env python3
"""Upload new or changed files under a local dir to s3://$S3_BUCKET/<prefix>/ (size+mtime check)."""
import boto3, json, os, sys
local, prefix = sys.argv[1], sys.argv[2].rstrip("/")
b = os.environ["S3_BUCKET"]
s3 = boto3.client("s3", region_name="ap-south-1")
state_p = os.path.join(local, ".s3_sync_state.json")
state = json.load(open(state_p)) if os.path.exists(state_p) else {}
n = 0
for root, _, files in os.walk(local):
    for f in files:
        p = os.path.join(root, f)
        if f == ".s3_sync_state.json": continue
        st = os.stat(p); sig = f"{st.st_size}:{int(st.st_mtime)}"
        rel = os.path.relpath(p, local)
        if state.get(rel) == sig: continue
        s3.upload_file(p, b, f"{prefix}/{rel}"); state[rel] = sig; n += 1
json.dump(state, open(state_p, "w"))
print(f"synced {n} files to s3://{b}/{prefix}")
