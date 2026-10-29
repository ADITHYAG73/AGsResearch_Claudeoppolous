#!/usr/bin/env python3
"""Compare a local folder with its S3 copy (name + byte size). Prints a full report.
With --delete-verified, deletes ONLY local files whose S3 copy exists with the identical size.
Usage: verify_free.py LOCAL_DIR S3_PREFIX [--delete-verified]"""
import boto3, os, sys
local, prefix = sys.argv[1], sys.argv[2].rstrip("/") + "/"
delete = "--delete-verified" in sys.argv
s3 = boto3.client("s3"); b = os.environ["S3_BUCKET"]
remote = {}
for p in s3.get_paginator("list_objects_v2").paginate(Bucket=b, Prefix=prefix):
    for o in p.get("Contents", []): remote[o["Key"][len(prefix):]] = o["Size"]
ok, missing, mismatch, freed = [], [], [], 0
for root, _, files in os.walk(local):
    for f in files:
        if f == ".s3_sync_state.json": continue
        path = os.path.join(root, f); rel = os.path.relpath(path, local); sz = os.path.getsize(path)
        if rel not in remote: missing.append(rel)
        elif remote[rel] != sz: mismatch.append((rel, sz, remote[rel]))
        else:
            ok.append(rel)
            if delete and f.endswith(".safetensors"):
                os.remove(path); freed += sz
print(f"{local} vs s3://{b}/{prefix}: verified {len(ok)}, missing {len(missing)}, size mismatch {len(mismatch)}, remote objects {len(remote)}")
for m in missing: print("MISSING", m)
for m in mismatch: print("MISMATCH", *m)
if delete: print(f"deleted {freed/2**30:.1f} GB of verified .safetensors")
