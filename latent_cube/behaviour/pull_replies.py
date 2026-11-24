#!/usr/bin/env python3
"""Pull the model's own full replies (free variant JSON) for all 201 prompts from S3 -> replies.jsonl. No scoring."""
import boto3, json
from concurrent.futures import ThreadPoolExecutor
s3 = boto3.client("s3", region_name="ap-south-1"); b = "posttrain-math-459653581741-ap-south-1"
rows = [json.loads(l) for l in open("../prompts.jsonl")]
def fn(i): return i.replace("|", "__").replace("/", "_").replace("·", "dot").replace("(", "").replace(")", "").replace("^", "pow").replace("²", "sq")
def get(r):
    m = json.load(s3.get_object(Bucket=b, Key=f"latent_cube/capture01/out/{fn(r['id'])}__free.json")["Body"])
    assert m["id"] == r["id"]
    keep = ("id","set","group","func","kth","template","prompt","power","coef","answer","free_reply","free_reply_finish","check_argmax_mismatch")
    d = {k: m.get(k) for k in keep}; d["reply_tokens"] = len(m["token_ids"]) - m["prompt_len"]; return d
with ThreadPoolExecutor(8) as ex: out = list(ex.map(get, rows))
with open("replies.jsonl", "w") as f:
    for d in out: f.write(json.dumps(d, ensure_ascii=False) + "\n")
print(len(out), "replies; finish:", {x: sum(d["free_reply_finish"] == x for d in out) for x in ("eos", "length")},
      "| reply tokens min/max:", min(d["reply_tokens"] for d in out), max(d["reply_tokens"] for d in out))
