"""Sandbox conditions: how often each model answers WITHOUT calling the tool, and how often that is right."""
import json, sys
d = sys.argv[1]
for m in ("Qwen3.5-4B-rerun", "Qwen3.5-4B-ifc4-s050", "Qwen3.5-4B-ifc4-s100"):
    for c in ("tool_off", "tool_on"):
        R = [json.loads(l) for l in open(f"{d}/scored_{m}_{c}.jsonl")]
        calls = [int(r["n_calls"]) for r in R]
        n0 = sum(k == 0 for k in calls)
        acc0 = sum(r["status"] == "correct" for r, k in zip(R, calls) if k == 0)
        print(f"{m:22s} {c:8s} no call: {n0:4d}/{len(R)} ({100*n0/len(R):5.1f}%)  correct among no-call: {acc0:4d}  mean calls {sum(calls)/len(R):.2f}")
