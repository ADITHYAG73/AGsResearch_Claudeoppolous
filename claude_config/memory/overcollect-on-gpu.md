---
name: overcollect-on-gpu
description: "Compute is expensive, storage is cheap — on every GPU session save everything plausibly useful (all layers, all positions, extra internals) to S3, even if unused."
metadata:
  node_type: memory
  type: feedback
  originSessionId: 5d62c3d5-0568-4062-a81c-214d5eefcc27
  modified: 2026-10-29T18:07:18.885Z
---

During a billed GPU session, over-collect rather than under-collect: residual stream at every depth, MLP neuron activations, attention/mixer outputs, attention patterns, linear-attention states where feasible, logits, generations, configs and versions. Push it all to AG's S3 bucket.

**Why:** AG, 2026-10-29: "compute is expensive, memory is cheap… we should over-collect instead of under-collect… it's going to sit there in our S3." Re-running a pod because something wasn't saved costs more than storing it.

**How to apply:** size the save plan in the rule-7 preflight (bytes per token × tokens) and only drop something when it is genuinely unaffordable (e.g. GatedDeltaNet recurrent state is ~48 MB per position for Qwen3.5-4B in fp32 — save it at selected positions, not all). Still write the comparisons down before looking at the data. Related: [[gpu-run-preflight]].
