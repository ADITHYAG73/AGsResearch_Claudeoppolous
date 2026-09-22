#!/usr/bin/env python3
"""NUMBERS-01. Does Qwen3.8-27B hold numbers in a structured way, and does the structure
repeat every 10?

The paper (Kantamneni & Tegmark 2502.00873 §4.1) fed single-token integers and looked at the
residual stream after layer 0. Qwen3.8 splits numbers into digits, so there is no single token
for "36". We read at the LAST digit token, which is the only position that has seen both digits
(attention is causal). Whether that position holds "36" or merely "6" is a question the data
answers: if it holds only "6", then 36 and 46 are identical there.

Numbers 10..99 only, so every prompt has exactly the same token count and the number sits at the
same index -> no position confound.

Two conditions:
  bare   "36"                              (closest to the paper's structure analysis)
  arith  "Output ONLY a number. 36+"       (the paper's own addition prompt, read at a's last digit)

Writes one .npy per condition, [90, 65, 5120] float16: 90 numbers x (embedding + 64 layers).
"""
import argparse, json, os, time
import numpy as np, torch

def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/models/Qwen3.8-27B")
    ap.add_argument("--out", default="/root/out/numbers")
    ap.add_argument("--lo", type=int, default=10); ap.add_argument("--hi", type=int, default=99)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    nums = list(range(a.lo, a.hi + 1))

    from transformers import AutoTokenizer, AutoConfig
    import transformers
    tok = AutoTokenizer.from_pretrained(a.model)
    cfg = AutoConfig.from_pretrained(a.model)
    log(f"config {cfg.architectures} | transformers {transformers.__version__}")

    # How does this tokenizer treat a two-digit number? Assert rather than assume.
    probe = {n: tok.encode(str(n), add_special_tokens=False) for n in (10, 36, 46, 99)}
    log("tokenisation of bare numbers: " + json.dumps({k: [tok.decode([i]) for i in v] for k, v in probe.items()}))
    per_number_tokens = len(probe[36])

    try:
        from transformers import Qwen3_5ForCausalLM as Cls; name = "Qwen3_5ForCausalLM"
    except Exception:
        from transformers import AutoModelForCausalLM as Cls; name = "AutoModelForCausalLM"
    t0 = time.time()
    model = Cls.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(a.device).eval()
    layers = model.model.layers
    log(f"LOAD OK via {name} in {time.time()-t0:.0f}s | {len(layers)} layers")

    conditions = {"bare": ("", ""), "arith": ("Output ONLY a number. ", "+")}
    meta = {"model": a.model, "numbers": nums, "per_number_tokens": per_number_tokens,
            "n_layers": len(layers), "conditions": {}}

    for cond, (prefix, suffix) in conditions.items():
        ids = [tok.encode(prefix + str(n) + suffix, add_special_tokens=False) for n in nums]
        lens = {len(x) for x in ids}
        assert len(lens) == 1, f"{cond}: prompts differ in length {lens} - would confound position"
        L = lens.pop()
        # index of the number's LAST digit token = end of prompt minus the suffix length
        pos = L - 1 - len(tok.encode(suffix, add_special_tokens=False)) if suffix else L - 1
        got = tok.decode([ids[26][pos]])                       # nums[26] == 36 -> expect "6"
        log(f"[{cond}] {L} tokens/prompt, reading index {pos}; for 36 that token is {got!r}")

        store = {}
        handles = [layers[i].register_forward_hook(
            (lambda i: lambda m, args, out: store.__setitem__(i, (out[0] if isinstance(out, tuple) else out)[:, pos, :].detach().float().cpu()))(i))
            for i in range(len(layers))]
        emb = {}
        h0 = model.model.embed_tokens.register_forward_hook(
            lambda m, args, out: emb.__setitem__(0, out[:, pos, :].detach().float().cpu()))

        t1 = time.time()
        with torch.no_grad():
            model(torch.tensor(ids, device=a.device))          # all 90 prompts in ONE batch
        for h in handles: h.remove()
        h0.remove()

        arr = np.stack([emb[0].numpy()] + [store[i].numpy() for i in range(len(layers))], axis=1).astype(np.float16)
        p = os.path.join(a.out, f"acts_{cond}.npy"); np.save(p, arr)
        meta["conditions"][cond] = {"prompt_example": prefix + "36" + suffix, "n_tokens": L,
                                    "read_index": pos, "token_at_read_index_for_36": got,
                                    "shape": list(arr.shape), "seconds": round(time.time()-t1, 1)}
        log(f"[{cond}] SAVED {arr.shape} -> {os.path.basename(p)} in {time.time()-t1:.1f}s")

    meta["peak_vram_gb"] = torch.cuda.max_memory_allocated()/1e9 if a.device == "cuda" else None
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    log(f"DONE, peak VRAM {meta['peak_vram_gb']} GB -> meta.json")

if __name__ == "__main__":
    main()
