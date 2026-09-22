#!/usr/bin/env python3
"""TRIG-01. Can Qwen3.8-27B evaluate sin/cos/tan in degrees, to 5 decimal places?

AG's design. x sweeps a full cycle in DEGREES; the model is told to answer with exactly five
decimal places; every answer is scored against Python's math library (double precision), so the
reward is verifiable rather than judged.

TWO CONDITIONS
  off   thinking disabled -> one forward pass. This is AG's actual question: no tools, no
        scratchpad. Full 1-degree sweep, 360 x 3 = 1080 asks.
  on    thinking enabled -> the model writes out a method. Every 5 degrees, 72 x 3 = 216 asks,
        because the METHOD will not differ between 37 and 38 degrees but the tokens cost real money.

WHAT IS MEASURED
  1. absolute error against the true value
  2. format compliance - exactly five decimal places, as instructed (a different kind of failure)
  3. whether a number could be parsed at all
  Poles are KEPT: tan(90) and tan(270) are undefined and what it says there is informative.

ACTIVATIONS (condition 'off' only): the residual stream at all 65 depths at the LAST PROMPT
token - the state just before it speaks. Batches are LEFT-padded so that token is at index -1
for every sample. Free at inference time; makes the 'how' question answerable later without
paying for another pod.
"""
import argparse, json, math, os, re, time
import numpy as np, torch

FNS = {"sin": math.sin, "cos": math.cos, "tan": math.tan}
def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)

def truth(fn, deg):
    """Exact value, or None at a pole. tan is undefined at 90 and 270 degrees."""
    if fn == "tan" and deg % 180 == 90: return None
    return FNS[fn](math.radians(deg))

def prompt_for(fn, deg):
    return (f"Compute {fn}({deg} degrees). "
            f"Respond with only the numeric value, rounded to exactly 5 decimal places.")

NUM = re.compile(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?")
def parse(text):
    """Take the LAST number in the answer; report its decimal-place count for the format check."""
    after = text.split("</think>")[-1]
    ms = NUM.findall(after)
    if not ms: return None, None, after.strip()[:120]
    s = ms[-1]
    try: v = float(s)
    except ValueError: return None, None, after.strip()[:120]
    dp = len(s.split(".")[1]) if "." in s and "e" not in s.lower() else None
    return v, dp, s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/models/Qwen3.8-27B")
    ap.add_argument("--out", default="/root/out/trig")
    ap.add_argument("--cond", default="off", choices=["off", "on"])
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--degrees", default=None,
                    help="explicit comma-separated angle list, overrides --step. Used for the "
                         "stratified tabulation sample (equal numbers of mult-15 / mult-5 / even "
                         "/ odd angles) so the tabulation gradient can be measured under thinking "
                         "without paying for a full 1-degree sweep.")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--acts", action="store_true", help="cache the residual stream at all depths")
    ap.add_argument("--tag", default=None, help="name for a --degrees run")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.cond}_step{a.step}" if not a.degrees else f"{a.cond}_{a.tag or 'degs'}"
    jsonl = os.path.join(a.out, f"results_{tag}.jsonl")
    done = set()
    if os.path.exists(jsonl):                      # resume: never redo paid work
        for line in open(jsonl):
            try: r = json.loads(line); done.add((r["fn"], r["deg"]))
            except Exception: pass
        log(f"resuming: {len(done)} already on disk")

    degs = ([int(x) for x in a.degrees.split(",") if x.strip()] if a.degrees
            else list(range(0, 360, a.step)))
    tasks = [(fn, d) for d in degs for fn in ("sin", "cos", "tan") if (fn, d) not in done]
    log(f"{len(tasks)} asks to do, condition '{a.cond}', step {a.step} deg")

    from transformers import AutoTokenizer, AutoConfig
    import transformers
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"                      # so the last prompt token is index -1 for all
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    log(f"transformers {transformers.__version__}, tokenizer ok")
    try:
        from transformers import Qwen3_5ForCausalLM as Cls; name = "Qwen3_5ForCausalLM"
    except Exception:
        from transformers import AutoModelForCausalLM as Cls; name = "AutoModelForCausalLM"
    t0 = time.time()
    model = Cls.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(a.device).eval()
    layers = model.model.layers
    log(f"LOAD OK via {name} in {time.time()-t0:.0f}s | {len(layers)} layers")

    acts_store, handles = [], []
    if a.acts:
        buf = {}
        # generate() runs ONE forward pass per new token, so a naive hook ends up holding the
        # LAST GENERATED token, not the last prompt token. Verified on the laptop: max abs
        # discrepancy 3.08, i.e. a completely different vector. Only the PREFILL pass carries
        # the whole prompt (seq_len > 1); every decode step after it is seq_len == 1. So take
        # the prefill only, and arm the flag freshly for each batch.
        armed = {"v": False}
        def grab(i, h):
            if armed["v"] and h.shape[1] > 1:
                buf[i] = h[:, -1, :].detach().float().cpu()      # LAST prompt token, left-padded
        def mk(i):
            def hook(m, args, out):
                grab(i, out[0] if isinstance(out, tuple) else out)
            return hook
        handles = [layers[i].register_forward_hook(mk(i)) for i in range(len(layers))]
        emb_h = model.model.embed_tokens.register_forward_hook(
            lambda m, args, out: grab(-1, out))

    f = open(jsonl, "a")
    t_start = time.time(); n_done = 0
    for b0 in range(0, len(tasks), a.batch):
        chunk = tasks[b0:b0 + a.batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": prompt_for(fn, d)}],
                 add_generation_prompt=True, tokenize=False,
                 enable_thinking=(a.cond == "on")) for fn, d in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to(a.device)
        if a.acts: buf.clear(); armed["v"] = True
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        if a.acts:
            armed["v"] = False
            assert len(buf) == len(layers) + 1, f"hooks caught {len(buf)}, expected {len(layers)+1}"
            acts_store.append(np.stack([buf[i].numpy() for i in range(-1, len(layers))], axis=1).astype(np.float16))
        gen = out[:, enc["input_ids"].shape[1]:]
        for (fn, d), row in zip(chunk, gen):
            text = tok.decode(row, skip_special_tokens=True)
            val, dp, raw = parse(text)
            tv = truth(fn, d)
            rec = {"fn": fn, "deg": d, "cond": a.cond, "raw": raw, "value": val, "decimals": dp,
                   "truth": tv, "pole": tv is None,
                   "abs_err": (abs(val - tv) if (val is not None and tv is not None) else None),
                   "format_ok": dp == 5, "n_gen_tokens": int((row != tok.pad_token_id).sum()),
                   "text": text[:4000] if a.cond == "on" else None}
            f.write(json.dumps(rec) + "\n")
        f.flush(); n_done += len(chunk)
        rate = n_done / max(1e-9, time.time() - t_start)
        log(f"  {n_done}/{len(tasks)} done · {rate:.1f} asks/s · eta {(len(tasks)-n_done)/max(rate,1e-9)/60:.1f} min")
    f.close()
    for h in handles: h.remove()
    if a.acts: emb_h.remove()

    if a.acts and acts_store:
        # shard, so a resumed run never overwrites the activations the first run paid for
        k = 0
        while os.path.exists(os.path.join(a.out, f"acts_{tag}_{k}.npy")): k += 1
        A = np.concatenate(acts_store, 0)
        np.save(os.path.join(a.out, f"acts_{tag}_{k}.npy"), A)
        json.dump([{"fn": fn, "deg": d} for fn, d in tasks],
                  open(os.path.join(a.out, f"acts_index_{tag}_{k}.json"), "w"))
        log(f"ACTS saved {A.shape} -> acts_{tag}_{k}.npy (rows align with acts_index_{tag}_{k}.json)")
    log(f"DONE {tag} in {(time.time()-t_start)/60:.1f} min"
        + (f", peak ALLOCATED {torch.cuda.max_memory_allocated()/1e9:.1f} GB"
           f" / reserved {torch.cuda.max_memory_reserved()/1e9:.1f} GB" if a.device == "cuda" else ""))

if __name__ == "__main__":
    main()
