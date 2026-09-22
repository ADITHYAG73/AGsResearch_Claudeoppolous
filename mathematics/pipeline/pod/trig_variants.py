#!/usr/bin/env python3
"""TRIG-01b. Is the integer-degree accuracy RECALL of a table, or COMPUTATION?

TRIG-01 found ~97% exact-to-5dp for sin/cos on integer degrees in Q1. That number on its own
cannot distinguish two very different mechanisms, because a 5-decimal trig table for integer
degrees is roughly a thousand numbers and is all over the training data.

So vary the INPUT in ways a printed table does not cover, holding the arithmetic difficulty
roughly fixed, and see which variations the accuracy survives:

  int      d            the control - reproduces TRIG-01 inside this run
  half     d + 0.5      half-degree tables DO exist, so this is the mildest departure
  dec2     d + 0.ab     two decimal places, pseudo-random per angle - essentially untabulated
  over360  d + 360      identical value to `int`, but requires reducing mod 360 first
  neg      -d           identical magnitude to `int`, requires the odd/even symmetry

READING IT. `over360` and `neg` are the sharp ones: the ANSWER is the same real number as the
control, so any drop is the reduction step failing, not the function being harder. A drop on
`dec2` alone is weaker evidence - a fractional input could be harder to compute as well as
harder to look up - so `dec2` is suggestive, while a drop on `over360`/`neg` is diagnostic.
Conversely, if `dec2` holds up near the control, that is strong evidence for real computation,
because nothing plausibly tabulated covers sin(37.42 degrees).
"""
import argparse, json, math, os, re, time
import numpy as np, torch

FNS = {"sin": math.sin, "cos": math.cos, "tan": math.tan}
def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)

def variants(d, rng):
    """(name, angle_asked, angle_equivalent_for_truth)"""
    ab = rng.randrange(1, 100) / 100.0          # .01 - .99, never .00
    return [("int", float(d), float(d)),
            ("half", d + 0.5, d + 0.5),
            ("dec2", round(d + ab, 2), round(d + ab, 2)),
            ("over360", float(d + 360), float(d)),
            ("neg", float(-d), float(-d))]

def truth(fn, deg):
    if fn == "tan" and abs((deg % 180) - 90) < 1e-9: return None
    return FNS[fn](math.radians(deg))

def fmt(x):
    return str(int(x)) if float(x).is_integer() else f"{x:g}"

def prompt_for(fn, deg):
    return (f"Compute {fn}({fmt(deg)} degrees). "
            f"Respond with only the numeric value, rounded to exactly 5 decimal places.")

NUM = re.compile(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?")
def parse(text):
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
    ap.add_argument("--max-deg", type=int, default=90, help="base angles 0..max-deg-1 (Q1)")
    ap.add_argument("--base-stride", type=int, default=1,
                    help="subsample base angles; the RNG is still keyed to the full 0..max-deg "
                         "sequence so dec2 offsets match a stride-1 run exactly (paired design)")
    ap.add_argument("--variants", default="int,half,dec2,over360,neg")
    ap.add_argument("--thinking", default="off", choices=["off", "on"])
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--tag", default="variants")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import random
    rng = random.Random(a.seed)

    jsonl = os.path.join(a.out, f"results_{a.tag}.jsonl")
    done = set()
    if os.path.exists(jsonl):
        for line in open(jsonl):
            try:
                r = json.loads(line); done.add((r["fn"], r["variant"], r["asked"]))
            except Exception: pass
        log(f"resuming: {len(done)} already on disk")

    want = [v.strip() for v in a.variants.split(",") if v.strip()]
    tasks = []
    n_base = 0
    for d in range(a.max_deg):
        # draw for EVERY d so the dec2 offset for a given base angle is identical whatever the
        # stride - a strided run is then exactly paired with the stride-1 run already on disk
        vs = variants(d, rng)
        if d % a.base_stride: continue
        n_base += 1
        for name, asked, equiv in vs:
            if name not in want: continue
            for fn in ("sin", "cos", "tan"):
                if (fn, name, asked) not in done:
                    tasks.append((fn, name, asked, equiv, d))
    log(f"{len(tasks)} asks to do ({n_base} base angles x {len(want)} variants x 3 functions), "
        f"thinking={a.thinking}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    from transformers import Qwen3_5ForCausalLM as Cls
    t0 = time.time()
    model = Cls.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(a.device).eval()
    log(f"LOAD OK in {time.time()-t0:.0f}s")

    f = open(jsonl, "a"); t_start = time.time(); n = 0
    for b0 in range(0, len(tasks), a.batch):
        chunk = tasks[b0:b0 + a.batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": prompt_for(fn, asked)}],
                 add_generation_prompt=True, tokenize=False,
                 enable_thinking=(a.thinking == "on"))
                 for fn, _, asked, _, _ in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to(a.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        for (fn, name, asked, equiv, base), row in zip(chunk, gen):
            text = tok.decode(row, skip_special_tokens=True)
            val, dp, raw = parse(text)
            tv = truth(fn, equiv)
            f.write(json.dumps({
                "fn": fn, "variant": name, "asked": asked, "equiv": equiv, "base_deg": base,
                "raw": raw, "value": val, "decimals": dp, "truth": tv, "pole": tv is None,
                "abs_err": (abs(val - tv) if (val is not None and tv is not None) else None),
                "format_ok": dp == 5, "thinking": a.thinking,
                "n_gen_tokens": int((row != tok.pad_token_id).sum()),
                "text": text[:4000] if a.thinking == "on" else None}) + "\n")
        f.flush(); n += len(chunk)
        r = n / max(1e-9, time.time() - t_start)
        log(f"  {n}/{len(tasks)} done · {r:.1f} asks/s · eta {(len(tasks)-n)/max(r,1e-9)/60:.1f} min")
    f.close()
    log(f"DONE {a.tag} in {(time.time()-t_start)/60:.1f} min")

if __name__ == "__main__":
    main()
