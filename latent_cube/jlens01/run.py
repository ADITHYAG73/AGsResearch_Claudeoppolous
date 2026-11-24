#!/usr/bin/env python3
"""JLENS-01: read Qwen3.5-4B with Anthropic's Jacobian lens (official code, Neuronpedia-fitted lens).

Parts (run in this order; each writes incrementally and resumes):
  sanity  — (a) unembed(final layer) == model logits; (b) the official walkthrough's boot prompt, J-lens vs logit lens;
            (c) save R1 of the forced sin-x hold capture for an offline comparison with CAPTURE-01 on S3.
  dm      — POSITIVE CONTROL: the paper's directed-modulation prompts (repo data/experiments/directed-modulation.json)
            on our model. Math + category targets; ranks of tracked target tokens at every source layer x carrier position.
  cube    — AG's question: every CAPTURE-01 capture (exact token ids from captures.json), J-lens AND logit-lens ranks of
            tracked tokens at every layer x position, top-20 J-lens tokens, and the J-transported states (fp16) for
            later CPU analysis with any hit definition.
Usage: python run.py --out OUT --part sanity|dm|cube [--limit N] [--model ...] [--lens-file ...] [--fake-lens]
"""
import argparse, json, os, sys, time, random
import torch

def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--part", required=True, choices=["sanity", "dm", "cube"])
ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
ap.add_argument("--revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
ap.add_argument("--lens-repo", default="neuronpedia/jacobian-lens"); ap.add_argument("--lens-rev", default="qwen-n1000")
ap.add_argument("--lens-file", default="qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt")
ap.add_argument("--fake-lens", action="store_true", help="dry run only: identity J matrices")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--carriers", type=int, default=3, help="carrier sentences per (target, phrasing) in dm")
ap.add_argument("--dm-json", default=os.path.join(os.path.dirname(__file__), "directed-modulation.json"))
ap.add_argument("--captures", default=os.path.join(os.path.dirname(__file__), "captures.json"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"

import transformers, jlens
from jlens.hooks import ActivationRecorder
log(f"VERSIONS torch {torch.__version__} transformers {transformers.__version__} device {dev} "
    f"{torch.cuda.get_device_name(0) if dev == 'cuda' else ''}")
tok = transformers.AutoTokenizer.from_pretrained(a.model, revision=a.revision)
hf = transformers.AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=torch.bfloat16).to(dev).eval()
M = jlens.from_hf(hf, tok, force_bos=False)   # we tokenise ourselves (chat template, no BOS — as CAPTURE-01)
NL, D = M.n_layers, M.d_model
log(f"MODEL READY {type(hf).__name__} n_layers={NL} d_model={D} layout={M.layout}")
if a.fake_lens:
    lens = jlens.JacobianLens({l: torch.eye(D) for l in range(NL - 1)}, n_prompts=0, d_model=D)
else:
    lens = jlens.JacobianLens.from_pretrained(a.lens_repo, filename=a.lens_file, revision=a.lens_rev)
log(f"LENS {lens}")
assert lens.source_layers == list(range(NL - 1)), lens.source_layers
J = {l: lens.jacobians[l].to(dev) for l in lens.source_layers}
SRC = lens.source_layers
V = hf.lm_head.weight.shape[0]

def states(ids):
    """Block outputs (residual stream after each layer) for one sequence: {layer: [T, D]}."""
    x = torch.tensor([ids], device=dev)
    with ActivationRecorder(M.layers, at=range(NL)) as rec, torch.no_grad():
        M.forward(x)
        return {l: rec.activations[l][0].detach() for l in range(NL)}

def ranks_of(logits, tids):
    """logits [n, V] -> rank (0 = top) of each tracked id: [n, k]. Ties counted strictly above."""
    # one tracked id at a time: an [n, k, V] comparison would be ~GBs for long captures
    return torch.stack([(logits > logits[:, t:t + 1]).sum(-1) for t in tids], dim=-1)   # [n, k]

def readout(acts, positions, tids, topk=0, jl=True):
    """Per source layer: ranks of tids (and optional top-k ids) at the given positions."""
    R, TOP = [], []
    for l in SRC:
        h = acts[l][positions].float()
        if jl: h = h @ J[l].T
        lg = M.unembed(h).float()
        R.append(ranks_of(lg, tids))
        if topk: TOP.append(lg.topk(topk, dim=-1).indices)
    return torch.stack(R), (torch.stack(TOP) if topk else None)     # [L, n, k], [L, n, topk]

def single_ids(words):
    """Map candidate surface strings to single-token ids (drop multi-token ones). Returns {string: id}."""
    out = {}
    for w in words:
        e = tok(w, add_special_tokens=False).input_ids
        if len(e) == 1: out[w] = e[0]
    return out

def chat(user, forced):
    text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    pid = tok(text, add_special_tokens=False).input_ids
    fid = tok(forced, add_special_tokens=False).input_ids
    return text, pid + fid, len(pid)

# ---------------------------------------------------------------- sanity
if a.part == "sanity":
    ids = tok("Fact: The currency used in the country shaped like a boot is", add_special_tokens=False).input_ids
    acts = states(ids)
    with torch.no_grad():
        ref = hf(torch.tensor([ids], device=dev)).logits[0].float()
    d = (M.unembed(acts[NL - 1]).float() - ref).abs().max().item()
    log(f"CHECK unembed(final block) vs model logits: max|diff| = {d:.4g}")
    pos = len(ids) - 2   # the walkthrough reads positions=[-2] ("boot")
    for l in [NL // 4, NL // 2, NL // 4 * 3, NL - 2]:
        for name, jl in (("logit-lens", False), ("J-lens", True)):
            h = acts[l][[pos]].float()
            if jl: h = h @ J[l].T
            top = M.unembed(h).float()[0].topk(5).indices.tolist()
            log(f"BOOT L{l:>2} {name:10s} {[tok.decode([t]) for t in top]}")
    log(f"BOOT model next token: {[tok.decode([t]) for t in ref[-1].topk(5).indices.tolist()]}")
    caps = json.load(open(a.captures))
    c = [c for c in caps if c["id"] == "tay|sin x|2|hold" and c["variant"] == "forced"][0]
    acts = states(c["token_ids"])
    torch.save({"R1": torch.stack([acts[l] for l in range(NL)]).half().cpu(), "token_ids": c["token_ids"]},
               f"{a.out}/sanity_R1_sinx2hold_forced.pt")
    log("SANITY DONE")

# ---------------------------------------------------------------- dm (positive control)
if a.part == "dm":
    dm = json.load(open(a.dm_json))
    NUM = {str(i): w for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                            "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
    trials = []
    rng = random.Random(20261124)
    carriers = dm["carrier_sentences"]
    for kind, targets in (("math", dm["math_problems"]), ("category", dm["topic_categories"])):
        for ti, t in enumerate(targets):
            x = t["expr"] if kind == "math" else t["name"]
            if kind == "math":
                ans = str(t["answer"]); w = NUM.get(ans.lstrip("-"))
                cands = [ans, " " + ans] + ([w, " " + w, w.capitalize(), " " + w.capitalize()] if w else [])
            else:
                cands = [s for m in t["members"] for s in (m, " " + m, m.lower(), " " + m.lower())]
            tracked = single_ids(cands)
            cs = rng.sample(range(len(carriers)), a.carriers)
            for ci in cs:
                for ph in [{"name": "none", "group": "none", "text": None}] + dm["phrasings"]:
                    trials.append(dict(kind=kind, target_index=ti, x=x, tracked=tracked, carrier_index=ci,
                                       phrasing=ph["name"], group=ph["group"],
                                       group_kind=dm["group_kind"].get(ph["group"], "baseline"), text=ph["text"]))
    if a.limit: trials = trials[: a.limit]
    path = f"{a.out}/dm_trials.jsonl"
    done = sum(1 for _ in open(path)) if os.path.exists(path) else 0
    log(f"DM {len(trials)} trials ({done} already done) -> {path}")
    t0 = time.time()
    with open(path, "a") as f:
        for n, tr in enumerate(trials):
            if n < done: continue
            carrier = carriers[tr["carrier_index"]]
            # User turn construction is OURS (the repo gives phrasings + carriers, not the exact user message):
            instr = "" if tr["text"] is None else " " + tr["text"].format(x=tr["x"])
            user = f"Write the following sentence.{instr}\nSentence: {carrier}"
            text, ids, P = chat(user, carrier)
            acts = states(ids)
            positions = list(range(P, len(ids)))            # the teacher-forced carrier tokens
            tids = list(tr["tracked"].values())
            R, TOP = readout(acts, positions, tids, topk=5)
            rec = dict(n=n, **{k: tr[k] for k in ("kind", "target_index", "x", "carrier_index", "phrasing", "group", "group_kind")},
                       tracked=tr["tracked"], user=user, prompt_len=P, token_ids=ids,
                       ranks=R.tolist(), top5=TOP.tolist(), min_rank=int(R.min()) if R.numel() else None)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            if n % 50 == 0 or n == len(trials) - 1:
                el = time.time() - t0
                log(f"[dm {n+1}/{len(trials)}] {tr['kind']} {tr['x']!r} {tr['phrasing']} min_rank={rec['min_rank']} "
                    f"eta {el/(n+1-done)*(len(trials)-n-1)/60:.1f} min")
    log("DM DONE")

# ---------------------------------------------------------------- cube (AG's question)
if a.part == "cube":
    caps = json.load(open(a.captures))
    if a.limit: caps = caps[: a.limit]
    words = ["³", "x", " x", "6", " six", "six", " cube", " cubic", "3", " three", " third", "/", "−", "-x", " -",
             "sin", " sin", " Taylor", " series", "²", "2", " two", " square", " squared", "1", "0", " zero",
             "5", "⁵", "4", "⁴", "!", " factorial", " Paris", " capital", " France", " minus", " negative"]
    tracked = single_ids(words)
    json.dump(tracked, open(f"{a.out}/cube_tracked_ids.json", "w"), ensure_ascii=False)
    log(f"CUBE {len(caps)} captures; tracked single-token ids: {tracked}")
    tids = list(tracked.values())
    os.makedirs(f"{a.out}/cube", exist_ok=True)
    t0 = time.time()
    from safetensors.torch import save_file
    for n, c in enumerate(caps):
        fn = f"{a.out}/cube/{n:03d}"
        if os.path.exists(fn + ".json"): continue
        ids = c["token_ids"]; T = len(ids)
        acts = states(ids)
        pos = list(range(T))
        RJ, TOPJ = readout(acts, pos, tids, topk=20, jl=True)
        RL, _ = readout(acts, pos, tids, topk=0, jl=False)
        transported = torch.stack([(acts[l].float() @ J[l].T) for l in SRC]).half()
        save_file({"ranks_jlens": RJ.int().cpu().contiguous(), "ranks_logitlens": RL.int().cpu().contiguous(),
                   "top20_jlens": TOPJ.int().cpu().contiguous(), "transported": transported.cpu().contiguous()},
                  fn + ".safetensors")
        json.dump(dict(index=n, **{k: c[k] for k in c if k != "token_ids"}, token_ids=ids,
                       tokens=tok.convert_ids_to_tokens(ids)), open(fn + ".json", "w"), ensure_ascii=False)
        if n % 10 == 0 or n == len(caps) - 1:
            el = time.time() - t0
            log(f"[cube {n+1}/{len(caps)}] {c['id']} [{c['variant']}] T={T} "
                f"eta {el/(n+1)*(len(caps)-n-1)/60:.1f} min")
    log("CUBE DONE")
