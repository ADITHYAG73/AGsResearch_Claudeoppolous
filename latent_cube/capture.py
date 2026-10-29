#!/usr/bin/env python3
"""LATENT-CUBE capture: run Qwen3.5-4B (text-only) on each prompt, thinking OFF, batch 1 (no padding),
and save every internal we can afford, one file per prompt, written as soon as that prompt finishes.

Per prompt (all token positions = prompt + generated reply, teacher-forced in ONE forward pass):
  R0  embedding output                         [T, 2560]
  R1  output of each of the 32 layers          [32, T, 2560]
  R2  token-mixer output (DeltaNet / attn)     [32, T, 2560]
  R4  MLP neuron values (down_proj input)      [32, T, 9216]
  R5  MLP output                               [32, T, 2560]
  logits                                       [T, 248320]
  attn  softmax weights, 8 full-attn layers    [8, 16, T, T]
  S   DeltaNet memory at 3 positions (fp32)    [3, 24, 32, 128, 128]   (--states)
All activations fp16 except S. Plus meta.json: prompt, chat-template text, token ids + strings, the
generated reply, finish reason, readout positions, and the sanity-check numbers below.

Sanity checks (logged for every prompt, printed loudly in --debug):
  1. teacher-forced argmax at each reply position == the greedily generated token
  2. final RMSNorm(R1[31]) @ embed^T == the model's logits  (proves R1[31] is the pre-norm residual)
  3. R1[l] == R1[l-1] + R2[l] + R5[l]                        (proves the taps are the residual pieces)
Usage: python capture.py --prompts prompts.jsonl --out /root/lc/out [--ids a,b] [--limit 2] [--debug] [--states]
"""
import argparse, json, os, sys, time
import torch

def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)

ap = argparse.ArgumentParser()
ap.add_argument("--prompts", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
ap.add_argument("--revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
ap.add_argument("--ids", default=""); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--max-new", type=int, default=2048)  # safety cap only; every hit is reported
ap.add_argument("--states", action="store_true", help="also save DeltaNet memory at 3 positions")
ap.add_argument("--debug", action="store_true")
a = ap.parse_args()

from transformers import AutoTokenizer, Qwen3_5ForCausalLM
import transformers
from safetensors.torch import save_file
log(f"VERSIONS torch {torch.__version__} transformers {transformers.__version__} gpu {torch.cuda.get_device_name(0)}")
try:
    import fla; log(f"fla {fla.__version__}")
except Exception as e:
    log(f"WARNING fla not importable ({e}): DeltaNet layers will use the slow PyTorch path (still correct)")

tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
model = Qwen3_5ForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=torch.bfloat16,
                                           attn_implementation="eager").cuda().eval()
lm = model.model
L = lm.layers
types = [layer.block_type for layer in L]
FULL = [i for i, t in enumerate(types) if t == "full_attention"]
LIN = [i for i, t in enumerate(types) if t == "linear_attention"]
log(f"MODEL READY {len(L)} layers, full attention at {FULL}")
assert len(L) == 32 and FULL == [3, 7, 11, 15, 19, 23, 27, 31], FULL

# ---- hooks: each stores the CURRENT forward's tensor; cleared before every capture pass ----
cap = {}
def keep(name, x): cap[name] = x.detach()
hooks = [lm.embed_tokens.register_forward_hook(lambda m, i, o: keep("R0", o[0]))]
for li, layer in enumerate(L):
    hooks.append(layer.register_forward_hook(lambda m, i, o, li=li: keep(f"R1.{li}", (o[0] if isinstance(o, tuple) else o)[0])))
    mixer = layer.linear_attn if types[li] == "linear_attention" else layer.self_attn
    hooks.append(mixer.register_forward_hook(lambda m, i, o, li=li: keep(f"R2.{li}", (o[0] if isinstance(o, tuple) else o)[0])))
    hooks.append(layer.mlp.down_proj.register_forward_pre_hook(lambda m, i, li=li: keep(f"R4.{li}", i[0][0])))
    hooks.append(layer.mlp.register_forward_hook(lambda m, i, o, li=li: keep(f"R5.{li}", o[0])))

IM_END = tok.convert_tokens_to_ids("<|im_end|>")
FORCED = {"imagine": "Okay.", "hold": "The capital of France is Paris."}

def capture_one(r, n, fn, vname, text, P, full, ftext, free_reply, finish):
    T = full.shape[1]
    ids = full[0].tolist()
    gen = ids[P:]
    # readout positions: last token of the user's own words (just before the first <|im_end|>),
    # last prompt token (model about to answer), last reply token
    pos = {"instr_end": ids.index(IM_END) - 1, "prompt_end": P - 1, "reply_end": T - 1}
    cap.clear()
    with torch.no_grad():
        out = model(full, output_attentions=True, use_cache=False)
    logits = out.logits[0]
    tens = {"R0": cap["R0"].half()}
    for k in ("R1", "R2", "R4", "R5"):
        tens[k] = torch.stack([cap[f"{k}.{li}"] for li in range(32)]).half()
    tens["logits"] = logits.half()
    att = [x for x in out.attentions if x is not None]          # hybrid model returns only the 8 attention layers
    assert len(att) == len(FULL), len(att)
    tens["attn"] = torch.stack([x[0] for x in att]).half()
    # ---- sanity checks on the RECORDING (not on the model's answers) ----
    am = logits[P - 1:T - 1].argmax(-1).tolist()
    s1 = sum(int(x != y) for x, y in zip(am, gen)) if ftext is None else -1   # n/a for a written reply
    with torch.no_grad():
        relog = model.lm_head(lm.norm(torch.stack([cap["R1.31"]]).to(torch.bfloat16)))[0]
    s2 = (relog.float() - logits.float()).abs().max().item()
    s3 = max(((cap[f"R1.{li}"].float() - (cap[f"R1.{li-1}"] if li else cap["R0"]).float()
               - cap[f"R2.{li}"].float() - cap[f"R5.{li}"].float()).abs().max().item()) for li in range(32))
    if a.states:   # DeltaNet memory at the readout positions (one prefix pass per position)
        S = []
        for pname in ("instr_end", "prompt_end", "reply_end"):
            with torch.no_grad():
                c = model(full[:, : pos[pname] + 1], use_cache=True).past_key_values
            S.append(torch.stack([c.layers[li].recurrent_states[0][0].float() for li in LIN]))
        tens["S"] = torch.stack(S)                                     # [3, 24, 32, 128, 128]
    base = f"{a.out}/{fn}__{vname}"
    save_file({k: v.contiguous().cpu() for k, v in tens.items()}, f"{base}.safetensors")
    meta = dict(r, variant=vname, chat_text=text, token_ids=ids, tokens=tok.convert_ids_to_tokens(ids),
                prompt_len=P, captured_reply=tok.decode(gen, skip_special_tokens=True), written_reply=ftext,
                free_reply=free_reply, free_reply_finish=finish, positions=pos,
                check_argmax_mismatch=s1, check_logits_maxdiff=s2, check_residual_sum_maxdiff=s3,
                saved={k: list(v.shape) for k, v in tens.items()})
    json.dump(meta, open(f"{base}.json", "w"), ensure_ascii=False)
    # full texts, never truncated (global rule 10)
    log(f"  {r['id']} [{vname}] T={T} free_reply_finish={finish} checks: argmax_mismatch={s1} "
        f"logits_diff={s2:.3g} resid_diff={s3:.3g} size={os.path.getsize(base + '.safetensors')/2**20:.0f}MB")
    log(f"  CAPTURED REPLY ({vname}): {meta['captured_reply']!r}")

rows = [json.loads(l) for l in open(a.prompts)]
if a.ids: want = set(a.ids.split(",")); rows = [r for r in rows if r["id"] in want]
if a.limit: rows = rows[: a.limit]
os.makedirs(a.out, exist_ok=True)
log(f"{len(rows)} prompts -> {a.out}")
t0 = time.time()

for n, r in enumerate(rows):
    fn = r["id"].replace("|", "__").replace("/", "_").replace("·", "dot").replace("(", "").replace(")", "").replace("^", "pow").replace("²", "sq")
    if os.path.exists(f"{a.out}/{fn}.done"): continue          # resume: finished prompts are skipped
    text = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    pid = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
    P = pid.shape[1]
    with torch.no_grad():
        g = model.generate(pid, max_new_tokens=a.max_new, do_sample=False, temperature=None, top_p=None, top_k=None)
    gen = g[0, P:].tolist()
    finish = "eos" if len(gen) < a.max_new else "length"   # generate stops early only on a stop token
    free_reply = tok.decode(gen, skip_special_tokens=True)
    # Dry run showed the model does NOT obey "do not write it": it says the term out loud. So every
    # imagine/hold prompt is captured TWICE (AG, 2026-10-29): "free" = the model's own reply, and
    # "forced" = a reply we write for it that deliberately does NOT contain the term (only Okay / Paris),
    # so any x^3 found at the reply positions was held, not said. "write" prompts: free only.
    variants = [("free", g, None)]
    forced = FORCED.get(r["template"])
    if forced is not None:
        fid = tok(forced, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
        variants.append(("forced", torch.cat([pid, fid], dim=1), forced))
    for vname, full, ftext in variants:
        capture_one(r, n, fn, vname, text, P, full, ftext, free_reply, finish)
    el = time.time() - t0
    open(f"{a.out}/{fn}.done", "w").write("ok")
    log(f"FREE REPLY: {free_reply!r}")
    if finish == "length": log(f"CAP HIT: {r['id']} free reply reached the {a.max_new}-token safety cap")
    log(f"[{n+1}/{len(rows)}] {r['id']} done; elapsed {el/60:.1f} min; eta {el/(n+1)*(len(rows)-n-1)/60:.1f} min")
log("CAPTURE DONE")
