#!/usr/bin/env python3
"""MATH-SMOKE-01: first look at Qwen3.8-27B on a GPU.

Four stages, each writes its own file under --out BEFORE the next starts, so a kill at any
point leaves everything earlier on disk:
  1. arch.txt        str(model), config, per-layer type table, param count, the module path
                     of the decoder layers (needed for hooks).
  2. trace.json      one math question, thinking ON, greedy, full token ids + text + answer.
  3. acts/L{n}.npy   residual stream (output of decoder layer n) at EVERY token of the trace,
                     float16 [seq, hidden], for the layers in --layers. One forward pass over
                     the generated sequence with forward hooks — NOT collected during generate.
  4. done.json       timings, peak VRAM, file list. Presence of this file == run complete.
Every stage prints a timestamped verdict line. Dry-run on a laptop with --model Qwen/Qwen3.5-0.8B.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

# GSM8K test split, item 0 (answer 18). Swap with --question / --answer.
GSM8K_0 = ("Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and "
           "bakes muffins for her friends every day with four. She sells the remainder at the "
           "farmers' market daily for $2 per fresh duck egg. How much in dollars does she make "
           "every day at the farmers' market?")

def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)

def find_decoder_layers(model):
    """Return (path, ModuleList) for the decoder layer stack. Qwen3.5 VL checkpoints nest it as
    model.language_model.layers; text-only as model.layers. Discover, don't assume."""
    cands = [(n, m) for n, m in model.named_modules()
             if isinstance(m, torch.nn.ModuleList) and n.endswith("layers") and len(m) > 4]
    if not cands:
        raise RuntimeError("no decoder layer ModuleList found")
    cands.sort(key=lambda nm: -len(nm[1]))
    return cands[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--out", default="/root/out/smoke")
    ap.add_argument("--layers", default="0,15,31,47,63", help="decoder layer indices to cache")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--question", default=GSM8K_0)
    ap.add_argument("--answer", default="18")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                    ("mps" if torch.backends.mps.is_available() else "cpu"))
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "acts"), exist_ok=True)
    dtype = getattr(torch, a.dtype)
    t0 = time.time()

    # Official transformers doc (model_doc/qwen3_5): "Use Qwen3_5ForCausalLM for text-only
    # generation"; Qwen3_5ForConditionalGeneration is the multimodal class. Qwen3.8-27B ships
    # architectures=[Qwen3_5ForConditionalGeneration], model_type qwen3_5, so the text-only
    # class is the documented way to load it without the vision tower.
    from transformers import AutoTokenizer, AutoConfig
    import transformers
    tok = AutoTokenizer.from_pretrained(a.model)
    cfg = AutoConfig.from_pretrained(a.model)
    log(f"config loaded: architectures={cfg.architectures} model_type={cfg.model_type} transformers={transformers.__version__}")
    for m in ("fla", "causal_conv1d"):   # same doc: without both, DeltaNet layers silently fall back to slow PyTorch
        try: __import__(m); log(f"kernel {m}: OK")
        except Exception as e: log(f"kernel {m}: MISSING -> slow fallback ({type(e).__name__})")
    try:
        from transformers import Qwen3_5ForCausalLM
        model = Qwen3_5ForCausalLM.from_pretrained(a.model, dtype=dtype)
        loader = "Qwen3_5ForCausalLM"
    except Exception as e:
        log(f"Qwen3_5ForCausalLM failed ({type(e).__name__}: {str(e)[:160]}); falling back to AutoModelForCausalLM")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype)
        loader = "AutoModelForCausalLM"
    model = model.to(a.device).eval()
    log(f"LOAD OK via {loader} in {time.time()-t0:.0f}s")

    # ---- stage 1: architecture ----
    layers_path, layers = find_decoder_layers(model)
    tc = getattr(cfg, "text_config", cfg)
    n_params = sum(p.numel() for p in model.parameters())
    with open(os.path.join(a.out, "arch.txt"), "w") as f:
        f.write(f"model: {a.model}\nloader: {loader}\nparams: {n_params/1e9:.2f}B\n")
        f.write(f"decoder layers at: {layers_path}  (n={len(layers)})\n")
        f.write(f"hidden_size={getattr(tc,'hidden_size',None)} heads={getattr(tc,'num_attention_heads',None)} "
                f"kv_heads={getattr(tc,'num_key_value_heads',None)} vocab={getattr(tc,'vocab_size',None)}\n")
        lt = getattr(tc, "layer_types", None)
        if lt:
            f.write("layer types:\n")
            for i, t in enumerate(lt):
                f.write(f"  {i:3d} {t}\n")
        f.write("\n===== config =====\n" + cfg.to_json_string() + "\n")
        f.write("\n===== str(model) =====\n" + str(model) + "\n")
        f.write("\n===== one decoder layer, named_parameters =====\n")
        for n, p in layers[0].named_parameters():
            f.write(f"  {n:60s} {tuple(p.shape)}\n")
    log(f"ARCH OK: {len(layers)} layers at {layers_path}, {n_params/1e9:.2f}B params -> arch.txt")

    # ---- stage 2: one question, thinking on, greedy ----
    msgs = [{"role": "user", "content": a.question}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, enable_thinking=True)
    ids = ids["input_ids"] if hasattr(ids, "input_ids") else ids
    inp = torch.tensor([ids], device=a.device)
    t1 = time.time()
    # Rule 7 Q1: HF generate is silent until it returns (QWEN-01 ran 2 h like that). Stream tokens
    # to stdout so the log grows while the model thinks.
    from transformers import TextStreamer
    streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=False)
    log(f"GENERATING (max {a.max_new_tokens} new tokens, greedy, thinking on) — live text follows")
    with torch.no_grad():
        out = model.generate(inp, max_new_tokens=a.max_new_tokens, do_sample=False, streamer=streamer)
    print(flush=True)
    seq = out[0].tolist()
    gen = seq[len(ids):]
    text = tok.decode(gen)
    finished = tok.eos_token_id in gen or (getattr(model.generation_config, "eos_token_id", None) is not None
               and any(t in gen for t in (model.generation_config.eos_token_id
               if isinstance(model.generation_config.eos_token_id, list) else [model.generation_config.eos_token_id])))
    trace = {"model": a.model, "question": a.question, "gold": a.answer, "prompt_ids": ids,
             "gen_ids": gen, "text": text, "n_prompt": len(ids), "n_gen": len(gen),
             "finished": finished, "gen_seconds": round(time.time()-t1, 1),
             "gold_in_text": a.answer in text}
    json.dump(trace, open(os.path.join(a.out, "trace.json"), "w"), indent=1)
    log(f"TRACE OK: {len(gen)} tokens in {trace['gen_seconds']}s, finished={finished}, "
        f"gold '{a.answer}' in text={trace['gold_in_text']} -> trace.json")
    print("----- first 600 chars -----\n" + text[:600] + "\n----- last 300 chars -----\n" + text[-300:], flush=True)

    # ---- stage 3: residual stream at every token, via hooks, one forward pass ----
    want = [int(x) for x in a.layers.split(",") if x.strip() != ""]
    want = [i for i in want if i < len(layers)]
    store, handles = {}, []
    def mk(i):
        def hook(mod, args, output):
            h = output[0] if isinstance(output, tuple) else output
            store[i] = h[0].detach().to(torch.float16).cpu().numpy()   # [seq, hidden]
        return hook
    for i in want:
        handles.append(layers[i].register_forward_hook(mk(i)))
    t2 = time.time()
    with torch.no_grad():
        model(torch.tensor([seq], device=a.device))
    for h in handles: h.remove()
    for i in want:
        np.save(os.path.join(a.out, "acts", f"L{i}.npy"), store[i])
    shapes = {i: list(store[i].shape) for i in want}
    norms = {i: float(np.linalg.norm(store[i].astype(np.float32), axis=1).mean()) for i in want}
    log(f"ACTS OK: layers {want}, shape {shapes[want[0]]}, mean ||h|| per layer {json.dumps({k: round(v,1) for k,v in norms.items()})} "
        f"in {time.time()-t2:.1f}s -> acts/")
    assert shapes[want[0]][0] == len(seq), "activation rows != sequence length"

    # ---- stage 4: done ----
    peak = torch.cuda.max_memory_allocated()/1e9 if a.device == "cuda" else None
    done = {"total_seconds": round(time.time()-t0, 1), "peak_vram_gb": peak, "layers_path": layers_path,
            "layers_cached": want, "act_shapes": shapes, "mean_norms": norms,
            "files": sorted(os.listdir(a.out)) + sorted("acts/"+f for f in os.listdir(os.path.join(a.out,"acts")))}
    json.dump(done, open(os.path.join(a.out, "done.json"), "w"), indent=1)
    log(f"DONE in {done['total_seconds']}s, peak VRAM {peak} GB -> done.json")

if __name__ == "__main__":
    main()
