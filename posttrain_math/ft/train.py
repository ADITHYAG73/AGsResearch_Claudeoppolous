#!/usr/bin/env python3
"""FT-01 full fine-tune of Qwen3.5-4B on Nemotron IFC-v4 (multi-turn IF). Runs ON THE POD.

Read before writing (transformers 5.17.0, models/qwen3_5/modeling_qwen3_5.py):
  * Qwen3_5ForConditionalGeneration = model.visual + model.language_model + lm_head (tied to
    model.language_model.embed_tokens). We load THIS class so saved key names match the hub
    checkpoint that vLLM reads.
  * `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`: the 15 multi-token-prediction tensors are
    dropped on load. save_checkpoint() copies them back unchanged from the original shards, and
    asserts the saved key set equals the original 738 keys.
  * forward(logits_to_keep=<index tensor>) computes logits only at those positions; we pass the
    positions that predict the trained (final assistant) turn, so no [8k x 248k] logits tensor.

Recipe (Claude's choices, recorded in the journal before launch):
  full fine-tune of the language model (vision tower frozen; it never sees an image here),
  fp32 master weights + AdamW, bf16 autocast, gradient checkpointing, batch 1 x accumulation 16,
  peak LR 1e-5, linear warmup 5% then cosine to 10%, betas (0.9, 0.95), wd 0, clip 1.0, 1 epoch.

Rule 7: a progress line every 10 micro-steps (loss, tok/s, peak memory, ETA); loss history to
loss.jsonl as it goes; checkpoints at 25/50/100% saved AND uploaded to S3 as they land;
memory is fixed after the first longest-example step (measured by --probe before the run).
"""
import argparse, json, math, os, shutil, time, glob
import torch

ORIG = "Qwen/Qwen3.5-4B"; REV = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)

def load_rows(path, k):
    rows = [json.loads(l) for l in open(path)]
    return rows[:k] if k else rows

def snapshot_dir():
    from huggingface_hub import snapshot_download
    return snapshot_download(ORIG, revision=REV)

def save_checkpoint(model, out_dir, snap, verify_equal=False):
    """Save bf16 weights with the hub's key names; restore mtp.*; copy tokenizer/config files."""
    from safetensors.torch import load_file, save_file
    from safetensors import safe_open
    os.makedirs(out_dir, exist_ok=True)
    # Save every tensor in the dtype the HUB checkpoint uses. The hub stores 48 tensors in fp32
    # (24 linear_attn.norm.weight + 24 linear_attn.A_log); the first version cast everything to
    # bf16, and the bit-exact save test caught it (max change 0.0039, 2026-10-18).
    orig_map = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]
    hub_dtype = {}
    for sh in sorted(set(orig_map.values())):
        with safe_open(f"{snap}/{sh}", "pt") as f:
            for k in f.keys():
                hub_dtype[k] = {"F32": torch.float32, "BF16": torch.bfloat16}[f.get_slice(k).get_dtype()]
    sd = {k: v.detach().to("cpu", hub_dtype.get(k, torch.bfloat16)).contiguous()
          for k, v in model.state_dict().items()}
    n32 = sum(v.dtype == torch.float32 for v in sd.values())
    log(f"  save dtypes: {n32} tensors fp32 (hub has {sum(d == torch.float32 for d in hub_dtype.values())}), rest bf16")
    sd.pop("lm_head.weight", None)                       # tied; the hub checkpoint has no lm_head
    model.save_pretrained(out_dir, state_dict=sd, max_shard_size="4GB", safe_serialization=True)
    orig_idx = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]
    new_idx_path = f"{out_dir}/model.safetensors.index.json"
    if os.path.exists(new_idx_path):
        new_idx = json.load(open(new_idx_path))
    else:                                                # single shard: build an index
        names = list(safe_open(f"{out_dir}/model.safetensors", "pt").keys())
        new_idx = {"metadata": {}, "weight_map": {n: "model.safetensors" for n in names}}
    have = set(new_idx["weight_map"])
    missing = [k for k in orig_idx if k not in have]
    extra = [k for k in have if k not in orig_idx]
    log(f"  save: {len(have)} keys written; missing vs hub {len(missing)} (e.g. {missing[:3]}); extra {extra[:5]}")
    assert all(k.startswith("mtp.") for k in missing), f"non-mtp keys missing: {[k for k in missing if not k.startswith('mtp.')][:10]}"
    assert not extra, f"keys not in the hub checkpoint: {extra[:10]}"
    if missing:
        mtp = {}
        for k in missing:
            with safe_open(f"{snap}/{orig_idx[k]}", "pt") as f:
                mtp[k] = f.get_tensor(k)
        save_file(mtp, f"{out_dir}/model-mtp.safetensors", metadata={"format": "pt"})
        for k in missing: new_idx["weight_map"][k] = "model-mtp.safetensors"
    json.dump(new_idx, open(new_idx_path, "w"), indent=1)
    assert set(new_idx["weight_map"]) == set(orig_idx), "key set differs from the hub checkpoint"
    # tokenizer, chat template, processor configs and the ORIGINAL config.json (same architecture)
    for f in os.listdir(snap):
        if not f.endswith(".safetensors") and f != "model.safetensors.index.json":
            shutil.copy(f"{snap}/{f}", f"{out_dir}/{f}")
    if verify_equal:   # untrained model: every tensor must round-trip bit-exactly
        bad = 0
        shards = sorted(set(orig_idx.values()))
        for sh in shards:
            o = load_file(f"{snap}/{sh}")
            for k, v in o.items():
                fn = new_idx["weight_map"][k]
                with safe_open(f"{out_dir}/{fn}", "pt") as f:
                    w = f.get_tensor(k)
                if w.dtype != v.dtype or w.shape != v.shape or not torch.equal(w, v): bad += 1
        log(f"  save round-trip check: {len(orig_idx)} tensors, {bad} differ from the hub checkpoint")
        assert bad == 0, "SAVE PATH CHANGES WEIGHTS"
    log(f"  checkpoint written: {out_dir}")

def s3_upload_dir(local, prefix):
    import boto3
    s3 = boto3.client("s3", region_name="ap-south-1")
    bucket = os.environ["S3_BUCKET"]
    for p in sorted(glob.glob(f"{local}/**", recursive=True)):
        if os.path.isfile(p):
            key = f"{prefix}/{os.path.relpath(p, local)}"
            s3.upload_file(p, bucket, key)
    # marker LAST: eval pods wait for it, so they never start on a half-uploaded checkpoint
    s3.put_object(Bucket=bucket, Key=f"{prefix}/_COMPLETE", Body=b"ok")
    log(f"  uploaded {local} -> s3://{bucket}/{prefix} (+_COMPLETE)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=0, help="use the first k rows (0 = all)")
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--probe", action="store_true", help="memory/throughput probe only; no optimizer step")
    ap.add_argument("--save-test", action="store_true", help="save the untrained model and verify bit-exact")
    ap.add_argument("--s3-prefix", default="")
    a = ap.parse_args()
    from transformers import Qwen3_5ForConditionalGeneration
    snap = snapshot_dir()
    torch.manual_seed(0)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(snap, dtype=torch.float32).cuda()
    model.config.use_cache = False
    for p in model.model.visual.parameters(): p.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"loaded {type(model).__name__}; trainable {ntrain/1e9:.3f}B params (vision frozen)")
    # DEBUG: which kernels are live (silent fallback to slow torch code is a known trap)
    import transformers.models.qwen3_5.modeling_qwen3_5 as M
    log(f"  kernels: chunk_gated_delta_rule={getattr(M.torch_chunk_gated_delta_rule,'__module__','?')} "
        f"causal_conv1d_fn={getattr(M.causal_conv1d_fn,'__module__','?')}")

    if a.save_test:
        save_checkpoint(model, f"{a.out}/s000_savetest", snap, verify_equal=True)
        log("SAVE TEST PASSED"); return

    rows = load_rows(a.data, a.k)
    def step_loss(r):
        ids = torch.tensor(r["input_ids"], device="cuda")[None]
        ls = r["label_start"]; L = ids.shape[1]
        pos = torch.arange(ls - 1, L - 1, device="cuda")
        tgt = ids[0, ls:L]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=ids, logits_to_keep=pos)
        logits = out.logits[0].float()
        return torch.nn.functional.cross_entropy(logits, tgt), L

    if a.probe:
        # Memory: the 3 longest rows. Speed: timed only on rows 8..23, AFTER warm-up — the first
        # version timed the first rows, which include one-off kernel autotuning, and reported
        # 267 tok/s against 3,087 in the real run (2026-10-18).
        longest = sorted(range(len(rows)), key=lambda i: -len(rows[i]["input_ids"]))[:3]
        torch.cuda.reset_peak_memory_stats()
        for i in longest + list(range(8)):
            loss, L = step_loss(rows[i]); loss.backward()
            log(f"  warm/mem len {L:5d} loss {loss.item():.3f} peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
        torch.cuda.synchronize(); t0 = time.time(); toks = 0
        for i in range(8, 24):
            loss, L = step_loss(rows[i]); loss.backward(); toks += L
            log(f"  probe len {L:5d} loss {loss.item():.3f} peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
        torch.cuda.synchronize()
        # add the optimizer state the real run will hold (AdamW: 2 fp32 tensors per trainable param)
        log(f"PROBE peak activations+grads {torch.cuda.max_memory_allocated()/2**30:.1f} GiB "
            f"+ AdamW state {2*ntrain*4/2**30:.1f} GiB; card {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB; "
            f"{toks/(time.time()-t0):.0f} tok/s (steady state, rows 8-23)")
        return

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    total = math.ceil(len(rows) / a.accum); warm = max(1, int(0.05 * total))
    def lr_at(s):
        if s < warm: return a.lr * (s + 1) / warm
        pr = (s - warm) / max(1, total - warm)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * pr)))
    saves = {max(1, round(total * f)): tag for f, tag in [(0.25, "s025"), (0.5, "s050"), (1.0, "s100")]}
    log(f"rows {len(rows)}  optimizer steps {total}  warmup {warm}  saves at {saves}")
    lossf = open(f"{a.out}/loss.jsonl", "a")
    step, t0, toks, run = 0, time.time(), 0, []
    for i, r in enumerate(rows):
        loss, L = step_loss(r)
        (loss / a.accum).backward(); toks += L; run.append(loss.item())
        if (i + 1) % a.accum == 0 or i == len(rows) - 1:
            for g in opt.param_groups: g["lr"] = lr_at(step)
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); opt.zero_grad(set_to_none=True); step += 1
            lossf.write(json.dumps({"step": step, "loss": sum(run[-a.accum:]) / len(run[-a.accum:]),
                                    "lr": lr_at(step - 1), "grad_norm": gn.item()}) + "\n"); lossf.flush()
            if step in saves:
                d = f"{a.out}/Qwen3.5-4B-ifc4-{saves[step]}"
                save_checkpoint(model, d, snap)
                if a.s3_prefix: s3_upload_dir(d, f"{a.s3_prefix}/{os.path.basename(d)}")
                log(f"CHECKPOINT {saves[step]} step {step}")
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            log(f"  {i+1}/{len(rows)} step {step}/{total} loss(last10) {sum(run[-10:])/10:.4f} "
                f"{toks/el:.0f} tok/s peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB "
                f"eta {el/(i+1)*(len(rows)-i-1)/60:.1f} min")
    log("TRAIN DONE")

if __name__ == "__main__":
    main()
