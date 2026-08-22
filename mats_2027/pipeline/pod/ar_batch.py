"""Batched AR (critic) reconstruction.

WHY. The official NLACritic.reconstruct() (natural_language_autoencoders/nla_inference.py:641)
scores ONE explanation per forward pass:

    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
    h   = backbone.model(ids, use_cache=False).last_hidden_state[0, -1]   # last token
    return value_head(h)

That is correct for a single sequence and leaves the GPU idle between calls (0% utilisation
observed in NOISE-01). ABLATE-01 produced 2303 variants; one at a time is hours on the meter.

SILENT-CORRUPTION RISKS once padding is introduced. None of them raise; all return plausible
numbers. MEASURED on CPU with a tiny Llama (deviation of the padded row from the
single-sequence answer), rather than asserted:

  1. `[0, -1]` takes the last POSITION. With RIGHT padding that is a PAD token, so you score
     padding instead of the explanation.          MEASURED: 1.51  -> REAL
     -> fix: LEFT padding, so position -1 is the last real token in every row.
  2. No attention_mask (unnecessary for one sequence). A padded batch without a mask lets
     real tokens attend to PAD.                    MEASURED: 1.56  -> REAL
     -> fix: always pass attention_mask.
  3. position_ids default to arange(seq_len), so with LEFT padding the real tokens are all
     shifted.                                      MEASURED: 2.1e-7 vs 2.4e-7 -> NO EFFECT
     Gemma-3 and Llama use ROTARY position embeddings, which are RELATIVE: attention depends
     only on the GAP between positions. Left padding shifts every real token by the same
     amount, so it cancels exactly. This WOULD matter for learned-absolute position
     embeddings (GPT-2, BERT).
     -> we still pass mask-derived position_ids: harmless, explicit, and correct if this is
        ever pointed at a non-RoPE model. But it is NOT what makes batching correct here -
        (1) and (2) are.

Correctness contract, asserted by tests/test_ar_batch.py on CPU with a tiny random model:
    reconstruct_batch(texts)[i]  ==  single-sequence reconstruct(texts[i])   (float tolerance)
for texts of DIFFERENT lengths. That is what catches all three bugs without a GPU.
"""
import torch

DEBUG = False          # set True (or ar_batch.DEBUG = True) to print shapes for every batch


def _encode(tokenizer, prompts, device):
    """Left-padded batch + attention mask + mask-derived position ids."""
    prev = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        add_special_tokens=True)   # BOS matters for Gemma (see nla_inference)
    finally:
        tokenizer.padding_side = prev
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    pos = (mask.cumsum(-1) - 1).clamp_min(0)

    B, L = ids.shape
    assert mask.shape == (B, L), f"mask {tuple(mask.shape)} != ids {(B, L)}"
    assert pos.shape == (B, L), f"pos {tuple(pos.shape)} != ids {(B, L)}"
    # LEFT padding means the final column is a REAL token in every row. If this ever fails,
    # padding_side did not take effect and position -1 would be a PAD token.
    assert bool(mask[:, -1].all()), (
        "last column contains padding - padding_side='left' did not take effect. "
        "Position -1 would be a PAD token and every vector would be garbage.")
    # Each row's real-token count must match its mask sum, and its last real position id
    # must be (n_real - 1).
    n_real = mask.sum(-1)
    # Consistency check on the position ids we construct. NOTE this is NOT load-bearing for
    # RoPE models (measured: no effect) - it guards the non-RoPE case and catches a malformed
    # mask. The two checks that actually matter are the last-column-is-real assert above and
    # passing attention_mask at all.
    assert torch.equal(pos[:, -1], n_real - 1), (
        f"position_ids[:, -1] {pos[:, -1].tolist()} != n_real-1 {(n_real-1).tolist()}")
    if DEBUG:
        print(f"    [ar_batch] B={B} L={L}  real tokens/row={n_real.tolist()}  "
              f"pad/row={(L - n_real).tolist()}  last-col all real={bool(mask[:, -1].all())}",
              flush=True)
    return ids, mask, pos


@torch.inference_mode()
def reconstruct_batch(backbone, tokenizer, value_head, template, explanations,
                      device, batch_size=16):
    """[explanation] -> predicted activation vectors, one row each. Same maths as
    NLACritic.reconstruct, run on a left-padded batch."""
    out = []
    for s in range(0, len(explanations), batch_size):
        chunk = explanations[s:s + batch_size]
        ids, mask, pos = _encode(tokenizer, [template.format(explanation=e) for e in chunk], device)
        hs = backbone.model(ids, attention_mask=mask, position_ids=pos,
                            use_cache=False).last_hidden_state
        assert hs.shape[:2] == ids.shape, f"hidden {tuple(hs.shape)} vs ids {tuple(ids.shape)}"
        h = hs[:, -1]                                  # left-padded => the real last token
        assert h.shape == (len(chunk), hs.shape[-1]), f"h {tuple(h.shape)}"
        v = value_head(h).float().cpu()
        assert v.shape == (len(chunk), hs.shape[-1]), (
            f"value_head out {tuple(v.shape)} != (B, d_model)={(len(chunk), hs.shape[-1])}")
        assert torch.isfinite(v).all(), "value_head produced NaN/Inf"
        if DEBUG:
            print(f"    [ar_batch] hidden={tuple(hs.shape)} -> last={tuple(h.shape)} "
                  f"-> vec={tuple(v.shape)}  |v| min={v.norm(dim=-1).min():.2f} "
                  f"max={v.norm(dim=-1).max():.2f}", flush=True)
        out.append(v)
    res = torch.cat(out, 0)
    assert res.shape[0] == len(explanations), f"{res.shape[0]} vectors for {len(explanations)} inputs"
    return res


def score_batch(preds, gold, mse_scale):
    """Direction-MSE and cosine, identical to NLACritic.score but vectorised.
    Both sides L2-normalised to mse_scale, so MSE = 2(1-cos)."""
    gold = torch.as_tensor(gold, dtype=torch.float32)
    if gold.ndim == 1: gold = gold.expand(preds.shape[0], -1)
    assert preds.shape == gold.shape, f"preds {tuple(preds.shape)} != gold {tuple(gold.shape)}"
    assert torch.isfinite(preds).all() and torch.isfinite(gold).all(), "NaN/Inf in score inputs"
    p = preds / preds.norm(dim=-1, keepdim=True).clamp_min(1e-12) * mse_scale
    g = gold / gold.norm(dim=-1, keepdim=True).clamp_min(1e-12) * mse_scale
    mse = ((p - g) ** 2).mean(-1)
    cos = (p * g).sum(-1) / (p.norm(dim=-1) * g.norm(dim=-1))
    # Both sides are normalised to length mse_scale, so mse == 2(1-cos)*mse_scale^2/d exactly.
    # A violation means the normalisation did not happen - i.e. the numbers are NOT the
    # direction-only MSE the paper reports, and are not comparable to anything.
    d = preds.shape[-1]
    expect = 2 * (1 - cos) * (mse_scale ** 2) / d
    assert torch.allclose(mse, expect, atol=1e-3, rtol=1e-3), (
        f"mse != 2(1-cos)*s^2/d  max dev {(mse-expect).abs().max().item():.2e}")
    assert (cos >= -1.001).all() and (cos <= 1.001).all(), f"cos out of range: {cos.min()},{cos.max()}"
    if DEBUG:
        print(f"    [ar_batch] scored {len(mse)}  mse[{mse.min():.4f},{mse.max():.4f}]  "
              f"cos[{cos.min():.4f},{cos.max():.4f}]", flush=True)
    return mse.tolist(), cos.tolist()
