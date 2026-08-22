"""CPU test for the batched AR path. No GPU, no checkpoint download.

Verifies the ONE property that matters and that cannot be eyeballed:

    reconstruct_batch(texts)[i]  ==  single-sequence reconstruct(texts[i])

for texts of DIFFERENT lengths. A left/right padding mistake, a missing attention_mask, or
default position_ids all break this equality while still returning plausible numbers.

Stand-in architecture: a tiny randomly-initialised Llama. The local transformers (4.49) has
no Gemma-3; the pod uses 5.3.0. The padding/masking/position-id semantics being tested are
architecture-independent - they are about how a padded batch is fed to any decoder-only HF
model. The Gemma-3-specific parts (BOS handling, the real checkpoint, value_head weights)
are covered by the 20-item batched-vs-serial check on the pod, not here.
"""
import sys, os, math
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pod"))
from ar_batch import reconstruct_batch, score_batch


def build_tiny():
    from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=32000, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4)
    model = LlamaForCausalLM(cfg).eval()
    tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    head = torch.nn.Linear(cfg.hidden_size, cfg.hidden_size).eval()
    return model, tok, head, cfg.hidden_size


@torch.inference_mode()
def single(backbone, tok, head, template, text):
    """Byte-for-byte the official single-item path (nla_inference.py:646)."""
    ids = tok(template.format(explanation=text), return_tensors="pt",
              add_special_tokens=True)["input_ids"]
    h = backbone.model(ids, use_cache=False).last_hidden_state[0, -1]
    return head(h).float()


def main():
    model, tok, head, d = build_tiny()
    template = "Explanation:\n{explanation}\nVector:"
    texts = [                                   # deliberately very different lengths
        "Short one.",
        "A considerably longer explanation that will tokenize to many more tokens than the "
        "first one, which is exactly the condition under which padding bugs appear.",
        "Medium length explanation about cricket.",
        "x",
    ]
    ref = torch.stack([single(model, tok, head, template, t) for t in texts])

    ok = True
    for bs in (1, 2, 4, 16):
        got = reconstruct_batch(model, tok, head, template, texts, device="cpu", batch_size=bs)
        md = (got - ref).abs().max().item()
        good = torch.allclose(got, ref, atol=1e-4, rtol=1e-4)
        ok &= good
        print(f"  batch_size={bs:2d}  max|batched - single| = {md:.3e}   {'PASS' if good else 'FAIL'}")

    # Negative control: the test must be capable of FAILING. Right-padding should break it.
    prev = tok.padding_side; tok.padding_side = "right"
    enc = tok([template.format(explanation=t) for t in texts], return_tensors="pt", padding=True,
              add_special_tokens=True)
    with torch.inference_mode():
        h_bad = model.model(enc["input_ids"], attention_mask=enc["attention_mask"],
                            use_cache=False).last_hidden_state[:, -1]
        bad = head(h_bad).float()
    tok.padding_side = prev
    bad_differs = not torch.allclose(bad, ref, atol=1e-4, rtol=1e-4)
    print(f"  negative control (RIGHT padding, no position_ids): "
          f"max|wrong - single| = {(bad-ref).abs().max().item():.3e}   "
          f"{'PASS (it does break)' if bad_differs else 'FAIL (test is blind)'}")
    ok &= bad_differs

    # Identity: both sides are normalised to length s = mse_scale, so
    #   mse = (1/d)||p-g||^2 = (1/d)(2s^2 - 2 p.g) = 2(1-cos) * s^2/d
    # The official docstring's flat "MSE = 2(1-cos)" therefore holds ONLY when s^2/d == 1,
    # i.e. mse_scale == sqrt(d_model) - which is exactly what the shipped sidecars use
    # (Gemma-3-12B L32: d=3840, mse_scale=61.97=sqrt(3840)). Checked at both scales.
    # 5.0 != sqrt(64)=8.0, so the FLAT identity must FAIL there and HOLD at sqrt(d).
    for s_scale, flat_expected in ((5.0, False), (math.sqrt(d), True)):
        m, c = score_batch(ref, torch.randn(d), mse_scale=s_scale)
        exp = [2 * (1 - ci) * s_scale ** 2 / d for ci in c]
        good = all(abs(mi - ei) < 1e-3 for mi, ei in zip(m, exp))
        flat = all(abs(mi - 2 * (1 - ci)) < 1e-3 for mi, ci in zip(m, c))
        disc = (flat == flat_expected)
        print(f"  score_batch  mse_scale={s_scale:6.3f}  mse == 2(1-cos)*s^2/d ... "
              f"{'PASS' if good else 'FAIL'}   flat 2(1-cos) holds: {flat} "
              f"(expected {flat_expected}) {'ok' if disc else 'TEST NOT DISCRIMINATING'}")
        if not (good and disc): return 1

    print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
