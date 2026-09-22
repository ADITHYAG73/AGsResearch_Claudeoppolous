#!/usr/bin/env python3
"""Is the ANSWER already in the residual stream before the model has said anything?

TRIG-01 cached the residual stream at all 65 depths at the LAST PROMPT TOKEN - the state after
reading "Compute sin(37 degrees)... 5 decimal places." and before emitting a single character.

If the tenths digit of the answer can be read out of that state by a linear probe, then the
value was computed (or retrieved) inside the forward pass, not assembled token by token as it
speaks. If it only becomes readable in the last few blocks, that locates WHERE. If it is never
readable, the digits are being produced during generation instead.

TARGETS, and the difference matters:
  true   the tenths digit of the CORRECT answer   - "did it work the value out"
  said   the tenths digit of the answer it GAVE   - "is its own output already determined"
  shuf   labels shuffled                          - the control that must sit at chance

`said` should be readable if anything is. The interesting comparison is `true` vs `said` on the
rows where the model was WRONG: if `true` is readable there while `said` is what it emits, the
model had the right value internally and failed to say it. Claude's note: that is a strong claim
and this probe alone would not establish it - it would need the wrong-only subset to be large
enough to train on, which is checked and reported below rather than assumed.

Uses its own small multinomial logistic regression (not a paper's probe), so numbers here are
NOT comparable to published probe accuracies.
"""
import json, os, sys
import numpy as np

def softmax_fit(X, y, n_cls, epochs=220, lr=0.4, l2=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    X = np.c_[X, np.ones(len(X))]
    W = rng.normal(0, 0.01, (X.shape[1], n_cls))
    Y = np.zeros((len(y), n_cls)); Y[np.arange(len(y)), y] = 1
    for _ in range(epochs):
        Z = X @ W; Z -= Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        W -= lr * (X.T @ (P - Y) / len(X) + l2 * W)
    return W

def acc(W, X, y):
    X = np.c_[X, np.ones(len(X))]
    return float((np.argmax(X @ W, 1) == y).mean())

def tenths(v):
    """tenths digit of |v|, i.e. the first digit after the point. None if not usable."""
    if v is None or not np.isfinite(v): return None
    return int(abs(v) * 10) % 10

def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    A = np.load(os.path.join(d, "acts_off_step1_0.npy"))            # [N, 65, 5120] float16
    idx = json.load(open(os.path.join(d, "acts_index_off_step1_0.json")))
    rows = [json.loads(l) for l in open(os.path.join(d, "results_off_step1.jsonl"))]
    by = {(r["fn"], r["deg"]): r for r in rows}
    print(f"activations {A.shape}, index {len(idx)}, results {len(rows)}")
    assert len(idx) == A.shape[0], "index and activation rows must align"

    keep, y_true, y_said = [], [], []
    for i, e in enumerate(idx):
        r = by.get((e["fn"], e["deg"]))
        if r is None or r["pole"] or r["truth"] is None or r["value"] is None: continue
        t, s = tenths(r["truth"]), tenths(r["value"])
        if t is None or s is None: continue
        keep.append(i); y_true.append(t); y_said.append(s)
    keep = np.array(keep); y_true = np.array(y_true); y_said = np.array(y_said)
    print(f"usable rows {len(keep)}   tenths-digit class balance (true): "
          f"{np.bincount(y_true, minlength=10).tolist()}")

    rng = np.random.default_rng(0)
    y_shuf = rng.permutation(y_true)
    depths = list(range(0, A.shape[1], 4)) + [A.shape[1]-1]
    depths = sorted(set(depths))
    n_splits = 5
    print(f"\n{n_splits}-fold random splits, chance = 10.0%")
    print(f"{'depth':>6}  {'TRUE digit':>14}  {'SAID digit':>14}  {'shuffled':>10}")
    out = {}
    for L in depths:
        X = A[keep, L, :].astype(np.float32)
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        accs = {"true": [], "said": [], "shuf": []}
        for s in range(n_splits):
            r2 = np.random.default_rng(100 + s)
            perm = r2.permutation(len(X)); cut = int(0.75 * len(X))
            tr, te = perm[:cut], perm[cut:]
            for name, yy in (("true", y_true), ("said", y_said), ("shuf", y_shuf)):
                W = softmax_fit(X[tr], yy[tr], 10, seed=s)
                accs[name].append(acc(W, X[te], yy[te]))
        m = {k: 100*np.mean(v) for k, v in accs.items()}
        sd = {k: 100*np.std(v) for k, v in accs.items()}
        out[L] = m
        print(f"{L:>6}  {m['true']:>8.1f}% ±{sd['true']:<4.1f}  {m['said']:>8.1f}% ±{sd['said']:<4.1f}  {m['shuf']:>9.1f}%")
    best = max(out, key=lambda L: out[L]["said"])
    print(f"\nbest depth for the SAID digit: {best} ({out[best]['said']:.1f}%)")
    json.dump({str(k): v for k, v in out.items()}, open(os.path.join(d, "probe_tenths.json"), "w"), indent=1)
    print(f"-> {os.path.join(d, 'probe_tenths.json')}")

if __name__ == "__main__":
    main()
