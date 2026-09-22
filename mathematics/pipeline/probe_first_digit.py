#!/usr/bin/env python3
"""PROBE-01 v2. Is the FIRST digit readable from the position of the SECOND digit?

v1 was thrown away: its 'last digit' sanity control sat at 0.0% at every depth because the split
held out last digits 8,9 — so those classes never appeared in training and the probe could not
emit them. A control that cannot pass is not a control. Fixed below, plus error bars.

DESIGN
  target 'first'  : predict the first digit (9 classes, chance 11.1%) from the 5120 values at the
                    SECOND digit's position. Split holds out two LAST digits, so every test number
                    is unseen while all 9 first-digit classes are present in training.
  control 'last'  : predict the LAST digit — the token actually being read. Split holds out two
                    FIRST digits, so all 10 last-digit classes are present in training. Must be
                    high at depth 0 or the pipeline is broken.
  control 'shuf'  : first-digit labels shuffled within the training set. Must sit at chance.
  Repeated over several random held-out pairs -> mean and spread instead of one number.
"""
import argparse, json, numpy as np

def fit_probe(Xtr, ytr, Xte, yte, classes, C=10.0, steps=400, lr=0.5):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    K, D, n = len(classes), A.shape[1], len(Xtr)
    idx = {c: i for i, c in enumerate(classes)}
    Y = np.zeros((n, K)); Y[np.arange(n), [idx[v] for v in ytr]] = 1
    W = np.zeros((D, K)); b = np.zeros(K)
    for _ in range(steps):
        Z = A @ W + b; Z -= Z.max(1, keepdims=True); P = np.exp(Z); P /= P.sum(1, keepdims=True)
        G = (P - Y) / n
        W -= lr * (A.T @ G + C / n * W); b -= lr * G.sum(0)
    pred = [classes[i] for i in (B @ W + b).argmax(1)]
    return float(np.mean([p == t for p, t in zip(pred, yte)]))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="mathematics/runs/2026-09-20_numbers/numbers/acts_bare.npy")
    ap.add_argument("--meta", default="mathematics/runs/2026-09-20_numbers/numbers/meta.json")
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--out", default="mathematics/runs/2026-09-20_numbers/probe_first_digit.json")
    a = ap.parse_args()

    nums = np.array(json.load(open(a.meta))["numbers"])
    A = np.load(a.acts).astype(np.float32)
    first, last = nums // 10, nums % 10
    fc, lc = sorted(set(first)), sorted(set(last))
    rng = np.random.default_rng(0)
    # splits, fixed across depths so the curves are comparable
    sp_first = [tuple(rng.choice(lc, 2, replace=False)) for _ in range(a.repeats)]   # hold out LAST digits
    sp_last  = [tuple(rng.choice(fc, 2, replace=False)) for _ in range(a.repeats)]   # hold out FIRST digits
    print(f"first-digit probe: {a.repeats} splits, each holding out 2 last digits  (chance {1/len(fc):.1%})")
    print(f"last-digit control: {a.repeats} splits, each holding out 2 first digits (chance {1/len(lc):.1%})\n")

    res = {"depths": [], "first": [], "first_sd": [], "shuf": [], "last": []}
    for L in range(A.shape[1]):
        X = A[:, L, :]
        f, s = [], []
        for held in sp_first:
            tr = np.array([i for i, n in enumerate(nums) if n % 10 not in held])
            te = np.array([i for i, n in enumerate(nums) if n % 10 in held])
            f.append(fit_probe(X[tr], first[tr], X[te], first[te], fc))
            sh = first[tr].copy(); np.random.default_rng(1).shuffle(sh)
            s.append(fit_probe(X[tr], sh, X[te], first[te], fc))
        ld = []
        for held in sp_last:
            tr = np.array([i for i, n in enumerate(nums) if n // 10 not in held])
            te = np.array([i for i, n in enumerate(nums) if n // 10 in held])
            ld.append(fit_probe(X[tr], last[tr], X[te], last[te], lc))
        res["depths"].append(L); res["first"].append(float(np.mean(f))); res["first_sd"].append(float(np.std(f)))
        res["shuf"].append(float(np.mean(s))); res["last"].append(float(np.mean(ld)))
    json.dump(res, open(a.out, "w"), indent=1)

    def kind(L): return "emb" if L == 0 else ("FULL" if (L-1) % 4 == 3 else "lin")
    print(f"{'depth':>6} {'kind':>5} {'FIRST digit':>18} {'shuffled':>10} {'LAST digit ctrl':>16}")
    for L in res["depths"]:
        if L <= 8 or L % 4 == 0 or L == 64:
            print(f"{L:>6} {kind(L):>5} {res['first'][L]:>11.1%} ±{res['first_sd'][L]:>5.1%} {res['shuf'][L]:>10.1%} {res['last'][L]:>16.1%}")
