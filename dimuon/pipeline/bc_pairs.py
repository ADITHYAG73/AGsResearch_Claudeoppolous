"""bc_pairs.py -- BIASCORR-01: all within-version beta PAIRS as flip events.

Within one gradient version, for every pair of betas (bi, bj) the data says which tensors differ
(float32 U). A candidate (momentum form x normaliser variant, bc_flips.x0v) predicts which tensors' X16
differ between its own predictions at bi and bj. Score: TP / FP / FN tensors summed over all pairs, and
exact X16 agreement in the TP tensors that the beta=0 emulation reproduces exactly (clean tensors).
Uses only the normaliser variants whose beta=0 floor is best in flips.txt (pair / f64 norm).
Output runs/BC/pairs.{txt,json}
"""
import sys, json, pathlib, itertools, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bc_lib as L, bc_analyze as A
from bc_flips import pred

J = json.load(open(A.D / "analysis.json"))


def main():
    cands = [c for c in L.CANDS if not c.endswith(("bf16", "fp16"))]
    variants = [(n, d, e) for n in ("pair", "f64") for d in ("mul", "recip", "div2")
                for e in ((0.0, 1e-7) if d != "div2" else (0.0,))]
    groups = []
    data_lines = []
    for r in ("1", "2"):
        R = J[r]
        for v in R["versions"]:
            runs = [n for n in R["runs"] if R["runs"][n]["data"]["gradient"] == v[0] and not n.endswith("rep")]
            G = A.ld(v[0], "G")
            U = {n: A.ld(n, "U") for n in runs}
            beta = {n: R["runs"][n]["data"]["beta"] for n in runs}
            names = sorted(G)
            XD = {n: {k: L.x16_of(U[n][k], L.predict(G[k], 0.0, "g")[0])[0] for k in names} for n in runs}
            pairs = []
            for a, b in itertools.combinations(runs, 2):
                ch = {k for k in names if np.any(U[a][k] != U[b][k])}
                pairs.append((a, b, ch))
                data_lines.append(f"r{r} grad {v[0]}: {a} vs {b}: {len(ch)} tensors differ {sorted(ch)}")
            groups.append((r, G, names, runs, beta, XD, pairs))
    lines = list(data_lines)
    P = lambda s: (print(s, flush=True), lines.append(s))
    n_events = sum(len(p[2]) for g in groups for p in g[6])
    P(f"total data events (tensor x pair): {n_events}")
    out = {}
    for norm, den, eps in variants:
        for c in cands:
            tp = fp = fn = tpx = tpclean = 0
            for r, G, names, runs, beta, XD, pairs in groups:
                pr = {n: {k: pred(G[k], beta[n], c, norm, den, eps) for k in names} for n in runs}
                for a, b, ch in pairs:
                    cc = {k for k in names if np.any(pr[a][k] != pr[b][k])}
                    tp += len(ch & cc); fp += len(cc - ch); fn += len(ch - cc)
                    for k in ch & cc:
                        if np.all(pr[a][k] == XD[a][k]):        # clean at a
                            tpclean += 1
                            tpx += int(np.all(pr[b][k] == XD[b][k]))
            key = f"{c}|{norm}|{den}|eps={eps:g}"
            out[key] = dict(TP=tp, FP=fp, FN=fn, TP_clean=tpclean, TP_clean_exact=tpx)
            P(f"{key:40s} TP {tp:3d} FP {fp:3d} FN {fn:3d}  TP clean {tpclean} of which exact {tpx}")
    (A.D / "pairs.txt").write_text("\n".join(lines) + "\n")
    json.dump(out, open(A.D / "pairs.json", "w"), indent=1)


if __name__ == "__main__":
    main()
