"""bc_flips.py -- BIASCORR-01: which (momentum form x normaliser arithmetic) reproduces the rare
beta-induced changes bit for bit?

Data facts used (runs/BC, see bc_analyze): within one gradient version, U(beta) differs from U(0) in at
most one tensor. A candidate is scored on every run:
  * same-version beta=0 pairs: tensors the candidate predicts to change (its X16(beta) != its X16(0))
    vs tensors that changed in the data; TP / FP / FN, and exact entry agreement in TP tensors;
  * all runs: tensors whose X16 is predicted exactly, and the beta=0 floor of the same normaliser
    (so "exact tensors" can be read against what the emulation can reach at all).
Normaliser variants for X0 = fp16(M / den):
  norm: pair = numpy float32 pairwise sum; f64 = float64 sum then sqrt, rounded to f32;
        seq = strictly sequential float32 accumulation; dot = BLAS float32 dot.
  den:  mul = n*1.01 (+eps); div2 = (M/n)/1.01 ; recip = M*(1/(n*1.01))
Output runs/BC/flips.{txt,json}
"""
import sys, json, pathlib, itertools, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bc_lib as L
import bc_analyze as A

f32 = np.float32
D = A.D


def nrm(M, kind):
    v = M.ravel()
    if kind == "pair":
        return np.sqrt(np.sum(v * v, dtype=f32))
    if kind == "f64":
        return f32(np.sqrt(np.sum(v.astype(np.float64) ** 2)))
    if kind == "seq":
        return np.sqrt(np.cumsum(v * v, dtype=f32)[-1])
    if kind == "dot":
        return np.sqrt(np.dot(v, v))
    raise ValueError(kind)


def x0v(M, norm, den, eps):
    M = np.asarray(M, f32)
    n = f32(nrm(M, norm))
    if den == "mul":
        Y = M / (n * f32(1.01) + f32(eps))
    elif den == "div2":
        Y = (M / n).astype(f32) / f32(1.01)
    elif den == "recip":
        Y = M * (f32(1.0) / (n * f32(1.01) + f32(eps)))
    return L.H(Y)


def pred(g, b, cand, norm, den, eps):
    return L.horner_from_x0(x0v(L.CANDS[cand](g, b), norm, den, eps))


def main():
    J = json.load(open(D / "analysis.json"))
    cands = [c for c in L.CANDS if not c.endswith(("bf16", "fp16"))]
    variants = [(n, d, e) for n in ("pair", "f64", "seq", "dot") for d in ("mul", "div2", "recip")
                for e in ((0.0, 1e-7) if d != "div2" else (0.0,))]
    lines, out = [], {}
    P = lambda s: (print(s, flush=True), lines.append(s))
    setup = []
    for r in ("1", "2"):
        R = J[r]
        Gs = {o[0]: A.ld(o[0], "G") for o in R["versions"]}
        runs = list(R["runs"])
        U = {n: A.ld(n, "U") for n in runs}
        beta = {n: R["runs"][n]["data"]["beta"] for n in runs}
        ver = {n: R["runs"][n]["data"]["gradient"] for n in runs}
        zero = {n: R["runs"][n]["data"].get("vs") for n in runs}
        XD = {}
        for n in runs:
            G = Gs[ver[n]]
            XD[n] = {k: L.x16_of(U[n][k], L.predict(G[k], 0.0, "g")[0])[0] for k in G}
        setup.append((r, Gs, runs, U, beta, ver, zero, XD))
    for norm, den, eps in variants:
        for c in cands:
            key = f"{c}|{norm}|{den}|eps={eps:g}"
            tp = fp = fn = tpx = 0
            exact = floor = tot_t = 0
            per = {}
            for r, Gs, runs, U, beta, ver, zero, XD in setup:
                cache0 = {}
                for n in runs:
                    G = Gs[ver[n]]
                    names = sorted(G)
                    if ver[n] not in cache0:
                        cache0[ver[n]] = {k: pred(G[k], 0.0, c, norm, den, eps) for k in names}
                    p0 = cache0[ver[n]]
                    pb = {k: pred(G[k], beta[n], c, norm, den, eps) for k in names}
                    Xd = XD[n]
                    ex = sum(int(np.all(pb[k] == Xd[k])) for k in names)
                    fl = sum(int(np.all(p0[k] == Xd[k])) for k in names)
                    exact += ex; floor += fl; tot_t += len(names)
                    rec = dict(exact=ex, floor_b0pred=fl)
                    if zero[n] and n != zero[n]:
                        Z = U[zero[n]]
                        dch = {k for k in names if np.any(U[n][k] != Z[k])}
                        cch = {k for k in names if np.any(pb[k] != p0[k])}
                        tp += len(dch & cch); fp += len(cch - dch); fn += len(dch - cch)
                        tpx += sum(int(np.all(pb[k] == Xd[k])) for k in dch & cch)
                        rec.update(data_changed=sorted(dch), cand_changed=sorted(cch))
                    per[n] = rec
            out[key] = dict(TP=tp, FP=fp, FN=fn, TP_exact=tpx, exact_tensors=exact, b0pred_floor=floor,
                            tensor_runs=tot_t, per=per)
            P(f"{key:40s} TP {tp} FP {fp:3d} FN {fn} TPexact {tpx} | exact tensors {exact}/{tot_t} "
              f"(beta0-prediction floor {floor})")
    (D / "flips.txt").write_text("\n".join(lines) + "\n")
    json.dump(out, open(D / "flips.json", "w"), indent=1)


if __name__ == "__main__":
    main()
