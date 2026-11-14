"""bc_analyze.py -- BIASCORR-01 analysis (offline, runs/BC/*.npz from bc_collect.py / bc_extra.py).

Gradient versions: the server produced two different step-1 gradients at each rank (PREDICT-01 Finding 0
now also at B = 0). Every Dimuon run is paired with the oracle whose Horner emulation (candidate "g")
reproduces it (median relF < 1e-6); runs sharing a gradient form a version group.
Per run: (data) tensors / float32 entries / fp16-matrix (X16) entries that differ from the same-version
beta = 0 run, when one exists; (candidates, bc_lib.CANDS x eps) predicted X16 vs data X16: mismatching
entries, exactly predicted tensors, float32 entries of U predicted exactly, and the flip test
(entries where the candidate's X16(beta) differs from its own X16(beta=0), and how many of those are also
data flips vs the same-version beta=0 data run).
Output: runs/BC/analysis.{txt,json}, runs/BC/table.md
"""
import sys, json, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bc_lib as L

D = pathlib.Path(__file__).resolve().parents[1] / "runs" / "BC"
BETAS = [0.0, 0.5, 0.75, 0.9, 0.99, 0.999, 0.3, 0.7]
f32 = np.float32


def ld(name, pre):
    z = np.load(D / f"{name}.npz")
    return {k[len(pre) + 1:]: z[k].astype(f32) for k in z.files}


def ndiff(a, b):
    return (sum(int(np.any(a[k] != b[k])) for k in a), sum(int(np.sum(a[k] != b[k])) for k in a))


def relF(G, U):
    return float(np.median([np.linalg.norm(L.predict(G[k], 0.0, "g")[1] - U[k]) / np.linalg.norm(U[k]) for k in G]))


def main(ranks=(1, 2), cands=None, epss=(1e-7, 0.0)):
    cands = cands or list(L.CANDS)
    out, lines = {}, []
    P = lambda s: (print(s, flush=True), lines.append(s))
    for r in ranks:
        oracles = sorted(p.stem for p in D.glob(f"r{r}_oracle*.npz"))
        Gs = {o: ld(o, "G") for o in oracles}
        # distinct gradient versions among oracles
        vers = []
        for o in oracles:
            for v in vers:
                if ndiff(Gs[o], Gs[v[0]])[1] == 0:
                    v.append(o); break
            else:
                vers.append([o])
        P(f"== rank {r}: oracle gradient versions: {vers}")
        if len(vers) == 2:
            a, b = Gs[vers[0][0]], Gs[vers[1][0]]
            P(f"   versions differ in {ndiff(a, b)} (tensors, entries); median per-tensor relF "
              f"{np.median([np.linalg.norm(a[k]-b[k])/np.linalg.norm(a[k]) for k in a]):.3e}")
        runs = [f"r{r}_b{b:g}" for b in BETAS] + (["r2_b0.9_rep"] if r == 2 else [])
        beta_of = {n: float(n.split("_b")[1].replace("_rep", "")) for n in runs}
        U = {n: ld(n, "U") for n in runs}
        ver_of = {}
        for n in runs:
            e = {v[0]: relF(Gs[v[0]], U[n]) for v in vers}
            best = min(e, key=e.get)
            ver_of[n] = best if e[best] < 1e-6 else None
            P(f"   {n:12s} -> gradient {ver_of[n]}  (emulation relF " + ", ".join(f"{k}: {x:.2e}" for k, x in e.items()) + ")")
        names = sorted(Gs[oracles[0]])
        tot = sum(Gs[oracles[0]][k].size for k in names)
        res = {}
        for n in runs:
            v = ver_of[n]
            if v is None:
                continue
            G, b = Gs[v], beta_of[n]
            ref0 = {k: L.predict(G[k], 0.0, "g")[0] for k in names}
            Xd = {k: L.x16_of(U[n][k], ref0[k])[0] for k in names}
            zero = [m for m in runs if ver_of[m] == v and beta_of[m] == 0.0]
            dat = dict(gradient=v, beta=b)
            if zero:
                Xz = {k: L.x16_of(U[zero[0]][k], ref0[k])[0] for k in names}
                nt, ne = ndiff(U[n], U[zero[0]])
                xt, xe = ndiff(Xd, Xz)
                dat.update(vs=zero[0], f32_tensors=nt, f32_entries=ne, x16_tensors=xt, x16_entries=xe,
                           changed=[k for k in names if np.any(U[n][k] != U[zero[0]][k])])
                P(f"   DATA {n} (grad {v}) vs {zero[0]}: f32 differs in {nt} tensors / {ne} entries; "
                  f"X16 in {xt} tensors / {xe} entries; tensors {dat['changed']}")
            else:
                Xz = None
                P(f"   DATA {n} (grad {v}): no same-version beta=0 run")
            row = {}
            for eps in epss:
                for c in cands:
                    pr = {k: L.predict(G[k], b, c, eps=eps) for k in names}
                    p0 = {k: L.predict(G[k], 0.0, c, eps=eps)[0] for k in names}
                    mis = sum(int(np.sum(pr[k][0] != Xd[k])) for k in names)
                    te = sum(int(np.all(pr[k][0] == Xd[k])) for k in names)
                    fm = sum(int(np.sum(pr[k][1] == U[n][k])) for k in names)
                    cf = {k: pr[k][0] != p0[k] for k in names}
                    ncf = sum(int(cf[k].sum()) for k in names)
                    nct = sum(int(cf[k].any()) for k in names)
                    both = sum(int((cf[k] & (Xd[k] != Xz[k])).sum()) for k in names) if Xz else None
                    row[f"{c}|eps={eps:g}"] = dict(x16_mismatch=mis, tensors_exact=te, f32_match=fm,
                                                   cand_flips=ncf, cand_flip_tensors=nct, both=both)
            res[n] = dict(data=dat, cands=row)
        out[r] = dict(versions=vers, ver_of=ver_of, total_entries=tot, n_tensors=len(names), runs=res)
        # compact table: rows candidates, columns runs; cell = X16 mismatches / exact tensors
        P(f"   -- rank {r}: X16 mismatching entries (of {tot}) / exactly-predicted tensors (of {len(names)})")
        rn = [n for n in runs if n in res]
        P("   " + "cand".ljust(24) + "".join(f"{n.replace(f'r{r}_',''):>16s}" for n in rn))
        for key in res[rn[0]]["cands"]:
            P("   " + key.ljust(24) + "".join(
                f"{res[n]['cands'][key]['x16_mismatch']:>10d}/{res[n]['cands'][key]['tensors_exact']:<5d}" for n in rn))
    (D / "analysis.txt").write_text("\n".join(lines) + "\n")
    json.dump({str(k): v for k, v in out.items()}, open(D / "analysis.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
