"""cmp_damping.py -- COMPARE-01: how large may the COLUMN-side damping be? (public sources assert 1e-4 relative
to the max on both sides; MULTISTEP-01 could not identify the column side.)

Rank-32 chain P2s/P4s (solo oracles, setting P3x1): step1 5ca507c2 (T0), step2 6b6b57f8 (T0),
step3 aab50395 (T1, lr 1e-3, solo, after-state b0a2258f). Momentum rolled with our rule (bf16 storage);
step 3 is predicted with row damping 1e-4*max and column damping eps_c*max for a grid of eps_c, and also
with the column damping relative to the MEAN. Also reports the spread of the column energies at step 3.
Output runs/COMPARE/damping.{txt,json}.
"""
import sys, json, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import p3_rule
from p3_rule import grad, state, tensors, gget, wget, rule, relF, summ, fmt, pe_apply
from fs_sim import rank_scale

OUT = pathlib.Path(__file__).resolve().parents[1] / "runs" / "COMPARE"
STEPS = ["5ca507c22b88b77d_t0_w1_P3x1", "6b6b57f85d3dbb33_t0_w1_P3x1", "aab5039568e48d84_t1_w1_P3x1"]
BEFORE, AFTER = "aab5039568e48d84", "b0a2258f"


def find_state(prefix):
    for p in (p3_rule.P3 / "states").glob(prefix + "*.npz"):
        return p.stem
    raise FileNotFoundError(prefix)


def step3(S, g, eps_c, ref="max", eps_r=1e-4, r=32, lr=1e-3, beta=0.9):
    f32 = np.float32
    m32 = (f32(beta) * np.asarray(S, f32) + f32(1 - beta) * np.asarray(g, f32)).astype(np.float64)
    S2 = np.asarray(S, np.float64) ** 2
    l, c = S2.sum(1), S2.sum(0)
    l = l + eps_r * l.max()
    c = c + eps_c * (c.max() if ref == "max" else c.mean())
    Dl, Dr = l ** -0.25, c ** -0.25
    V = Dl[:, None] * pe_apply(Dl[:, None] * m32 * Dr[None, :], "x0f16") * Dr[None, :]
    return -rank_scale(r, lr) * V / np.linalg.norm(V)


def main(stride=4):
    G = [grad(k) for k in STEPS]
    W0, W1 = state(BEFORE), state(find_state(AFTER))
    mods = tensors(W0)[::stride]
    grid = [("max", e) for e in (0.0, 3e-5, 6e-5, 1e-4, 1.5e-4, 3e-4, 1e-3, 1e-2, 3e-2, 1e-1)] + [("mean", e) for e in (1e-4, 1e-2)]
    res = {}
    for kind in ("A", "B"):
        acc = {f"{ref}{e:g}": [] for ref, e in grid}
        spread = []
        for kd, m in mods:
            if kd != kind:
                continue
            S = None
            for k in range(2):
                _, S = rule(S, gget(G[k], kind, m), 0.9, 1e-3, 32)
            d = wget(W1, kind, m) - wget(W0, kind, m)
            S64 = np.asarray(S, np.float64)
            if not np.any(S64):
                continue
            c = (S64 ** 2).sum(0)
            spread.append(c.min() / c.max())
            for ref, e in grid:
                acc[f"{ref}{e:g}"].append(relF(step3(S64, gget(G[2], kind, m), e, ref), d))
        res[kind] = {k: summ(v) for k, v in acc.items()}
        res[kind]["colE_min_over_max"] = summ(spread)
    lines = [f"rank 32, step 3 from aab50395 (solo), row damping 1e-4*max; column damping varied; every {stride}-th module"]
    for kind, d in res.items():
        for k, s in d.items():
            lines.append(f"{kind} {k:20s} {fmt(s)}  n={s.get('n')}")
    print("\n".join(lines))
    (OUT / "damping.txt").write_text("\n".join(lines) + "\n")
    json.dump(res, open(OUT / "damping.json", "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
