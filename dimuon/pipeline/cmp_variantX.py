"""cmp_variantX.py -- COMPARE-01: do public orthogonaliser descriptions reproduce the step variants X / Y?

PREDICT-01 Finding 0: from P6 state 1 (3362e501..., rank 8, T1, beta 0.9, lr 1e-3) the unclipped step 2
gave two results, X (trunk, after-state d7d226c6...) and Y (repeats, 92f25fda...). At this step lora_A
has zero stored momentum (G_A = 0 at step 1), so dA is the fresh law PE5(g2_A) -- a clean test of the
orthogonaliser alone. lora_B is preconditioned by its step-1 momentum bf16(0.1 g1_B).
Orthogonalisers: x0f16 (our rule default: X0 fp16, exact after), ours (FIRSTSTEP fused fp16 emulation),
horner (romeerp/dimuon-replay CPU path), horner_sK{k} (Gram reduction split into k fp16 partial sums).
Output: runs/COMPARE/variantX.{txt,json}. Every `stride`-th module.
"""
import sys, json, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from p3_rule import grad, state, tensors, gget, wget, precond, relF, summ, fmt, pe_apply
from fs_sim import bf16, rank_scale
from fs_ns16 import ns16
from cmp_horner import horner, PE5

OUT = pathlib.Path(__file__).resolve().parents[1] / "runs" / "COMPARE"
f32 = np.float32
S0, S1, SX, SY = "0ff0b9c991af1568", "3362e501604407c2", "d7d226c60d71ee91", "92f25fdad8ccd344"
G1, G2 = "0ff0b9c991af1568_t0_w1_P3", "3362e501604407c2_t1_w1_P3"

ORTH = {
    "x0f16": lambda M: pe_apply(M, "x0f16"),
    "ours": lambda M: ns16(M, PE5),
    "horner": lambda M: horner(M),
    "horner_sK2": lambda M: horner(M, splitk=2),
    "horner_sK8": lambda M: horner(M, splitk=8),
    "horner_sK32": lambda M: horner(M, splitk=32),
}


def predict(kind, m, g1, g2, orth, r=8, lr=1e-3, beta=0.9):
    S = None
    if kind == "B":  # step-1 momentum of B; A's is exactly 0 (G_A = 0 at B = 0)
        S = bf16(f32(1 - beta) * np.asarray(gget(g1, "B", m), f32)).astype(np.float64)
    g = np.asarray(gget(g2, kind, m), f32)
    S32 = np.zeros_like(g) if S is None else S.astype(f32)
    m32 = (f32(beta) * S32 + f32(1 - beta) * g).astype(f32)
    Dl, Dr = precond(None if S is None else S)
    if Dl is None:
        V = orth(m32.astype(np.float64))
    else:
        inner = (Dl[:, None] * m32.astype(np.float64) * Dr[None, :])
        V = Dl[:, None] * orth(inner) * Dr[None, :]
    return -rank_scale(r, lr) * V / np.linalg.norm(V)


def main(stride=2):
    g1, g2 = grad(G1), grad(G2)
    W1, WX, WY = state(S1), state(SX), state(SY)
    mods = tensors(W1)[::stride]
    res = {}
    for kind in ("A", "B"):
        acc = {f"{o}_vs_{t}": [] for o in ORTH for t in ("X", "Y")}
        acc["X_vs_Y"] = []
        for kd, m in mods:
            if kd != kind:
                continue
            dX = wget(WX, kind, m) - wget(W1, kind, m)
            dY = wget(WY, kind, m) - wget(W1, kind, m)
            acc["X_vs_Y"].append(relF(dX, dY))
            for o, f in ORTH.items():
                U = predict(kind, m, g1, g2, f)
                acc[f"{o}_vs_X"].append(relF(U, dX))
                acc[f"{o}_vs_Y"].append(relF(U, dY))
        res[kind] = {k: summ(v) for k, v in acc.items()}
    lines = []
    for kind, d in res.items():
        for k, s in d.items():
            lines.append(f"{kind} {k:18s} {fmt(s)}  (n={s.get('n')})")
    txt = "median / p90 / max relF, P6 step 2 (state 3362e501, T1), every %d-th tensor\n" % stride + "\n".join(lines)
    print(txt)
    (OUT / "variantX.txt").write_text(txt + "\n")
    json.dump(res, open(OUT / "variantX.json", "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
