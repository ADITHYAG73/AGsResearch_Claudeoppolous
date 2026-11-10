"""cmp_anpaure.py -- COMPARE-01: run the public reimplementations on our P6 chain (rank 8, 6 steps).

(1) anpaure/dimuon `dimuon.py` (commit d6570bb, 2026-10-28) imported unchanged from the clone given as
    argv[1]; its Dimuon optimizer is fed our oracle gradients step by step (its own bf16 momentum
    state), starting each step from the MEASURED pre-step weights; predicted dW = after - before.
    Run with precision "fp16" (its default) and "fp64".
(2) our rule (p3_rule.rule) with the PE replaced by romeerp's Horner fp16 emulation (cmp_horner.horner),
    momentum rolled from oracles exactly as in PREDICT-01 P6.
(3) our rule as frozen (x0f16), for reference.
Residual = relF(pred, measured dW) per tensor, median / p90 / max, A and B. Every 2nd module.
Output runs/COMPARE/anpaure_p6.{txt,json}.
"""
import sys, json, pathlib, importlib.util, numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from p3_rule import grad, state, tensors, gget, wget, rule, relF, summ, fmt, man
from cmp_horner import horner

OUT = pathlib.Path(__file__).resolve().parents[1] / "runs" / "COMPARE"
torch.set_num_threads(4)


def load_anpaure(path):
    spec = importlib.util.spec_from_file_location("anpaure_dimuon", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(anp_path, nsteps=6, stride=2):
    anp = load_anpaure(anp_path)
    M = man()["steps"]
    steps = [M[f"P6_r8/step{i}"] for i in range(1, nsteps + 1)]
    mods = tensors(state(steps[0]["before"]))[::stride]
    # anpaure optimisers, one per precision; params in STORED orientation (A: r x d_in, B: d_out x r)
    names = [f"{k}/{m}.lora_{k}" for k, m in mods]
    params = {p: {n: torch.nn.Parameter(torch.zeros(1)) for n in names} for p in ("fp16", "fp64")}
    opts = {p: anp.Dimuon(list(params[p].values()), lr=1e-3, beta1=0.9, precision=p) for p in params}
    momH = {n: None for n in names}
    momO = {n: None for n in names}
    out = {}
    for i, s in enumerate(steps, 1):
        W0, W1, G = state(s["before"]), state(s["after"]), grad(s["grad_key"])
        acc = {f"{c}_{k}": [] for c in ("anp_fp16", "anp_fp64", "horner", "ours") for k in "AB"}
        for p in params:
            for n in names:
                k = n[0]
                prm = params[p][n]
                prm.data = torch.from_numpy(np.asarray(W0.z[n], np.float32).copy())
                prm.grad = torch.from_numpy(np.asarray(G.z[f"G{n}"], np.float32).copy())
            opts[p].step()
        for (k, m), n in zip(mods, names):
            d = np.asarray(W1.z[n], np.float64) - np.asarray(W0.z[n], np.float64)
            for p in params:
                dp = params[p][n].data.double().numpy() - np.asarray(W0.z[n], np.float64)
                acc[f"anp_{p}_{k}"].append(relF(dp, d))
            dt = wget(W1, k, m) - wget(W0, k, m)
            g = gget(G, k, m)
            # horner-PE variant of our rule (re-implemented inline; p3_rule.pe_apply has no horner mode)
            Uh, momH[n] = rule_h(momH[n], g)
            Uo, momO[n] = rule(momO[n], g, 0.9, 1e-3, 8, mode="x0f16")
            acc[f"horner_{k}"].append(relF(Uh, dt))
            acc[f"ours_{k}"].append(relF(Uo, dt))
        out[i] = {k: summ(v) for k, v in acc.items()}
        line = f"step{i}: " + " | ".join(f"{c} A {fmt(out[i][c + '_A'])} B {fmt(out[i][c + '_B'])}"
                                         for c in ("ours", "horner", "anp_fp16", "anp_fp64"))
        print(line, flush=True)
    (OUT / "anpaure_p6.txt").write_text("\n".join(
        f"step{i} {c:9s} A {fmt(v[c + '_A'])}   B {fmt(v[c + '_B'])}" for i, v in out.items()
        for c in ("ours", "horner", "anp_fp16", "anp_fp64")) + "\n")
    json.dump({str(k): v for k, v in out.items()}, open(OUT / "anpaure_p6.json", "w"), indent=1)


def rule_h(S, g, beta=0.9, lr=1e-3, r=8):
    """p3_rule.rule with the orthogonaliser replaced by the Horner fp16 emulation."""
    import p3_rule
    from fs_sim import bf16, rank_scale
    f32 = np.float32
    g32 = np.asarray(g, f32)
    S32 = np.zeros_like(g32) if S is None else np.asarray(S, f32)
    m32 = (f32(beta) * S32 + f32(1 - beta) * g32).astype(f32)
    Dl, Dr = p3_rule.precond(None if S is None else S32.astype(np.float64))
    M = m32.astype(np.float64)
    if not np.any(M):
        V = np.zeros_like(M)
    elif Dl is None:
        V = horner(M)
    else:
        V = Dl[:, None] * horner(Dl[:, None] * M * Dr[None, :]) * Dr[None, :]
    n = np.linalg.norm(V)
    U = -rank_scale(r, lr) * V / n if n > 0 else V
    return U, bf16(m32).astype(f32)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6)
