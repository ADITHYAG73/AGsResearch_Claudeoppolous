#!/usr/bin/env python3
"""FT-01 numbers: base (re-run on A100) vs fine-tuned checkpoints, same questions, same harness.

Every comparison is PAIRED on the same 3,362 questions (McNemar for right/wrong flips).
The A40-vs-A100 base comparison is the hardware noise floor: how many answers flip with no change
to the model at all. A fine-tuning effect is only meaningful if it is clearly larger than that.

"Does the sandbox shield maths?" is answered per question: d_i = (change alone) - (change with
sandbox), each change in {-1, 0, +1}; mean of d over questions with a paired 95% CI.
Usage: python ft01_numbers.py <ft01_results_dir> <base_a40_dir> <out.json>
"""
import json, math, os, sys, glob, collections

CONDS = ["notool_off", "notool_on", "tool_off", "tool_on"]
# 12 questions whose expressions or answers match textbook constants found in the training rows (DATA-01)
CONTAM = set("q00091 q00135 q00136 q00182 q02415 q02438 q02158 q02226 q02426 q02446 q03070 q03211".split())
FAMS = ["trig_int", "trig_var", "transcendental", "composite", "nested", "mixed", "undefined"]

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))

def load(d, model, cond):
    p = os.path.join(d, f"scored_{model}_{cond}.jsonl")
    if not os.path.exists(p): return None
    rows = {r["id"]: r for r in map(json.loads, open(p))}
    if cond == "tool_on":   # merge the 12-turn re-ask, exactly as BASE-01 reported it
        p12 = os.path.join(d, f"scored_{model}_tool_on_turns12.jsonl")
        if os.path.exists(p12):
            for r in map(json.loads, open(p12)): rows[r["id"]] = dict(r, cond="tool_on")
    return rows

def ok(r): return r["status"] == "correct"

def summary(rows):
    n = len(rows); k = sum(ok(r) for r in rows.values())
    return {"n": n, "correct": k, "pct": round(100 * k / n, 1), "ci95": wilson(k, n),
            "within_1e-3_pct": round(100 * sum(bool(r.get("close_1e-3")) for r in rows.values()) / n, 1),
            "no_answer": sum(r["status"] not in ("correct", "wrong") for r in rows.values()),
            "mean_tokens": round(sum(r.get("gen_tokens", 0) for r in rows.values()) / n)}

def mcnemar(a, b):
    ids = sorted(set(a) & set(b))
    gain = sum((not ok(a[i])) and ok(b[i]) for i in ids)
    loss = sum(ok(a[i]) and (not ok(b[i])) for i in ids)
    z = (gain - loss) / math.sqrt(gain + loss) if gain + loss else 0.0
    return {"n_paired": len(ids), "gained": gain, "lost": loss, "net": gain - loss, "z": round(z, 2)}

def shield(base_alone, ft_alone, base_sbx, ft_sbx):
    ids = sorted(set(base_alone) & set(ft_alone) & set(base_sbx) & set(ft_sbx))
    d = [(ok(ft_alone[i]) - ok(base_alone[i])) - (ok(ft_sbx[i]) - ok(base_sbx[i])) for i in ids]
    n = len(d); m = sum(d) / n; sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    ca = sum(ok(ft_alone[i]) - ok(base_alone[i]) for i in ids) / n
    cs = sum(ok(ft_sbx[i]) - ok(base_sbx[i]) for i in ids) / n
    return {"n": n, "change_alone_pts": round(100 * ca, 2), "change_sandbox_pts": round(100 * cs, 2),
            "alone_minus_sandbox_pts": round(100 * m, 2),
            "ci95_pts": (round(100 * (m - 1.96 * sd / math.sqrt(n)), 2), round(100 * (m + 1.96 * sd / math.sqrt(n)), 2))}

def ifeval_rows(d, model, cond):
    """Per-prompt strict pass plus whether the response exists (thinking closed inside the budget)."""
    sp = os.path.join(d, f"ifeval_scored_{model}_{cond}_strict.jsonl"); rp = os.path.join(d, f"ifeval_raw_{model}_{cond}.jsonl")
    if not (os.path.exists(sp) and os.path.exists(rp)): return None
    closed = {r["key"]: (cond == "off" or "</think>" in r["raw"]) for r in map(json.loads, open(rp))}
    return {r["key"]: {"status": "correct" if r["follow_all"] else "wrong", "closed": closed[r["key"]]}
            for r in map(json.loads, open(sp))}

def main():
    ftdir, a40dir, outp = sys.argv[1], sys.argv[2], sys.argv[3]
    models = sorted({os.path.basename(p).split("_")[1] for p in glob.glob(f"{ftdir}/scored_*.jsonl")})
    base = "Qwen3.5-4B-rerun"
    assert base in models, f"no A100 base re-run in {ftdir}: {models}"
    ckpts = [m for m in models if m != base]
    R = {m: {c: load(ftdir, m, c) for c in CONDS} for m in models}
    R["Qwen3.5-4B-A40"] = {c: load(a40dir, "Qwen3.5-4B", c) for c in CONDS}
    out = {"summary": {}, "paired_vs_base_a100": {}, "hardware_noise_a40_vs_a100": {},
           "shield": {}, "family_pct": {}, "tool_labels": {}, "ifeval": {},
           "summary_excl_contam": {}, "paired_vs_base_a100_excl_contam": {}, "ifeval_paired": {}}
    for m, cs in R.items():
        out["summary"][m] = {c: summary(r) for c, r in cs.items() if r}
        out["family_pct"][m] = {c: {f: round(100 * sum(ok(x) for x in r.values() if x["family"] == f) /
                                         max(1, sum(x["family"] == f for x in r.values())), 1) for f in FAMS}
                                for c, r in cs.items() if r}
        out["tool_labels"][m] = {c: dict(collections.Counter(x.get("tool_label") for x in r.values()))
                                 for c, r in cs.items() if r and c.startswith("tool")}
    for c in CONDS:
        if R["Qwen3.5-4B-A40"][c] and R[base][c]:
            out["hardware_noise_a40_vs_a100"][c] = mcnemar(R["Qwen3.5-4B-A40"][c], R[base][c])
    drop = lambda r: {i: x for i, x in r.items() if i not in CONTAM}
    for m, cs in R.items():
        out["summary_excl_contam"][m] = {c: summary(drop(r)) for c, r in cs.items() if r}
    for m in ckpts:
        out["paired_vs_base_a100_excl_contam"][m] = {c: mcnemar(drop(R[base][c]), drop(R[m][c])) for c in CONDS if R[base][c] and R[m][c]}
        for t in ("off", "on"):
            a, b = ifeval_rows(ftdir, base, t), ifeval_rows(ftdir, m, t)
            if not (a and b): continue
            both = [k for k in a if k in b and a[k]["closed"] and b[k]["closed"]]
            out["ifeval_paired"].setdefault(m, {})[t] = {
                "all": mcnemar(a, b),
                "closed_in_both": dict(mcnemar({k: a[k] for k in both}, {k: b[k] for k in both}),
                                       base_pct=round(100 * sum(ok(a[k]) for k in both) / max(1, len(both)), 1),
                                       ft_pct=round(100 * sum(ok(b[k]) for k in both) / max(1, len(both)), 1)),
                "closed_base": sum(x["closed"] for x in a.values()), "closed_ft": sum(x["closed"] for x in b.values())}
    for m in ckpts:
        out["paired_vs_base_a100"][m] = {c: mcnemar(R[base][c], R[m][c]) for c in CONDS if R[base][c] and R[m][c]}
        out["shield"][m] = {t: shield(R[base][f"notool_{t}"], R[m][f"notool_{t}"], R[base][f"tool_{t}"], R[m][f"tool_{t}"])
                            for t in ("off", "on") if all(R[x][f"{k}_{t}"] for x in (base, m) for k in ("notool", "tool"))}
    for p in glob.glob(f"{ftdir}/ifeval_summary_*.json"):
        out["ifeval"][os.path.basename(p)[len("ifeval_summary_"):-5]] = json.load(open(p))
    json.dump(out, open(outp, "w"), indent=1)
    # ---- readable print
    print("ACCURACY (strict 6 s.f.) %  [95% CI]  (no answer)")
    for m in out["summary"]:
        print(f"  {m:24s} " + "  ".join(f"{c}: {s['pct']:5.1f} [{s['ci95'][0]},{s['ci95'][1]}] ({s['no_answer']})"
                                       for c, s in out["summary"][m].items()))
    print("HARDWARE NOISE, same model A40 vs A100 (answers that flip):")
    for c, x in out["hardware_noise_a40_vs_a100"].items(): print(f"  {c}: +{x['gained']} / -{x['lost']}  z={x['z']}")
    for m in ckpts:
        print(f"PAIRED vs base(A100): {m}")
        for c, x in out["paired_vs_base_a100"][m].items(): print(f"  {c}: +{x['gained']} / -{x['lost']}  net {x['net']}  z={x['z']}")
        for t, s in out["shield"][m].items(): print(f"  shield thinking {t}: {s}")
    print("EXCLUDING the 12 contamination-hit ids:")
    for m in ckpts:
        print(f"  {m}: " + "  ".join(f"{c}: +{x['gained']}/-{x['lost']} z={x['z']}" for c, x in out["paired_vs_base_a100_excl_contam"][m].items()))
    print("IFEVAL:", json.dumps(out["ifeval"], indent=1))
    print("IFEVAL PAIRED vs base:", json.dumps(out["ifeval_paired"], indent=1))

if __name__ == "__main__":
    main()
