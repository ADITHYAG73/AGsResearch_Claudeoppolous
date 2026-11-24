"""bc_table.py -- BIASCORR-01: markdown summary table from runs/BC/analysis.json (+ data facts)."""
import json, pathlib
D = pathlib.Path(__file__).resolve().parents[1] / "runs" / "BC"
J = json.load(open(D / "analysis.json"))
keep = ["g", "ema_a64", "ema_a32", "lerp_a64", "ema_a64_bcdiv64", "ema_a64_bcmul64", "ema_a64_bf16", "ema_a64_fp16"]
L = ["# BIASCORR-01 tables", "",
     "Cell = fp16-matrix (X16) entries of the exported update NOT predicted exactly / tensors predicted exactly "
     "(of 152). Emulation = bc_lib (Horner PE5, numpy pairwise f32 norm, X0 = fp16(M/(1.01 n + eps))). "
     "Each run is scored against the oracle gradient of ITS gradient version (grad column).", ""]
for r in ("1", "2"):
    R = J[r]; runs = list(R["runs"])
    L.append(f"## rank {r} ({R['total_entries']} entries)")
    L.append("")
    L.append("| candidate | " + " | ".join(f"{n.split('_',1)[1]} ({R['runs'][n]['data']['gradient'].split('_')[1]})" for n in runs) + " |")
    L.append("|---" * (len(runs) + 1) + "|")
    L.append("| DATA: tensors / X16 entries changed vs same-version beta=0 | " + " | ".join(
        (f"{R['runs'][n]['data']['f32_tensors']} / {R['runs'][n]['data']['x16_entries']}" if "vs" in R["runs"][n]["data"] else "no b0")
        for n in runs) + " |")
    for eps in ("1e-07", "0"):
        for c in keep:
            k = f"{c}|eps={eps}"
            L.append(f"| {c}, eps {eps} | " + " | ".join(
                f"{R['runs'][n]['cands'][k]['x16_mismatch']} / {R['runs'][n]['cands'][k]['tensors_exact']}" for n in runs) + " |")
    L.append("")
(D / "table.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
