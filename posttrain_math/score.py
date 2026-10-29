#!/usr/bin/env python3
"""Score raw answers against the frozen answer key. Reads raw_*.jsonl, never re-runs a model.

CORRECT means right to 6 significant figures: |answer - truth| <= half a unit in the 6th
significant digit of the truth. For a truth of exactly zero (q["is_zero"], e.g. sin(180°), whose
stored mpmath value is a ~1e-43 residue of pi): |answer| < 5e-6. Fixed 2026-10-16; before that the
six such questions were unscoreable. For a question with
no real value: the model must say so ("undefined" or an equivalent).

Three outcomes are kept apart because they are different facts about the model:
  truncated / turn_limit   it never produced an answer inside the budget (not a maths error)
  unparsed                 it answered, but with no number we can read
  wrong                    it answered with a number, and the number is wrong

For the sandbox conditions every question also gets ONE label saying where it went:
  correct_with_tool · correct_without_calling · no_call_wrong · code_error ·
  tool_gave_wrong_value · right_value_wrong_answer · malformed_call · truncated · turn_limit
"""
import json, math, os, re, sys, glob
from collections import Counter, defaultdict

NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
UNDEF = re.compile(r"undefined|not defined|does not exist|no real|not a real|\bnan\b|\binf(inity)?\b|complex", re.I)

def half_unit(t, sf=6):
    return 5e-6 if t == 0 else 0.5 * 10 ** (math.floor(math.log10(abs(t))) - (sf - 1))

def matches(a, t, sf=6):
    if t == 0: return abs(a) < 5e-6
    return abs(a - t) <= half_unit(t, sf) * (1 + 1e-9)

def parse_final(text):
    body = text.split("</think>")[-1].replace(",", "").replace("−", "-").strip()
    if UNDEF.search(body) and not NUM.search(body.replace("undefined", "")):
        return "undefined", None
    ms = NUM.findall(body)
    if not ms: return ("undefined", None) if UNDEF.search(body) else ("unparsed", None)
    try: return "number", float(ms[-1])
    except ValueError: return "unparsed", None

def judge(r, q):
    """-> (status, close, label)   status in correct/wrong/unparsed/truncated/turn_limit/malformed"""
    end = r["end"]
    used = r["n_calls"] > 0
    if end in ("truncated", "turn_limit", "malformed_call"):
        return ("malformed" if end == "malformed_call" else end), False, end
    kind, val = parse_final(r.get("final", ""))
    if q["undefined"]:
        ok = kind == "undefined"
        return ("correct" if ok else ("unparsed" if kind == "unparsed" else "wrong")), ok, None
    t = 0.0 if q.get("is_zero") else float(q["truth"])   # is_zero: truth is exact 0; stored value is mpmath pi residue (~1e-43)
    if kind != "number":
        return ("unparsed" if kind == "unparsed" else "wrong"), False, None
    ok = matches(val, t)
    close = ok or (abs(val - t) <= 1e-3 * max(abs(t), 1e-12) if t != 0 else abs(val) < 1e-3)
    return ("correct" if ok else "wrong"), close, None

def tool_label(r, q, status):
    if status in ("truncated", "turn_limit", "malformed"): return {"malformed": "malformed_call"}.get(status, status)
    used = r["n_calls"] > 0
    if status == "correct": return "correct_with_tool" if used else "correct_without_calling"
    if not used: return "no_call_wrong"
    if all(c["rc"] != 0 for c in r["calls"]): return "code_error"
    if q["undefined"]: return "tool_gave_wrong_value"
    t = 0.0 if q.get("is_zero") else float(q["truth"])
    seen = any(matches(float(x), t, 6) or (t != 0 and abs(float(x) - t) <= 1e-9 * abs(t))
               for c in r["calls"] for x in NUM.findall(c["stdout"].replace(",", ""))
               if _is_float(x))
    return "right_value_wrong_answer" if seen else "tool_gave_wrong_value"

def _is_float(x):
    try: float(x); return True
    except ValueError: return False

def main():
    qpath, raws = sys.argv[1], sys.argv[2:]
    Q = {q["id"]: q for q in (json.loads(l) for l in open(qpath))}
    files = [f for pat in raws for f in sorted(glob.glob(pat))]
    for f in files:
        rows = [json.loads(l) for l in open(f)]
        if not rows: continue
        tool = rows[0]["cond"].startswith("tool")
        fam = defaultdict(Counter); labels = Counter(); toks = []
        out = []
        for r in rows:
            q = Q[r["id"]]
            status, close, _ = judge(r, q)
            lab = tool_label(r, q, status) if tool else None
            fam[q["family"]][status] += 1; fam[q["family"]]["close"] += close; fam[q["family"]]["n"] += 1
            if lab: labels[lab] += 1
            toks.append(r["gen_tokens"])
            out.append({"id": r["id"], "family": q["family"], "sub": q["sub"], "expr": q["expr"],
                        "arg_class": q["arg_class"], "depth": q["depth"], "truth": q["truth"],
                        "answer": r.get("final", "")[-80:], "status": status, "close_1e-3": close,
                        "tool_label": lab, "n_calls": r["n_calls"], "gen_tokens": r["gen_tokens"],
                        "model": r["model"], "cond": r["cond"]})
        with open(f.replace("raw_", "scored_"), "w") as fh:
            for o in out: fh.write(json.dumps(o, ensure_ascii=False) + "\n")
        n = len(rows); tot = Counter()
        for c in fam.values(): tot.update(c)
        print(f"\n=== {rows[0]['model']} · {rows[0]['cond']} · n={n} ===")
        print(f"{'family':<16}{'n':>5}{'correct':>9}{'within 1e-3':>13}{'wrong':>7}{'unparsed':>9}{'no answer':>10}")
        for k, c in list(fam.items()) + [("ALL", tot)]:
            na = c["truncated"] + c["turn_limit"] + c["malformed"]
            print(f"{k:<16}{c['n']:>5}{100*c['correct']/c['n']:>8.1f}%{100*c['close']/c['n']:>12.1f}%"
                  f"{c['wrong']:>7}{c['unparsed']:>9}{na:>10}")
        toks.sort()
        print(f"generated tokens per question: median {toks[len(toks)//2]}, max {toks[-1]}")
        if tool:
            print("where each question went:")
            for k, v in labels.most_common(): print(f"   {k:<26}{v:>5}  ({100*v/n:.1f}%)")

if __name__ == "__main__":
    main()
