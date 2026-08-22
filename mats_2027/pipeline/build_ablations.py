"""Build ablation variants for the paired-Δ noise measurement. LAPTOP, no GPU.

In : explanations_k8.json (pod)  +  claims (Stage 3, decomposed from those explanations)
Out: ablations.json — for each explanation: the 'full' text, plus one 'del:<claim>' variant
     per claim, made by removing the sentence that carries that claim.

The noise we want = sd of Δ for the SAME claim across the K resamples it appears in. So we
also report which claims recur (matched by normalised text) — those are the measurable ones.
"""
import argparse, json, re, difflib
import pyarrow.parquet as pq
from collections import defaultdict

def sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]

def carrier_sentence(expl_sents, claim_text):
    """The explanation sentence that best carries this claim (highest word overlap)."""
    cw = set(re.findall(r"\w+", claim_text.lower())) - {"the","text","a","an","of","about","in","contains","mentions","discusses","is"}
    best, score = None, 0.0
    for s in expl_sents:
        sw = set(re.findall(r"\w+", s.lower()))
        sc = len(cw & sw) / max(1, len(cw))
        if sc > score: best, score = s, sc
    return best, score

def norm(t):
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explanations", required=True)
    ap.add_argument("--claims", required=True)
    ap.add_argument("--out", default="ablations.json")
    a = ap.parse_args()

    expls = json.load(open(a.explanations))
    claims = pq.read_table(a.claims).to_pylist()
    by_expl = defaultdict(list)
    for c in claims: by_expl[(c["doc_id"], c["pos"], c["k"])].append(c)

    rows, unmatched = [], 0
    for e in expls:
        key = (e["doc_id"], e["pos"], e["k"])
        sents = sentences(e["explanation"])
        rows.append({**{k: e[k] for k in ("doc_id","pos","k")}, "claim_id": None, "variant": "full", "text": e["explanation"]})
        for c in by_expl.get(key, []):
            carrier, sc = carrier_sentence(sents, c["text"])
            if carrier is None or sc < 0.3: unmatched += 1; continue
            ablated = e["explanation"].replace(carrier, "", 1)
            ablated = re.sub(r"\n{3,}", "\n\n", ablated).strip()
            rows.append({**{k: e[k] for k in ("doc_id","pos","k")}, "claim_id": c["claim_id"],
                         "claim_text": c["text"], "claim_norm": norm(c["text"]), "level": c["level"],
                         "variant": f"del:{c['claim_id']}", "carrier": carrier, "match_score": round(sc,2),
                         "text": ablated})

    # recurrence: same normalised claim text across different k of the same activation
    rec = defaultdict(set)
    for r in rows:
        if r["variant"] != "full": rec[(r["doc_id"], r["claim_norm"])].add(r["k"])
    recurring = {k: sorted(v) for k, v in rec.items() if len(v) >= 3}

    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"{len(rows)} variants ({sum(r['variant']=='full' for r in rows)} full + "
          f"{sum(r['variant']!='full' for r in rows)} ablated) -> {a.out}   unmatched claims: {unmatched}")
    print(f"\nclaims recurring in >=3 of the K resamples (these are the measurable ones): {len(recurring)}")
    for (doc, cn), ks in sorted(recurring.items(), key=lambda x: -len(x[1]))[:12]:
        print(f"  [{doc[9:24]:15s}] k={ks}  {cn[:70]}")

if __name__ == "__main__":
    main()
