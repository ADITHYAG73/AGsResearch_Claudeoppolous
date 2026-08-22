"""Prepare a BLIND grading file. The only script that ever sees the full claims table.

Input : claims.parquet (Stage 3) + activations.parquet (Stage 2, for prefixes)
Output: grading_<grader>.json — ONLY claim_id, claim text, prefix. Nothing else.
        Shuffled with a fixed seed, interleaved across passages.
        Optionally appends a RETEST block: `--retest N` re-inserts N already-seen claims
        under fresh ids at the end, for self-agreement measurement.

Blindness is a property of THIS FILE's contents, not of the UI.
"""
import argparse, hashlib, json, random
import pyarrow.parquet as pq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True)
    ap.add_argument("--activations", required=True)
    ap.add_argument("--grader", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--retest", type=int, default=50, help="claims to re-present at the end for self-agreement")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    C = pq.read_table(a.claims).to_pylist()
    A = pq.read_table(a.activations, columns=["doc_id", "pos", "detokenized_text_truncated"]).to_pylist()
    prefix = {(r["doc_id"], r["pos"]): r["detokenized_text_truncated"] for r in A}

    # STRIP: keep only what the grader may see
    items = [{"claim_id": c["claim_id"], "claim": c["text"],
              "prefix": prefix[(c["doc_id"], c["pos"])]} for c in C]

    rng = random.Random(hashlib.sha256(f"{a.seed}|{a.grader}".encode()).digest())
    rng.shuffle(items)                                    # interleaves across passages

    # RETEST block: re-present N random already-shown claims under new ids, at the END
    retest = rng.sample(items, min(a.retest, len(items)))
    for r in retest:
        items.append({"claim_id": f"RETEST::{r['claim_id']}", "claim": r["claim"], "prefix": r["prefix"]})

    out = a.out or f"grading_{a.grader}.json"
    json.dump({"grader": a.grader, "seed": a.seed, "n_main": len(C), "n_retest": len(retest),
               "items": items}, open(out, "w"), ensure_ascii=False, indent=1)
    leaked = set(items[0].keys()) - {"claim_id", "claim", "prefix"}
    print(f"{len(items)} items ({len(C)} + {len(retest)} retest) -> {out}")
    print(f"fields exposed to grader: {sorted(items[0].keys())}   leaked: {leaked or 'none'}")

if __name__ == "__main__":
    main()
