"""Build ablation variants by REWRITING one claim out of an explanation.

WHY REWRITE AND NOT DELETE (SOURCE-01, 2026-08-22). The paper's own shipped artifact
(`rewritten_text` in the fennec s244 widget) shows their ablation is a REWRITE: the whole
explanation is reproduced with one fact surgically excised and the prose re-joined. Their
prompt was never published; the operation is specified by their outputs.

Our previous `build_ablations.py` DELETED the sentence carrying the claim, which takes
neighbouring claims as collateral and damages fluency - i.e. Rival R2 (coherence) is
uncontrolled under deletion and controlled by construction under rewriting. Deletion also
makes our Delta incomparable with their Delta-mse.

Worked example (Laxman pos 254, k=0), removing
"The text mentions a Test match between India and West Indies":
    -  ...factual cricket history format, with a summary of a Test match between India and West Indies.
    +  ...factual cricket history format, with a summary of a Test series.
Eight words changed; the other two paragraphs byte-identical. Deletion would have removed
the whole first paragraph and destroyed two other claims.

KNOWN LIMITATION, logged up front (H3, raised by AG 2026-08-22): single-claim ablation
measures the MARGINAL contribution of a claim given the rest of the explanation stays put.
Content duplicated elsewhere in the same explanation survives the ablation, so redundant
claims will score Delta ~ 0 regardless of truth. Not fixed here - measured later.

NOT VERIFIED YET: that exactly the target claim disappears and the others survive. Verifying
that needs a SEMANTIC matcher - re-decomposing and string-matching does not work, because
Haiku re-phrases the same claim differently on a second pass (the same failure that left
NOISE-01 with 6 recurring claims out of 401). The matcher is on the build list; until it
exists, this output is unverified and must be labelled so.

Input : explanations.parquet + claims.parquet
Output: ablations.parquet - one row per variant.
        variant "full"   -> the original explanation, once per (doc_id,pos,k)
        variant "<claim_id>" -> that explanation with the one claim rewritten out
"""
import argparse, json, os, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import pyarrow as pa, pyarrow.parquet as pq

MODEL = "claude-haiku-4-5-20251001"     # same pinned snapshot as Stage 3 and Stage 4

PROMPT = """Below is an explanation written by an interpretability tool about a language \
model's internal activation, and ONE claim that the explanation makes.

Rewrite the explanation so that it no longer makes that claim, changing as little as possible.

<explanation>
{explanation}
</explanation>

<claim_to_remove>
{claim}
</claim_to_remove>

Rules:
- Remove ONLY that claim. Every other assertion in the explanation must survive, in its \
original wording wherever possible.
- Do NOT delete a whole sentence or bullet just because the claim sits inside it. Excise the \
claim and re-join the remaining text so it reads naturally.
- Keep the same structure: the same number of bullets/paragraphs, the same headers, the same \
formatting.
- Do NOT add any new information, and do not rephrase surviving content beyond what is needed \
to keep the sentence grammatical.
- If the claim IS the entire content of a bullet, keep the bullet and its header and leave \
only the parts that do not assert the claim; if nothing remains, keep the header alone.

Return JSON matching the provided schema."""

SCHEMA = {"type": "object", "properties": {"rewritten": {"type": "string"}},
          "required": ["rewritten"], "additionalProperties": False}
OUTPUT_CONFIG = {"format": {"type": "json_schema", "schema": SCHEMA}}

def params_for(explanation, claim):
    # max_tokens generous: the model must reproduce the WHOLE explanation, not a fragment.
    return dict(model=MODEL, max_tokens=3000, output_config=OUTPUT_CONFIG,
                messages=[{"role": "user",
                           "content": PROMPT.format(explanation=explanation, claim=claim)}])

def rewrite(client, explanation, claim, tries=3):
    """Retries on truncation. Same degeneration guard as Stage 3: a truncated rewrite would
    silently become a SHORTER explanation, which the AR would score as a huge Delta - a
    fabricated result, not a crash. So truncation must never be accepted."""
    last = None
    for attempt in range(1, tries + 1):
        r = client.messages.create(**params_for(explanation, claim))
        if r.stop_reason != "max_tokens":
            txt = json.loads(next(b.text for b in r.content if b.type == "text"))["rewritten"]
            if not txt.strip():
                last = "empty rewrite"
            elif len(txt.strip()) >= len(explanation.strip()):
                # Removing a claim can never make the text longer. Observed at 0.8% on the
                # first full run: instead of excising, the model invented replacement content
                # ("the greatest victories in Indian cricket") that the AV never wrote. The AR
                # would then score fabricated text and return a real-looking Delta. Also
                # catches the case where the explanation comes back unchanged (Delta == 0
                # for no reason). Both are silent corruption, so both must retry.
                last = (f"not shorter than input ({len(txt.strip())} >= "
                        f"{len(explanation.strip())}) - likely invented or unchanged")
            else:
                return txt, r.model
        else:
            last = f"truncated at {r.usage.output_tokens} output tokens"
        print(f"      retry {attempt}/{tries}: {last}", flush=True)
    raise RuntimeError(f"rewrite failed on all {tries} attempts ({last})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explanations", required=True)
    ap.add_argument("--claims", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="first N claims only (smoke)")
    a = ap.parse_args()

    import anthropic
    client = anthropic.Anthropic()

    E = pq.read_table(a.explanations, columns=["doc_id", "pos", "k", "explanation"]).to_pylist()
    C = pq.read_table(a.claims).to_pylist()
    expl = {(e["doc_id"], e["pos"], e["k"]): e["explanation"] for e in E}
    if a.limit: C = C[:a.limit]

    rows = []
    # one "full" row per explanation - the baseline every Delta is measured against
    seen = set()
    for c in C:
        key = (c["doc_id"], c["pos"], c["k"])
        if key in seen: continue
        seen.add(key)
        rows.append({"variant": "full", "claim_id": "", "doc_id": key[0], "pos": key[1],
                     "k": key[2], "level": "", "subtype": "", "claim": "",
                     "text": expl[key], "rewrite_model": ""})
    print(f"{len(seen)} explanations, {len(C)} claims -> {len(seen) + len(C)} variants")

    def work(c):
        key = (c["doc_id"], c["pos"], c["k"])
        try:
            txt, m = rewrite(client, expl[key], c["text"])
            return c, txt, m, None
        except Exception as e:
            return c, None, None, f"{type(e).__name__}: {e}"

    fails, done, t0 = [], 0, time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for c, txt, m, err in ex.map(work, C):        # map preserves input order
            done += 1
            if err:
                fails.append({"claim_id": c["claim_id"], "error": err}); continue
            rows.append({"variant": c["claim_id"], "claim_id": c["claim_id"],
                         "doc_id": c["doc_id"], "pos": c["pos"], "k": c["k"],
                         "level": c["level"], "subtype": c["subtype"], "claim": c["text"],
                         "text": txt, "rewrite_model": m})
            if done % 200 == 0: print(f"  [{done}/{len(C)}] {time.time()-t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), a.out)

    abl = [r for r in rows if r["variant"] != "full"]
    full = {(r["doc_id"], r["pos"], r["k"]): r["text"] for r in rows if r["variant"] == "full"}
    d = [len(r["text"]) - len(full[(r["doc_id"], r["pos"], r["k"])]) for r in abl]
    unchanged = sum(r["text"].strip() == full[(r["doc_id"], r["pos"], r["k"])].strip() for r in abl)
    import statistics as st
    print(f"\n{len(rows)} variants ({len(full)} full + {len(abl)} ablated) -> {a.out}")
    print(f"failures: {len(fails)}   wall {time.time()-t0:.0f}s")
    print(f"char delta vs full:  median {st.median(d):+.0f}  min {min(d):+d}  max {max(d):+d}")
    print(f"IDENTICAL to full (nothing removed): {unchanged}  ({100*unchanged/len(abl):.1f}%)")
    if fails:
        json.dump(fails, open(a.out.replace(".parquet", "_failures.json"), "w"), indent=1)
        print(f"!! failures written to {a.out.replace('.parquet','_failures.json')}")

if __name__ == "__main__":
    main()
