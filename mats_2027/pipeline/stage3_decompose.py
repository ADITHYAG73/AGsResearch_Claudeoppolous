"""Stage 3 — decompose each explanation into atomic, gradable claims.

RECONSTRUCTION. The paper's decompose prompt was never published (not in the paper,
appendix, or official repo). This prompt reproduces the paper's OUTPUT FORMAT exactly
(<claim_N level=... subtype=...> tags; levels THEME/ENTITY/DETAIL) and borrows the
claim definition from the published "factual groundedness" grader — minus its
"exclude reasonable inferences" rule, which would filter out the claims H1 is about.

Input : explanations.parquet (Stage 2)     Output: claims.parquet
Pilot : --pilot uses the 6 SMOKE-01 explanations (no parquet needed)
Batch : --batch submits via the Message Batches API (50% off) — use ONLY after the
        prompt is frozen on pilot output.
"""
import argparse, json, os, re, sys, time
import pyarrow as pa, pyarrow.parquet as pq

MODEL = "claude-haiku-4-5-20251001"   # pinned: the exact grader snapshot stamped in the paper's widget data
PROMPT = """You are helping evaluate an interpretability tool (a Natural Language Autoencoder, NLA). \
The NLA looked at a language model's internal activation — taken while the model read a text \
passage — and wrote the explanation below describing what that activation encodes. You do NOT \
see the passage. Your job is only to break the explanation into atomic claims.

<explanation>
{explanation}
</explanation>

Extract every specific declarative claim the explanation makes ABOUT THE TEXT the model was \
reading: what it contains, says, mentions, discusses, or is about. One claim per tag; each \
claim must stand alone and be checkable against a passage.

Tag each claim with a level and a subtype:
  THEME  — genre, topic, structure, era, register, format
  ENTITY — a named person, place, organisation, title, team, event
  DETAIL — a quote, date, number, score, statistic, specific value

Rules:
- EVERY claim must be a complete sentence asserting something about the text, e.g. \
"The text mentions Dravid", "The text is about a Test series in 2002", "The text contains the \
sentence '...'". A bare name, date or phrase on its own is NOT a claim — never output one.
- Each piece of information appears in exactly ONE claim. If you split a sentence into an \
entity claim and a detail claim, do NOT also output the unsplit sentence. Do not repeat.
- Use the explanation's own wording; do not add, infer, or complete anything.
- If the explanation is cut off mid-phrase (e.g. ends with "the India vs"), DROP that \
fragment entirely. Never guess how it would have continued.
- Include claims the explanation states as inferences ("suggests", "implies", "likely describes") \
— phrase them as the assertion being made about the text.
- EXCLUDE forward-looking claims entirely: anything about what the text is about to do, \
will do, sets up, signals, anticipates, or likely continues with. These describe the FUTURE \
of the text, which cannot be checked against it. "The text sets up a concluding statement" \
is forward-looking — do not emit it. "The text mentions Dravid" is backward-looking — emit it.
- EXCLUDE pure meta-commentary about the explanation itself and hedges with no content.
- Do not judge whether claims are true. Do not add claims the explanation does not make.

Return the claims as JSON matching the provided schema, numbered from 1."""

SUBTYPES = ["genre","topic","structure","era","register","format","content",
            "person","place","organisation","title","team","event",
            "quote","date","number","score","statistic","value"]
CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_idx": {"type": "integer"},
                    "level":     {"type": "string", "enum": ["THEME", "ENTITY", "DETAIL"]},
                    "subtype":   {"type": "string", "enum": SUBTYPES},
                    "text":      {"type": "string"},
                },
                "required": ["claim_idx", "level", "subtype", "text"],
                "additionalProperties": False,
            },
        },
        "total_claims": {"type": "integer"},
    },
    "required": ["claims", "total_claims"],
    "additionalProperties": False,
}
OUTPUT_CONFIG = {"format": {"type": "json_schema", "schema": CLAIMS_SCHEMA}}

def parse(text):
    """Structured output guarantees valid JSON matching CLAIMS_SCHEMA."""
    d = json.loads(text)
    claims = [{"claim_idx": c["claim_idx"], "level": c["level"], "subtype": c["subtype"],
               "text": " ".join(c["text"].split())} for c in d["claims"]]
    return claims, d["total_claims"]

def params_for(explanation):
    """The exact Messages params — shared by the sync path and the batch path."""
    return dict(model=MODEL, max_tokens=4000, output_config=OUTPUT_CONFIG,
                messages=[{"role": "user", "content": PROMPT.format(explanation=explanation)}])

def decompose_sync(client, explanation, tries=3):
    """Returns (raw_json_text, resolved_model_id).

    Retries on max_tokens truncation. Observed ~0.4% of the time (1/240 on POS-01) the
    model degenerates into emitting near-duplicate paraphrases (claim_idx ran past 100)
    until it hits the cap. Re-sampling the SAME input returns a clean 14-claim answer, so
    it is sampling noise, not a property of the explanation — a retry is the correct fix,
    not a larger max_tokens."""
    last = None
    for attempt in range(1, tries + 1):
        r = client.messages.create(**params_for(explanation))
        if r.stop_reason == "max_tokens":
            last = f"truncated at {r.usage.output_tokens} output tokens"
        else:
            txt = next(b.text for b in r.content if b.type == "text")
            d = json.loads(txt)
            # The SAME degeneration can finish under the cap. Its signature: the model emits
            # dozens of vacuous paraphrases while its own total_claims still reports the real
            # count (observed 95 emitted vs total_claims=4). Trust the declared count.
            if d["total_claims"] == len(d["claims"]):
                return txt, r.model
            last = f"declared {d['total_claims']} but emitted {len(d['claims'])} claims"
        print(f"      retry {attempt}/{tries}: {last}", flush=True)
    raise RuntimeError(f"degenerate output on all {tries} attempts ({last})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explanations", default="out/explanations.parquet")
    ap.add_argument("--out", default="out/claims.parquet")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--explanations-json", default=None, help="plain JSON list of {doc_id,pos,k,explanation}")
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--workers", type=int, default=8, help="concurrent requests (sync path)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N explanations (smoke)")
    a = ap.parse_args()

    import anthropic
    client = anthropic.Anthropic()            # reads ANTHROPIC_API_KEY from the environment

    if a.explanations_json:
        items = json.load(open(a.explanations_json))
    elif a.pilot:
        rt = json.load(open("mats_2027/runs/2026-08-20_smoke/roundtrip_results.json"))
        items = [{"doc_id": r["passage"], "pos": r["pos"], "k": 0, "explanation": r["explanation"]} for r in rt]
        a.out = "mats_2027/runs/2026-08-20_smoke/claims_pilot.parquet"
    else:
        items = pq.read_table(a.explanations, columns=["doc_id", "pos", "k", "explanation"]).to_pylist()

    if a.batch:
        raise SystemExit("batch mode: freeze the prompt on --pilot output first, then implement via AnthropicBatchProvider")

    if a.limit: items = items[:a.limit]

    from concurrent.futures import ThreadPoolExecutor
    def work(it):
        try:
            raw, resolved = decompose_sync(client, it["explanation"])
            return it, raw, resolved, None
        except Exception as e:                     # one bad item must not lose the other 239
            return it, None, None, f"{type(e).__name__}: {e}"

    rows, bad, done, failures = [], 0, 0, []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        # map() preserves input order, so `rows` is deterministic regardless of completion order
        for it, raw, resolved, err in ex.map(work, items):
            done += 1
            if err is not None:
                bad += 1
                failures.append({"doc_id": it["doc_id"], "pos": it["pos"], "k": it["k"], "error": err})
                print(f"[{done}/{len(items)}] {it['doc_id'][:28]:28s} pos={it['pos']} k={it['k']} "
                      f"-> FAILED  {err[:90]}", flush=True)
                continue
            claims, tot = parse(raw)
            if not claims or (tot is not None and tot != len(claims)): bad += 1
            for c in claims:
                rows.append({"claim_id": f"{it['doc_id']}:{it['pos']}:k{it['k']}:c{c['claim_idx']}",
                             "doc_id": it["doc_id"], "pos": it["pos"], "k": it["k"], **c,
                             "decompose_raw": raw, "decompose_model": resolved})
            print(f"[{done}/{len(items)}] {it['doc_id'][:28]:28s} pos={it['pos']} k={it['k']} "
                  f"-> {len(claims)} claims  (declared {tot})", flush=True)
    print(f"\nwall clock: {time.time()-t0:.1f}s at {a.workers} workers", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), a.out)
    from collections import Counter
    print(f"\n{len(rows)} claims from {len(items)} explanations -> {a.out}   parse problems: {bad}")
    print("levels:", dict(Counter(r["level"] for r in rows)))
    print("subtypes:", dict(Counter(r["subtype"] for r in rows)))
    if failures:
        json.dump(failures, open(a.out.replace(".parquet", "_failures.json"), "w"), indent=1)
        print(f"\n!! {len(failures)} explanations FAILED - written to "
              f"{a.out.replace('.parquet','_failures.json')}")
        for f in failures[:10]: print("   ", f["doc_id"], f["pos"], f["k"], f["error"][:100])

if __name__ == "__main__":
    main()
