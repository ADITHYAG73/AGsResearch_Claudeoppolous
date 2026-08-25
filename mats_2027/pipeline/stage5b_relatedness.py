"""Stage 5b - relatedness: for each FALSE claim, how connected is it to the passage?

Asked ONLY of claims already judged NOT_IN_TEXT or CONTRADICTED, exactly as the paper does:
its `vibe_response` emits one relatedness tag per false claim and none for supported ones
(fennec s244: 11 claims, 3 false, 3 relatedness tags).

LABEL SET. The paper's shipped example contains only DIRECT and ADJACENT. Its prose compares
"false claims that are somewhat relevant to the context" against "unrelated claims", which
requires a third category the example happens not to contain. UNRELATED is therefore OUR
name for that third bucket - the concept is theirs, the string is an inference. Collapse for
comparison with the paper: related = DIRECT + ADJACENT, unrelated = UNRELATED.

Three values rather than the paper's binary, for the same reason S/C/N was kept over
SUPPORTED/NOT_IN_TEXT: collapsing later is free, un-collapsing is impossible. H1 needs the
related-false cell, and DIRECT vs ADJACENT is exactly the near-miss / same-world distinction
that cell is about.

SIMPLIFIED TO BINARY after a failed pilot (2026-08-25). The first version asked for the
paper's three-way DIRECT / ADJACENT / UNRELATED. On the 22 false claims of one hand-checked
activation it returned ADJACENT 20 times, including cases AG had explicitly ruled DIRECT
("mentions 2002" against a passage stating 2001; "mentions West Indies" against a passage
naming Australia as the opponent). It was also SELF-INCONSISTENT - the same claim text got
ADJACENT at k=0 and DIRECT at k=2 - and its single UNRELATED was plainly wrong ("the text is
an encyclopedic article" about a Wikipedia passage).

The deciding argument was not that the prompt could be fixed, but that H1 does not need the
split. H1 asks whether the RELATED-FALSE cell is two populations under one label; the test is
bimodality of Delta WITHIN that cell. It needs the cell to exist, not to be subdivided. So
the fine distinction was bought at the cost of reliability and was never used.

CONVENTION (binary):
  RELATED    the claim belongs to the same subject area as the passage - whether it names a
             wrong value for something the passage does discuss, or names something absent
             from the passage that still belongs to the same topic, domain or period.
  UNRELATED  the claim has no connection to what the passage is about.

PREDICTION RECORDED BEFORE RUNNING (2026-08-25): UNRELATED will be under 5%. The AV is
describing a cricket activation, so it confabulates WITHIN cricket - all 22 false claims on
the one activation inspected by hand were cricket. If that holds, the paper's
related-vs-unrelated contrast is not reproducible on this corpus. That does not kill H1,
which needs only the related-false cell, but it must be stated rather than discovered later.

NOTE: this is the FOURTH stage on claude-haiku-4-5-20251001 (decompose, judge, rewrite,
relatedness). A blind spot shared across all four is invisible to us. AG's two-judge idea
(runningdoc_AG.md, 2026-08-21) is the only proposed check.

Input : verdicts.parquet + activations.parquet   Output: relatedness.parquet
"""
import argparse, json, os, time
from concurrent.futures import ThreadPoolExecutor
import pyarrow as pa, pyarrow.parquet as pq

MODEL = "claude-haiku-4-5-20251001"

PROMPT = """A language model was reading a text passage. An interpretability tool inspected \
the model's internal activation at one token and described it. That description was split \
into claims, and this claim was already judged NOT supported by the passage.

Your job is only to say HOW CONNECTED the false claim is to the passage.

<passage>
{prefix}
</passage>

<false_claim>
{claim}
</false_claim>

Answer with exactly one of:

  RELATED    The claim belongs to the SAME SUBJECT AREA as the passage. This covers both:
               - the claim names a wrong value for something the passage DOES discuss
                 (a different year, a different counterparty, a different figure), and
               - the claim names something ABSENT from the passage that still belongs to the
                 same topic, domain, field or period.
             A claim describing the passage's own genre, format or register is RELATED.

  UNRELATED  The claim has NO connection to what the passage is about - a different subject
             area entirely, such that someone reading the passage would not recognise the
             claim as being about the same thing at all.

The test: would a reader of this passage say the claim is at least ABOUT the same subject,
even though it is false? If yes -> RELATED. Only if the claim comes from a different subject
area entirely -> UNRELATED.

Being false is NOT evidence of being unrelated. Most false claims here are expected to be
RELATED. Reserve UNRELATED for genuine topic changes.

Return JSON matching the provided schema."""

SCHEMA = {"type": "object",
          "properties": {"relatedness": {"type": "string",
                                         "enum": ["RELATED", "UNRELATED"]}},
          "required": ["relatedness"], "additionalProperties": False}
OUTPUT_CONFIG = {"format": {"type": "json_schema", "schema": SCHEMA}}

def params_for(prefix, claim):
    return dict(model=MODEL, max_tokens=200, output_config=OUTPUT_CONFIG,
                messages=[{"role": "user", "content": PROMPT.format(prefix=prefix, claim=claim)}])

def rate(client, prefix, claim, tries=3):
    for _ in range(tries):
        r = client.messages.create(**params_for(prefix, claim))
        if r.stop_reason != "max_tokens":
            return json.loads(next(b.text for b in r.content if b.type == "text"))["relatedness"], r.model
    raise RuntimeError("truncated on all attempts")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdicts", required=True)
    ap.add_argument("--activations", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    import anthropic
    client = anthropic.Anthropic()

    V = pq.read_table(a.verdicts).to_pylist()
    prefix = {(r["doc_id"], r["pos"]): r["detokenized_text_truncated"] for r in
              pq.read_table(a.activations,
                            columns=["doc_id", "pos", "detokenized_text_truncated"]).to_pylist()}
    F = [r for r in V if r["verdict"] != "SUPPORTED"]
    if a.limit: F = F[:a.limit]
    print(f"{len(V)} claims total · {len(F)} FALSE -> relatedness asked of these only "
          f"(supported claims are skipped, as the paper does)")

    def work(c):
        try:
            v, m = rate(client, prefix[(c["doc_id"], c["pos"])], c["claim"])
            return c, v, m, None
        except Exception as e:
            return c, None, None, f"{type(e).__name__}: {e}"

    rows, fails, done, t0 = [], [], 0, time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for c, v, m, err in ex.map(work, F):
            done += 1
            if err: fails.append({"claim_id": c["claim_id"], "error": err}); continue
            rows.append({"claim_id": c["claim_id"], "doc_id": c["doc_id"], "pos": c["pos"],
                         "k": c["k"], "level": c["level"], "subtype": c["subtype"],
                         "claim": c["claim"], "verdict": c["verdict"],
                         "relatedness": v, "relatedness_model": m})
            if done % 200 == 0: print(f"  [{done}/{len(F)}] {time.time()-t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), a.out)
    from collections import Counter
    c = Counter(r["relatedness"] for r in rows)
    print(f"\n{len(rows)} rated -> {a.out}   failures: {len(fails)}   wall {time.time()-t0:.0f}s")
    for k in ("RELATED", "UNRELATED"):
        print(f"  {k:10s} {c[k]:5d}  ({100*c[k]/max(len(rows),1):5.1f}%)")
    print(f"\n  RELATED = {c['RELATED']}   <- H1 lives entirely in this cell")
    print(f"  predicted UNRELATED <5%: "
          f"{'HELD' if c['UNRELATED'] < 0.05*len(rows) else 'BROKEN'}")
    print("\n  by claim level:")
    for lv in ("THEME", "ENTITY", "DETAIL"):
        sub = [r for r in rows if r["level"] == lv]
        if not sub: continue
        cc = Counter(r["relatedness"] for r in sub)
        print(f"    {lv:7s} n={len(sub):4d}  RELATED {cc['RELATED']:4d}  "
              f"UNRELATED {cc['UNRELATED']:4d}")
    if fails: json.dump(fails, open(a.out.replace(".parquet", "_failures.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
