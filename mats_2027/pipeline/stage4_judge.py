"""Stage 4 - judge each claim against the prefix the model actually read.

SCALE: S / C / N, identical to the human harness (harness/grade.html), so judge and human
labels are directly comparable. The paper's shipped verdicts are binary (SUPPORTED /
NOT_IN_TEXT); C+N collapse to their "false" bucket for comparison, and the split is kept
because H1 needs it - a claim merely ABSENT from the text can still be encoded in the
activation, a claim the text CONTRADICTS is far less likely to be a faithful readout.

CONVENTIONS are AG's, fixed on 2026-08-21 in grade.html and extended 2026-08-22 after the
warm-up. They are reproduced here VERBATIM in intent so that tomorrow's agreement number
measures one scale against itself, not two different scales against each other.

Input : claims.parquet + activations.parquet     Output: verdicts.parquet
"""
import argparse, json, os, time
from concurrent.futures import ThreadPoolExecutor
import pyarrow as pa, pyarrow.parquet as pq

MODEL = "claude-haiku-4-5-20251001"     # same snapshot as Stage 3 and as the paper's grader

PROMPT = """A language model was reading a text passage. An interpretability tool looked at \
the model's internal activation at one token and wrote a description of it. That description \
was then split into individual claims about the passage.

You are grading ONE such claim against the EXACT passage the model had read at that point. \
The passage stops mid-text on purpose - that is precisely how much the model had seen.

<passage>
{prefix}
</passage>

<claim>
{claim}
</claim>

Is this claim supported BY THIS PASSAGE? You are NOT judging whether it is true about the \
world - only whether this passage supports it.

Answer with exactly one of:
  SUPPORTED   - the passage says this, or directly implies it
  CONTRADICTED - the passage says otherwise
  NOT_IN_TEXT - absent; the passage neither states it nor disputes it

Rules:
- Grade against the passage AS WRITTEN. Typos do not break a match when the referent is \
unambiguous.
- VAGUE IS NOT FALSE. "The text is about cricket" for a Tendulkar biography is imprecise, \
not fabricated - that is SUPPORTED.
- QUOTE CLAIMS ("The text contains the sentence/phrase X"): SUPPORTED only if that string is \
actually present in the passage, allowing for ellipsis and trivial typos. Topical resemblance \
is NOT enough. If the passage discusses the same subject but the quoted wording is not there, \
the quote is invented -> NOT_IN_TEXT (or CONTRADICTED if the passage states something the \
quote directly contradicts).
- Claims about the FINAL TOKEN or where the passage ends: grade against how the passage \
actually ends.
- Predictions about what comes NEXT ("likely continues with...") - grade only the factual \
part about what the passage already contains. If there is no such factual part, NOT_IN_TEXT.
- A different name for the same thing (alias, translation, nickname) is SUPPORTED when the \
referent is unambiguous.
- Absent AND undisputed is NOT_IN_TEXT, never CONTRADICTED. Reserve CONTRADICTED for cases \
where the passage asserts something incompatible with the claim.

Return JSON matching the provided schema."""

SCHEMA = {"type": "object",
          "properties": {"verdict": {"type": "string",
                                     "enum": ["SUPPORTED", "CONTRADICTED", "NOT_IN_TEXT"]}},
          "required": ["verdict"], "additionalProperties": False}
OUTPUT_CONFIG = {"format": {"type": "json_schema", "schema": SCHEMA}}

def params_for(prefix, claim):
    return dict(model=MODEL, max_tokens=200, output_config=OUTPUT_CONFIG,
                messages=[{"role": "user",
                           "content": PROMPT.format(prefix=prefix, claim=claim)}])

def judge(client, prefix, claim, tries=3):
    for _ in range(tries):
        r = client.messages.create(**params_for(prefix, claim))
        if r.stop_reason != "max_tokens":
            txt = next(b.text for b in r.content if b.type == "text")
            return json.loads(txt)["verdict"], r.model
    raise RuntimeError("truncated on all attempts")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True)
    ap.add_argument("--activations", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    import anthropic
    client = anthropic.Anthropic()

    C = pq.read_table(a.claims).to_pylist()
    prefix = {(r["doc_id"], r["pos"]): r["detokenized_text_truncated"] for r in
              pq.read_table(a.activations,
                            columns=["doc_id", "pos", "detokenized_text_truncated"]).to_pylist()}
    if a.limit: C = C[:a.limit]

    def work(c):
        try:
            v, m = judge(client, prefix[(c["doc_id"], c["pos"])], c["text"])
            return c, v, m, None
        except Exception as e:
            return c, None, None, f"{type(e).__name__}: {e}"

    rows, fails, done, t0 = [], [], 0, time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for c, v, m, err in ex.map(work, C):          # map preserves input order
            done += 1
            if err:
                fails.append({"claim_id": c["claim_id"], "error": err}); continue
            rows.append({"claim_id": c["claim_id"], "doc_id": c["doc_id"], "pos": c["pos"],
                         "k": c["k"], "level": c["level"], "subtype": c["subtype"],
                         "claim": c["text"], "verdict": v, "judge_model": m})
            if done % 200 == 0: print(f"  [{done}/{len(C)}] {time.time()-t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), a.out)
    from collections import Counter
    print(f"\n{len(rows)} verdicts -> {a.out}   failures: {len(fails)}   "
          f"wall {time.time()-t0:.0f}s")
    print("verdicts:", dict(Counter(r["verdict"] for r in rows)))
    for lv in ("THEME", "ENTITY", "DETAIL"):
        sub = [r for r in rows if r["level"] == lv]
        if sub:
            s = sum(r["verdict"] == "SUPPORTED" for r in sub)
            print(f"  {lv:7s} n={len(sub):5d}  SUPPORTED {100*s/len(sub):5.1f}%")
    if fails: json.dump(fails, open(a.out.replace(".parquet", "_failures.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
