"""Stage 5a - semantic matcher: group claims that assert the SAME thing across resamples.

WHY THIS EXISTS. NOISE-02 found only 22 usable recurring claims out of 2065 because
recurrence was matched with `norm()` - lowercase, strip non-alphanumerics, exact string
compare. At T=1 the AV writes a different explanation each resample and Haiku then phrases
the same idea differently in each, so almost nothing matches verbatim. One activation
(VVS Laxman pos 254) produced 32 claims, 30 distinct strings, 2 exact matches - while
containing FOUR wordings of one genre claim and THREE of one Dravid claim.

Consequence: the paired-Delta noise, and therefore H2's kill condition (does spread fall as
1/sqrt(K)?), are unmeasurable without this. It also unblocks ABLATE-01's unverified ablation
fidelity check and H3's redundancy score. Three things, one component.

PRIOR ART. SOURCE-01 established the paper does this with an LLM call, not string matching:
its shipped `match_response` is `<appears_in_N>1,2,4,6,7,8</appears_in_N>`. The PROMPT was
never published (verified: the paper HTML has decompose/verify/vibe/match _response keys and
no corresponding _prompt keys; nothing matcher-like in kitft/natural_language_autoencoders or
EasyNLA). Output format is theirs; the prompt below is a reconstruction.

CONVENTION - AG's, accepted 2026-08-23. Two claims are the SAME iff removing either would
take the same information out of the explanation.
  SAME    : verb synonyms (mentions/references/discusses); articles, punctuation, tense;
            name completion (Dravid / Rahul Dravid); a dropped or added QUALIFIER on the same
            head noun (encyclopedic article / encyclopedic-biographical article); different
            phrasing of the frame.
  DIFFERENT: adds a new ENTITY, DATE or NUMBER (mentions 2002 != mentions the 2002 Test
            series against West Indies); different PREDICATE TYPE - "X appears in the text"
            vs "the text is about X"; quote claims unless the quoted string is materially the
            same.
  Tie-break: a dropped adjective is the SAME claim; an added noun is a DIFFERENT one.

WHY IT ERRS TOWARD SPLITTING. Grouping too aggressively pools Delta from genuinely different
claims, inflates measured noise, and would kill H2 falsely - a false negative we would
publish. Splitting too aggressively yields too few groups and simply announces itself.

Input : claims.parquet     Output: claim_groups.parquet  (claim_id -> group_id per activation)
"""
import argparse, json, os, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import pyarrow as pa, pyarrow.parquet as pq

MODEL = "claude-haiku-4-5-20251001"          # same pinned snapshot as stages 3, 4 and ablate

PROMPT = """An interpretability tool was run several times on the same internal activation \
from a language model, producing several independent descriptions of it. Each description was \
then split into atomic claims about the text the model was reading.

Below are all the claims, tagged with which run (k) each came from. The SAME underlying claim \
often appears in several runs, worded differently each time.

<claims>
{claims}
</claims>

Group the claims that assert the SAME thing about the text.

TEST: two claims are the same if removing either one would take the same information out of \
the description.

Treat as the SAME claim:
- verb synonyms: "mentions" / "references" / "discusses" / "contains"
- differences in articles, punctuation, tense, or how the sentence is framed
- a partial vs full name for the same referent
- a dropped or added ADJECTIVE on the same head noun

Treat as DIFFERENT claims:
- one adds an ENTITY, DATE or NUMBER the other does not.
      "the text mentions <THING>"  is NOT  "the text mentions <THING> in <YEAR>"
      "the text mentions <PLACE>"  is NOT  "the text mentions <PLACE>'s <INSTITUTION>"
- different PREDICATE TYPE - something appearing in the text vs the text being about it.
      "the text mentions <X>"  is NOT  "the text is about <X>"
      "the text mentions <X>"  is NOT  "the text is a description of <X>"
- different PROPERTY, even when both concern the writing. Genre, structure, tone, register \
and subject matter are DIFFERENT properties, and a claim about one is not a claim about
another.
      "the text uses a <GENRE> format"      is NOT  "the text is written as <STRUCTURE>"
      "the text has a <TONE> tone"          is NOT  "the text uses a <GENRE> format"
      "the text is organised as <STRUCTURE>" is NOT  "the text has a <TONE> tone"
- quote claims: the same only if the quoted string is materially the same.

Tie-break: a dropped adjective is the SAME claim; an added noun makes it a DIFFERENT claim.

When in doubt, SPLIT rather than merge. Too many small groups is a much smaller problem than \
one group holding claims that say different things.

LABEL RULE - this is how you check your own work. Each group needs a label that is ONE \
assertion every member of that group makes. If the label needs an "or" to cover the members \
("<GENRE-A> or <STRUCTURE-B> format"), the members do not assert the same thing: split the \
group until every label is a single assertion.

Every claim_id must appear in exactly one group. A claim that matches nothing else forms a \
group of one.

Return JSON matching the provided schema."""

SCHEMA = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "claim_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["label", "claim_ids"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["groups"],
    "additionalProperties": False,
}
OUTPUT_CONFIG = {"format": {"type": "json_schema", "schema": SCHEMA}}

def params_for(claims_block):
    return dict(model=MODEL, max_tokens=4000, output_config=OUTPUT_CONFIG,
                messages=[{"role": "user", "content": PROMPT.format(claims=claims_block)}])

def match_one(client, rows, tries=3):
    """Returns [(label, [claim_id, ...]), ...] for ONE activation.

    Retries when the model drops or invents claim_ids. A silently incomplete grouping would
    quietly shrink the sample the noise estimate is computed on, so partition-completeness is
    checked, not assumed."""
    block = "\n".join(f'<claim id="{r["claim_id"]}" k="{r["k"]}">{r["text"]}</claim>' for r in rows)
    want = {r["claim_id"] for r in rows}
    last = None
    for attempt in range(1, tries + 1):
        r = client.messages.create(**params_for(block))
        if r.stop_reason == "max_tokens":
            last = f"truncated at {r.usage.output_tokens} tokens"
        else:
            g = json.loads(next(b.text for b in r.content if b.type == "text"))["groups"]
            got = [cid for grp in g for cid in grp["claim_ids"]]
            if len(got) != len(set(got)):
                last = "a claim_id appears in more than one group"
            elif set(got) != want:
                last = (f"not a partition: {len(want - set(got))} missing, "
                        f"{len(set(got) - want)} invented")
            else:
                return [(grp["label"], grp["claim_ids"]) for grp in g], r.model
        print(f"      retry {attempt}/{tries}: {last}", flush=True)
    # REPAIR rather than lose the activation. The analysis needs a valid partition, nothing
    # more. Duplicates keep their first occurrence; claims the model dropped become
    # singletons - the conservative direction, consistent with "when in doubt, SPLIT".
    print(f"      repairing after {tries} failed attempts ({last})", flush=True)
    seen, fixed = set(), []
    for grp in g:
        keep = [c for c in grp["claim_ids"] if c in want and c not in seen]
        seen.update(keep)
        if keep: fixed.append((grp["label"], keep))
    for cid in want - seen:
        fixed.append(("", [cid]))
    return fixed, r.model


# ---------------------------------------------------------------- pass 2: the verifier
# Pilots 1 and 2 both OVER-MERGED: the model wrote a disqualifying label ("<GENRE-A> or
# <STRUCTURE-B>") and merged anyway, i.e. it would not apply the label rule to itself while
# clustering 32 claims in one shot. Auditing ONE small group in isolation is a far easier
# task, and guard-and-retry has worked everywhere else in this pipeline.
VERIFY_PROMPT = """These claims were grouped together on the grounds that they all assert the \
SAME thing about a text. Check that.

<group>
{claims}
</group>

Two claims assert the same thing if removing either one would take the same information out \
of the description.

SAME: verb synonyms; articles, punctuation, tense; a partial vs full name for one referent; \
a dropped or added ADJECTIVE on the same head noun.

DIFFERENT:
- one adds an ENTITY, DATE or NUMBER the other does not
- different PREDICATE TYPE: something appearing in the text vs the text being ABOUT it
- different PROPERTY. Genre, structure, tone, register and subject matter are DIFFERENT \
properties. A claim about the genre is not a claim about the tone, and neither is a claim \
about what the text is about.
- quote claims, unless the quoted string is materially the same

If every claim here asserts the same thing, return the group unchanged as a single subgroup.
Otherwise SPLIT it into subgroups, each holding only claims that do assert the same thing. A \
claim that matches nothing else in the group becomes a subgroup of one.

Each subgroup needs a label that is ONE assertion every member of it makes. If a label would \
need an "or" to cover its members, that subgroup must be split further.

Every claim_id given must appear in exactly one subgroup. Do not invent ids.

Return JSON matching the provided schema."""

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "subgroups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "claim_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["label", "claim_ids"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["subgroups"],
    "additionalProperties": False,
}

def verify_group(client, rows, tries=3):
    """Audit ONE proposed group. Returns [(label, [claim_id,...]), ...] - possibly split."""
    if len(rows) < 2:
        return [("", [r["claim_id"] for r in rows])]
    block = "\n".join(f'<claim id="{r["claim_id"]}">{r["text"]}</claim>' for r in rows)
    want = {r["claim_id"] for r in rows}
    last = None
    for attempt in range(1, tries + 1):
        r = client.messages.create(
            model=MODEL, max_tokens=2000,
            output_config={"format": {"type": "json_schema", "schema": VERIFY_SCHEMA}},
            messages=[{"role": "user", "content": VERIFY_PROMPT.format(claims=block)}])
        if r.stop_reason == "max_tokens":
            last = "truncated"
        else:
            g = json.loads(next(b.text for b in r.content if b.type == "text"))["subgroups"]
            got = [cid for sg in g for cid in sg["claim_ids"]]
            if len(got) != len(set(got)):
                last = "a claim_id appears in more than one subgroup"
            elif set(got) != want:
                last = f"not a partition: {len(want-set(got))} missing, {len(set(got)-want)} invented"
            elif any(" or " in sg["label"].lower() for sg in g) and attempt < tries:
                # The label rule, ENFORCED rather than requested - pilot 2 showed the model
                # writes the disqualifying label and merges anyway. But only retry on it:
                # the PARTITION is what the analysis uses; the label is a heuristic. Throwing
                # away a sound partition over label wording costs sample size for nothing.
                last = "a subgroup label still contains 'or'"
            else:
                return [(sg["label"], sg["claim_ids"]) for sg in g]
        print(f"      verify retry {attempt}/{tries}: {last}", flush=True)
    # after retries, fall back to MAXIMAL SPLIT rather than accept a bad merge: over-merging
    # inflates measured noise and would kill H2 falsely; under-merging only costs sample size.
    print(f"      verify failed ({last}) - falling back to singletons", flush=True)
    return [("", [r["claim_id"]]) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="first N activations only (smoke)")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip pass 2 (the verifier) - for comparing against the pilots")
    a = ap.parse_args()

    import anthropic
    client = anthropic.Anthropic()

    C = pq.read_table(a.claims).to_pylist()
    by_act = defaultdict(list)
    for c in C: by_act[(c["doc_id"], c["pos"])].append(c)
    acts = sorted(by_act)
    if a.limit: acts = acts[:a.limit]
    print(f"{sum(len(by_act[k]) for k in acts)} claims across {len(acts)} activations "
          f"(one model call per activation)")

    def work(key):
        try:
            groups, m = match_one(client, by_act[key])
            if not a.no_verify:
                by_id = {r["claim_id"]: r for r in by_act[key]}
                out = []
                for label, cids in groups:
                    out += verify_group(client, [by_id[c] for c in cids])
                groups = [(lbl or label, cids) for lbl, cids in out]
            return key, groups, m, None
        except Exception as e:
            return key, None, None, f"{type(e).__name__}: {e}"

    rows, fails, done, t0 = [], [], 0, time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for key, groups, m, err in ex.map(work, acts):
            done += 1
            if err:
                fails.append({"doc_id": key[0], "pos": key[1], "error": err}); continue
            for gi, (label, cids) in enumerate(groups, 1):
                for cid in cids:
                    rows.append({"claim_id": cid, "doc_id": key[0], "pos": key[1],
                                 "group_id": f"{key[0]}:{key[1]}:g{gi}", "group_label": label,
                                 "group_size": len(cids), "match_model": m})
            if done % 10 == 0: print(f"  [{done}/{len(acts)}] {time.time()-t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), a.out)

    from collections import Counter
    per_group = Counter()
    for r in rows: per_group[r["group_id"]] += 1
    dist = Counter(per_group.values())
    print(f"\n{len(rows)} claims -> {len(per_group)} groups -> {a.out}   "
          f"failures: {len(fails)}   wall {time.time()-t0:.0f}s")
    print("  group size distribution: " + "  ".join(f"{n}x{dist[n]}" for n in sorted(dist)))
    print(f"  groups with >=3 members: {sum(v for n, v in dist.items() if n >= 3)}   "
          f"(exact-string matching gave 22 on the full set)")
    if fails:
        json.dump(fails, open(a.out.replace(".parquet", "_failures.json"), "w"), indent=1)
        print(f"  !! failures -> {a.out.replace('.parquet','_failures.json')}")

if __name__ == "__main__":
    main()
