#!/usr/bin/env python3
"""FT-01 data prep. Runs ON THE TRAINING POD (never on the Mac).

1. Draw N conversations uniformly from nvidia/Nemotron-SFT-Instruction-Following-Chat-v4,
   split multi_turn_instruction_following (54 GB JSONL), WITHOUT downloading the file:
   seeded random byte offsets, HTTP range read, take the first complete line after the offset.
   (Row i is picked with probability proportional to the length of row i-1, which is independent
   of row i itself, so the draw is not biased toward long or short rows.)
2. Cut each conversation after the LAST assistant turn marked in metadata.train_turns_mask
   (the card: "Use metadata.train_turns_mask to select the assistant turns for training").
   Only that final turn is trained; earlier turns are context. Qwen's template renders reasoning
   only for turns after the last user message, so this is the one turn whose reasoning is visible.
3. Render with Qwen3.5's OWN chat template (reasoning_content -> <think>...</think>), check the
   generation prompt is an exact token prefix of the full conversation, drop rows > MAX_LEN tokens.
4. Contamination check against the 3,362 frozen questions: expression strings and 6-s.f. answers.
5. Write train.jsonl (token ids + index where labels start) and a report; progress lines throughout.
"""
import argparse, json, random, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = ("https://huggingface.co/datasets/nvidia/Nemotron-SFT-Instruction-Following-Chat-v4/"
       "resolve/main/data/multi_turn_instruction_following.jsonl")
SIZE = 54283343615          # bytes, from the HF tree API (2026-10-18)

def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)

RATE = {"429": 0, "other_err": 0}
def fetch_row(off, span=600_000, tries=40):
    # HF limits unauthenticated resolves to 3,000 per 5-minute fixed window (response headers,
    # 2026-10-18: ratelimit-policy "fixed window";q=3000;w=300). On 429, wait Retry-After and retry;
    # the first version gave up after 4 quick retries and would have silently dropped rows.
    import urllib.error
    for t in range(tries):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={off}-{off + span}"})
            data = urllib.request.urlopen(req, timeout=60).read()
            parts = data.split(b"\n")
            if len(parts) < 3:          # row longer than the span: widen
                span *= 4; continue
            return json.loads(parts[1])
        except urllib.error.HTTPError as e:
            if e.code == 429:
                RATE["429"] += 1
                time.sleep(int(e.headers.get("retry-after", "30")) + 1 + (off % 7))
            else:
                RATE["other_err"] += 1; time.sleep(3)
        except Exception:
            RATE["other_err"] += 1; time.sleep(3)
    return None

def cut_at_last_trained(row):
    msgs, mask = row["messages"], row["metadata"]["train_turns_mask"]
    idx = [i for i, m in enumerate(mask) if m]
    if not idx: return None
    last = idx[-1]
    if msgs[last]["role"] != "assistant": return None
    return msgs[: last + 1]

def norm(s): return re.sub(r"\s+", "", s).lower()

def sig6(x):
    v = float(x)
    if v == 0: return None
    return f"{v:.6g}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-draw", type=int, default=7000)
    ap.add_argument("--n-keep", type=int, default=5000)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)

    rng = random.Random(a.seed)
    offsets = sorted(rng.randrange(0, SIZE - 1_000_000) for _ in range(a.n_draw))
    log(f"drawing {a.n_draw} rows by range read")
    rows, t0 = [], time.time()
    with ThreadPoolExecutor(32) as ex:
        for i, r in enumerate(ex.map(fetch_row, offsets), 1):
            if r is not None: rows.append(r)
            if i % 500 == 0:
                log(f"  fetched {i}/{a.n_draw} ({len(rows)} parsed) {i/(time.time()-t0):.1f} rows/s  429s so far {RATE['429']} other errors {RATE['other_err']}")
    seen, uniq = set(), []
    for r in rows:
        if r["uuid"] not in seen: seen.add(r["uuid"]); uniq.append(r)
    log(f"parsed {len(rows)}, unique {len(uniq)}")
    rng.shuffle(uniq)

    out = open(f"{a.out}/train.jsonl", "w")
    stats = dict(drawn=a.n_draw, parsed=len(rows), unique=len(uniq), no_mask=0, prefix_mismatch=0,
                 too_long=0, kept=0, tokens_total=0, tokens_trained=0, n_turns=[], lengths=[])
    texts = []
    for r in uniq:
        if stats["kept"] >= a.n_keep: break
        msgs = cut_at_last_trained(r)
        if msgs is None: stats["no_mask"] += 1; continue
        full = tok.apply_chat_template(msgs, tokenize=False)
        prefix = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        if not full.startswith(prefix): stats["prefix_mismatch"] += 1; continue
        ids_p = tok(prefix, add_special_tokens=False)["input_ids"]
        ids_f = tok(full, add_special_tokens=False)["input_ids"]
        if ids_f[: len(ids_p)] != ids_p: stats["prefix_mismatch"] += 1; continue
        if len(ids_f) > a.max_len: stats["too_long"] += 1; continue
        out.write(json.dumps({"uuid": r["uuid"], "input_ids": ids_f, "label_start": len(ids_p)}) + "\n")
        stats["kept"] += 1; stats["tokens_total"] += len(ids_f)
        stats["tokens_trained"] += len(ids_f) - len(ids_p)
        stats["n_turns"].append(sum(m["role"] == "user" for m in msgs)); stats["lengths"].append(len(ids_f))
        texts.append(full)
    out.close()
    L = sorted(stats.pop("lengths")); nt = stats.pop("n_turns")
    stats["len_median"] = L[len(L) // 2]; stats["len_p95"] = L[int(len(L) * .95)]; stats["len_max"] = L[-1]
    stats["user_turns_mean"] = sum(nt) / len(nt)
    log(f"kept {stats['kept']}  tokens {stats['tokens_total']/1e6:.1f}M (trained {stats['tokens_trained']/1e6:.1f}M)")

    # ---- contamination check against the frozen question set
    blob = norm("\n".join(texts))
    qs = [json.loads(l) for l in open(a.questions)]
    expr_hits, ans_hits = [], []
    for q in qs:
        e = norm(q["expr"])
        if len(e) >= 6 and e in blob: expr_hits.append(q["id"] + " " + q["expr"])
        if not q["undefined"]:
            s = sig6(q["truth"])
            # only answers with >= 5 significant digits are informative (skip 0.5, 1, etc.)
            if s and len(re.sub(r"[^0-9]", "", s).lstrip("0")) >= 5 and s in blob:
                ans_hits.append(q["id"] + " " + q["expr"] + " = " + s)
    stats["contam_expr_hits"] = len(expr_hits); stats["contam_answer_hits"] = len(ans_hits)
    stats["contam_expr_examples"] = expr_hits[:40]; stats["contam_answer_examples"] = ans_hits[:40]
    # rough maths-keyword share, for the report only (not a filter)
    kw = re.compile(r"\b(sin|cos|tan|log|ln|sqrt|integral|derivative|equation|calculate|compute)\b|\d+\s*[\+\-\*/×÷\^]\s*\d+", re.I)
    stats["rows_with_math_keyword"] = sum(bool(kw.search(t)) for t in texts)
    json.dump(stats, open(f"{a.out}/data_report.json", "w"), indent=1)
    log("REPORT " + json.dumps({k: v for k, v in stats.items() if not k.endswith("examples")}))
    log("PREP DONE")

if __name__ == "__main__":
    main()
