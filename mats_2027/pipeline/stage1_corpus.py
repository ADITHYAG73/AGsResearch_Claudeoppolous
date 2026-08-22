"""Stage 1 — build the cricket passage corpus from Wikipedia, reproducibly.

One paragraph per article (independence). Paragraph chosen by seeded RNG per article.
Length gate uses the REAL Gemma tokenizer. Every row carries full provenance
(url + revision id) and `article_chars` as a fame/memorisation proxy.

Output: mats_2027/corpus/cricket_passages.jsonl  (one JSON object per line)
Run:    python mats_2027/pipeline/stage1_corpus.py
"""
import hashlib, json, os, random, re, sys, time, urllib.parse, urllib.request

SEED = 42
MIN_TOK, MAX_TOK = 60, 400          # Gemma tokens. 60 > _MIN_POSITION=50 with margin.
TARGET = 150
OBSCURE_FLOOR = 40                  # at least this many from the domestic/obscure buckets
OUT = "mats_2027/corpus/cricket_passages.jsonl"

CATEGORIES = {
    # bucket name        : (category title, cap)
    "test_cricketers"    : ("Category:India_Test_cricketers", 90),
    "grounds"            : ("Category:Test_cricket_grounds_in_India", 25),
    "domestic"           : ("Category:Indian_domestic_cricket_competitions", 25),
    "tamil_nadu"         : ("Category:Cricket_in_Tamil_Nadu", 12),
}
API = "https://en.wikipedia.org/w/api.php"
H = {"User-Agent": "nla-research/0.1 (MATS application research)"}
SKIP_TITLE = re.compile(r"^(List of|Category:|Template:|\d{4} )|\(disambiguation\)", re.I)

def q(params):
    params.update(format="json")
    url = API + "?" + urllib.parse.urlencode(params)
    delay = 2.0
    for attempt in range(8):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=30))
        except urllib.error.HTTPError as e:
            if e.code == 429:                       # rate limited: honour Retry-After, else backoff
                wait = float(e.headers.get("Retry-After", delay))
                print(f"    [429] backing off {wait:.0f}s", flush=True)
                time.sleep(wait); delay = min(delay * 2, 60)
                continue
            raise
        except Exception:
            if attempt == 7: raise
            time.sleep(delay); delay = min(delay * 2, 60)
    raise RuntimeError("gave up after 8 attempts")

def members(cat):
    out, cont = [], {}
    while True:
        r = q({"action":"query","list":"categorymembers","cmtitle":cat,"cmlimit":"500","cmnamespace":"0", **cont})
        out += [m["title"] for m in r["query"]["categorymembers"]]
        if "continue" not in r: break
        cont = r["continue"]
    return [t for t in out if not SKIP_TITLE.search(t)]

def article(title):
    r = q({"action":"query","prop":"extracts|revisions|info","explaintext":"1","rvprop":"ids",
           "inprop":"url","titles":title,"redirects":"1"})
    page = next(iter(r["query"]["pages"].values()))
    if "missing" in page or not page.get("extract"): return None
    return {"title": page["title"], "pageid": page["pageid"], "url": page["fullurl"],
            "revision": page["revisions"][0]["revid"], "text": page["extract"]}

def paragraphs(text):
    """Real prose paragraphs only: drop headings, stubs, reference-y lines."""
    out = []
    for i, p in enumerate(text.split("\n")):
        p = p.strip()
        if len(p) < 200 or p.startswith("==") or p.endswith(":"): continue
        if re.match(r"^(See also|References|External links|Notes)\b", p): continue
        out.append((i, p))
    return out

def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/gemma-3-12b-it")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows, seen_pages = [], set()
    if os.path.exists(OUT):                        # RESUME: never refetch what we have
        for line in open(OUT):
            r = json.loads(line); rows.append(r); seen_pages.add(r["pageid"])
        print(f"resuming with {len(rows)} passages already on disk")
    outf = open(OUT, "a")

    for bucket, (cat, cap) in CATEGORIES.items():
        titles = members(cat)
        rng = random.Random(hashlib.sha256(f"{SEED}|{cat}".encode()).digest())
        rng.shuffle(titles)
        got = sum(r["bucket"] == bucket for r in rows)
        print(f"\n[{bucket}] {len(titles)} candidate articles, cap {cap}, have {got}", flush=True)
        for t in titles:
            if got >= cap: break
            a = article(t)
            if not a or a["pageid"] in seen_pages: continue
            paras = paragraphs(a["text"])
            if not paras: continue
            # per-article RNG: same article -> same paragraph, regardless of run order
            arng = random.Random(hashlib.sha256(f"{SEED}|{a['pageid']}".encode()).digest())
            arng.shuffle(paras)
            for pidx, ptxt in paras:                      # first paragraph that passes the gate
                n = len(tok(ptxt)["input_ids"])
                if MIN_TOK <= n <= MAX_TOK:
                    seen_pages.add(a["pageid"]); got += 1
                    rows.append({
                        "doc_id": f"wiki:{a['pageid']}:rev{a['revision']}:p{pidx}",
                        "title": a["title"], "url": a["url"], "pageid": a["pageid"],
                        "revision": a["revision"], "bucket": bucket,
                        "article_chars": len(a["text"]),          # fame / memorisation proxy
                        "para_index": pidx, "text": ptxt,
                        "n_chars": len(ptxt), "n_tokens_gemma": n,
                    })
                    outf.write(json.dumps(rows[-1], ensure_ascii=False) + "\n"); outf.flush()
                    print(f"  + {a['title'][:40]:40s} {n:4d} tok  article={len(a['text']):7d} chars", flush=True)
                    break
            time.sleep(0.6)                         # ~1.5 req/s — polite

    outf.close()

    print(f"\n=== {len(rows)} passages -> {OUT} ===")
    from collections import Counter
    print("by bucket:", dict(Counter(r["bucket"] for r in rows)))
    toks = sorted(r["n_tokens_gemma"] for r in rows)
    print(f"tokens: min {toks[0]}  median {toks[len(toks)//2]}  max {toks[-1]}")
    ac = sorted(r["article_chars"] for r in rows)
    print(f"article_chars (fame proxy): min {ac[0]}  median {ac[len(ac)//2]}  max {ac[-1]}")
    obscure = sum(r["bucket"] in ("domestic","tamil_nadu","grounds") for r in rows)
    print(f"obscure-bucket passages: {obscure}  (floor {OBSCURE_FLOOR}: {'OK' if obscure>=OBSCURE_FLOOR else 'SHORT'})")

if __name__ == "__main__":
    main()
