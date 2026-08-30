"""D5 control: is the THEME->DETAIL Delta gradient driven by claims that NAME THE FINAL TOKEN?

Motivation. The AR reconstructs the activation AT THE FINAL TOKEN of the prefix. A claim that
quotes or names that token is therefore load-bearing for a reason that has nothing to do with
specificity or redundancy. If such claims are concentrated in DETAIL, they could manufacture the
DETAIL > THEME gradient reported in SCORE-01.

Flag (deliberately conservative, stated so it can be argued with):
  the last CONTENT word of the prefix (regex [A-Za-z0-9-]+, >=3 chars, not a stopword) occurs in
  the claim text on a word boundary. Stopwords and 1-2 char tokens are excluded because 13 of the
  60 prefixes end in "in"/"the"/"of"/"a"/"as"/"to"/"and", which would match almost any claim.

This is a heuristic, not a labelled category. It cannot distinguish "names the final token" from
"happens to use the same word".
"""
import re, argparse, collections
import numpy as np, pyarrow.parquet as pq

STOP = set("the a an of in on at to and or is was were be been for with by as it its his her "
           "their this that from into during had has have not but which who".split())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas", default="mats_2027/runs/2026-08-23_ar/deltas.parquet")
    ap.add_argument("--acts", default="mats_2027/runs/2026-08-22_pos10/activations.parquet")
    a = ap.parse_args()
    d = pq.read_table(a.deltas).to_pandas(); d = d[d.valid == True]
    A = {(r["doc_id"], r["pos"]): r["detokenized_text_truncated"]
         for r in pq.read_table(a.acts).to_pylist()}

    def last_content_word(doc, pos):
        toks = re.findall(r"[A-Za-z0-9-]+", A[(doc, pos)].rstrip())
        return toks[-1] if toks else ""

    d["lw"] = [last_content_word(r.doc_id, r.pos) for r in d.itertuples()]
    def flag(r):
        lw = r.lw.lower()
        if len(lw) < 3 or lw in STOP: return False
        return re.search(r"\b" + re.escape(lw) + r"\b", str(r.claim).lower()) is not None
    d["names_final"] = [flag(r) for r in d.itertuples()]
    assert d.names_final.sum() > 0, "flag never fires - check the regex"

    def ci(x): return 1.96 * x.std(ddof=1) / len(x) ** 0.5
    print(f"prefixes ending in a stopword/short token: "
          f"{sum(1 for v in d.groupby(['doc_id','pos']).lw.first() if len(v)<3 or v.lower() in STOP)} of 60")
    print(f"flagged: {d.names_final.sum()} of {len(d)} ({100*d.names_final.mean():.1f}%)\n")
    for tag, sub in (("ALL CLAIMS", d), ("FLAGGED (names final token)", d[d.names_final]),
                     ("UNFLAGGED", d[~d.names_final])):
        print(tag)
        for lv in ("THEME", "ENTITY", "DETAIL"):
            g = sub[sub.level == lv]
            print(f"  {lv:<7} n={len(g):4d}  mean {g.delta.mean():+.5f}  ci+-{ci(g.delta):.5f}")
        t, dd = sub[sub.level=="THEME"].delta.mean(), sub[sub.level=="DETAIL"].delta.mean()
        print(f"  DETAIL / THEME = {dd/t:.1f}x\n")

if __name__ == "__main__":
    main()
