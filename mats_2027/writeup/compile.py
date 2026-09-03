"""Assemble the write-up from ag_drafts/ into one document.

Deterministic and re-runnable: edit a section file, re-run, get the updated document.
What this does: orders the sections, strips Claude's provenance blocks and the author's own
`<...>` / `<<...>>` notes-to-self, inserts figure references with captions, and reports word
counts per section. What it does NOT do: rewrite or re-voice AG's prose.
"""
import re, os, textwrap

D = "mats_2027/writeup/ag_drafts/"
FIG = "mats_2027/writeup/figures/"

ORDER = [
    ("A. Executive summary",              "secA.md"),
    ("Randomly selected examples",        "secEX.md"),
    ("B. The question, and why it matters","secB.md"),
    ("C. Setup",                          "secC.md"),
    ("D1. Specificity replicates",        "secD1.md"),
    ("D2. What Δ does and does not tell you","secD2.md"),
    ("D3. H1 — one population, not two",  "secD3.md"),
    ("D4. H2 — the noise is real and random","secD4.md"),
    ("D5. Removal improves reconstruction 30% of the time","secD5.md"),
    ("D6. The AV imports names it knows", "secD6.md"),
    ("E. What I verified myself",         "secE.md"),
    ("F. Limitations",                    "secF.md"),
    ("F2. Reflections",                   "secF2.md"),
    ("Appendix. One activation, end to end", "secAPP.md"),
]

# figure -> the section it should sit in
FIGS = {
    "secA.md":  ["G4_h2_spread", "G5_savarkar"],
    "secC.md":  ["G0_pipeline"],
    "secD1.md": ["G1_specificity"],
    "secD3.md": ["G2_h1_null", "G3_dip_power"],
    "secD4.md": ["G4_h2_spread"],
    "secD6.md": ["G5_savarkar"],
}

def clean(t):
    # Claude provenance blocks and the FACTS list, which are notes, not content
    t = re.split(r"\n-{3,}\nNOTE \(Claude", t)[0]
    t = re.split(r"\n-{3,}\nFACTS NOT YET IN THIS SECTION", t)[0]
    t = re.sub(r"^#\s*D\d SCAFFOLD.*?\n(#.*\n)*", "", t)          # any stray scaffold header
    t = re.sub(r"<<[^>]*>>", "", t)                                # <<AG: ...>> markers
    t = re.sub(r"^\s*<[^>\n]{15,}>\s*$", "", t, flags=re.M)        # standalone <notes to self>
    t = re.sub(r"\([^)]*\?\?[^)]*\)", "", t)                       # (question to claude??)
    t = re.sub(r"<[^>\n]*\?\?[^>\n]*>", "", t)                       # <examples??>
    t = re.sub(r"<same as [^>\n]*>", "", t)                          # <same as case 1>
    t = re.sub(r"\(link to neel[^)]*\)", "", t, flags=re.I)          # unresolved link note
    t = re.sub(r"\bruled out\.wrote\b", "ruled out.", t)
    t = re.sub(r"\.wrote\b", ".", t)                                 # stray paste artefact
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

out, report = [], []
for title, fn in ORDER:
    p = os.path.join(D, fn)
    body = clean(open(p).read()) if os.path.exists(p) else ""
    words = len(body.split())
    if not body:
        body = f"*[{title} — not yet written]*"
    # drop a duplicated in-file heading, we supply our own
    body = re.sub(r"^#+\s*" + re.escape(title.split(".")[0]) + r"[^\n]*\n+", "", body)
    body = re.sub(r"^#+\s*[^\n]*\n+", "", body, count=1) if body.startswith("#") else body
    figs = "\n\n".join(
        f"**Figure {n.split('_')[0]}.** {open(FIG + n + '.caption.txt').read().strip()}\n\n"
        f"![{n}]({FIG}{n}.png)"
        for n in FIGS.get(fn, []) if os.path.exists(FIG + n + ".caption.txt"))
    out.append(f"## {title}\n\n{body}\n" + (f"\n{figs}\n" if figs else ""))
    report.append((title, words, len(FIGS.get(fn, []))))

doc = ("# Is the NLA activation reconstructor a weak per-claim verifier because the signal is "
       "absent, or because it is buried in noise?\n\n"
       "*Adithya Giridharan · MATS 12.0 application task · September 2026*\n\n"
       "---\n\n" + "\n---\n\n".join(out))
open("mats_2027/writeup/DRAFT.md", "w").write(doc)

total = sum(w for _, w, _ in report)
print(f"{'section':<48}{'words':>7}{'figs':>6}")
for t, w, f in report: print(f"{t:<48}{w:>7}{f:>6}")
print(f"{'TOTAL':<48}{total:>7}")
print("\nwrote mats_2027/writeup/DRAFT.md")
