"""Turn DRAFT.md into text that survives Google Docs' markdown import intact.

Two things break on import and both are fixed here:
  1. UNDERSCORES. Docs escapes them, so `injection_scale` arrives as `injection\\_scale`.
     Every identifier is rewritten without underscores.
  2. INLINE MARKDOWN INSIDE TABLE CELLS. Bold markers inside a cell come through literally,
     so all emphasis is stripped from table rows (Docs bolds the header row itself).
Figure embeds become explicit insert markers, since local paths cannot resolve in a Doc.
"""
import re, sys

UNDERSCORED = {
    "injection_scale": "injection scale",
    "mse_scale": "mse scale",
    "_MIN_POSITION": "MIN POSITION",
    "MIN_POSITION": "MIN POSITION",
    "stage0_extract.py": "stage0-extract.py",
    "decompose_response": "decompose-response",
    "verify_response": "verify-response",
    "vibe_response": "vibe-response",
    "match_response": "match-response",
    "*_prompt": "prompt",
    "observed_delta": "Δ(observed)",
    "underlying_delta": "Δ(underlying)",
    "mats_2027/writeup/figures/": "mats-2027/writeup/figures/",
    "noise_fit.py": "noise-fit.py",
    "final_token_control.py": "final-token-control.py",
    "patch_control.py": "patch-control.py",
    "sample_stratified.py": "sample-stratified.py",
}

def main(src="mats_2027/writeup/DRAFT.md", dst="/tmp/DRAFT_docs.md"):
    s = open(src).read()
    s = re.sub(r"!\[([A-Za-z0-9_]+)\]\([^)]+\)",
               lambda m: "INSERT FIGURE IMAGE: " + m.group(1).replace("_", " "), s)
    for a, b in UNDERSCORED.items():
        s = s.replace(a, b)
    out = []
    for line in s.split("\n"):
        if line.strip().startswith("|"):
            line = line.replace("**", "").replace("*", "")
        out.append(line)
    s = "\n".join(out)
    # unwrap hard-wrapped paragraphs: inside a block of plain prose, join the lines.
    blocks, joined = s.split("\n\n"), []
    for b in blocks:
        lines = b.split("\n")
        if (len(lines) > 1 and not b.lstrip().startswith(("|", "#", "- ", "* "))
                and not any(re.match(r"\s*\d+\.\s", l) for l in lines)):
            b = " ".join(l.strip() for l in lines)
        joined.append(b)
    s = "\n\n".join(joined)
    # unwrap list items too: a line that does not start a new item continues the previous one
    fixed = []
    for line in s.split("\n"):
        starts_item = re.match(r"\s*(?:[-*]\s|\d+\.\s)", line)
        if (fixed and line.strip() and not starts_item and not line.startswith(("|", "#"))
                and re.match(r"\s*(?:[-*]\s|\d+\.\s)", fixed[-1] or "")):
            fixed[-1] = fixed[-1].rstrip() + " " + line.strip()
        else:
            fixed.append(line)
    s = "\n".join(fixed)
    s = re.sub(r"[ ]{2,}", " ", s)
    left = [w for w in re.findall(r"\S*_\S*", s)]
    assert not left, f"underscores still present: {left[:8]}"
    open(dst, "w").write(s)
    print(f"wrote {dst}: {len(s.split())} words, {len(s)} chars, 0 underscores")

if __name__ == "__main__":
    main(*sys.argv[1:])
