#!/usr/bin/env python3
"""LATENT-CUBE prompt set with exact ground truth (Fractions, no recall).

"k-th term" = k-th NON-ZERO term of the Maclaurin series (AG's reading: 2nd term of sin x = -x^3/6).
Three readout templates per item (AG decision 1: record all three):
  write  : asks for the term; last prompt token = "about to write it"; also the behavioural check
  imagine: asks to imagine it, not write it
  hold   : imagine it, then answer an unrelated question (capital of France) -> read while it says Paris
Plus a free-form latent set (cube volume, Rubik's 3^3, ...) and controls.
Writes prompts.jsonl.
"""
import json, sys
from fractions import Fraction as F
from math import factorial

N = 12  # coefficients x^0..x^11

def mul(a, b): return [sum(a[i] * b[k - i] for i in range(k + 1)) for k in range(N)]
def div(a, b):  # power-series a/b, b[0] != 0
    q = [F(0)] * N
    for k in range(N):
        q[k] = (a[k] - sum(q[i] * b[k - i] for i in range(k))) / b[0]
    return q
def scale(a, c): return [a[k] * c ** k for k in range(N)]  # f(c x)

SIN = [F((-1) ** ((k - 1) // 2), factorial(k)) if k % 2 else F(0) for k in range(N)]
COS = [F((-1) ** (k // 2), factorial(k)) if k % 2 == 0 else F(0) for k in range(N)]
EXP = [F(1, factorial(k)) for k in range(N)]
SINH = [F(1, factorial(k)) if k % 2 else F(0) for k in range(N)]
COSH = [F(1, factorial(k)) if k % 2 == 0 else F(0) for k in range(N)]
LN1P = [F(0)] + [F((-1) ** (k + 1), k) for k in range(1, N)]
ATAN = [F((-1) ** ((k - 1) // 2), k) if k % 2 else F(0) for k in range(N)]
GEOM = [F(1)] * N  # 1/(1-x)

FUNCS = {  # name in prompt -> (series, group)
    "sin x": (SIN, "base"), "cos x": (COS, "base"), "e^x": (EXP, "base"),
    "sinh x": (SINH, "base"), "cosh x": (COSH, "base"), "tan x": (div(SIN, COS), "base"),
    "arctan x": (ATAN, "base"), "ln(1+x)": (LN1P, "base"), "1/(1-x)": (GEOM, "base"),
    "sin(2x)": (scale(SIN, 2), "composed"), "sin(x/2)": (scale(SIN, F(1, 2)), "composed"),
    "cos(2x)": (scale(COS, 2), "composed"), "e^(2x)": (scale(EXP, 2), "composed"),
    "sin(x)·e^x": (mul(SIN, EXP), "composed"), "sin²x": (mul(SIN, SIN), "composed"),
    "x·sin x": ([F(0)] + SIN[:-1], "composed"),
}
ORD = {1: "first", 2: "second", 3: "third", 4: "fourth"}

def nonzero_terms(s): return [(k, s[k]) for k in range(N) if s[k] != 0]
def fmt(k, c):
    sign = "-" if c < 0 else ""
    c = abs(c); xs = "" if k == 0 else ("x" if k == 1 else f"x^{k}")
    if k == 0: return f"{sign}{c}"
    num = "" if c.numerator == 1 else str(c.numerator)
    t = f"{num}{xs}"
    return f"{sign}{t}" if c.denominator == 1 else f"{sign}{t}/{c.denominator}"

TPL = {
    "write":   "What is the {o} term of the Taylor series of {f} about x = 0? Reply with the term only.",
    "imagine": "Imagine the {o} term of the Taylor series of {f} about x = 0. Do not write it.",
    "hold":    "Imagine the {o} term of the Taylor series of {f} about x = 0, and keep it in mind. Now answer: what is the capital of France?",
}

rows = []
for fname, (s, group) in FUNCS.items():
    nz = nonzero_terms(s)
    for kth in range(1, 5):
        k, c = nz[kth - 1]
        for tpl, t in TPL.items():
            rows.append({"id": f"tay|{fname}|{kth}|{tpl}", "set": "taylor", "group": group, "func": fname,
                         "kth": kth, "template": tpl, "prompt": t.format(o=ORD[kth], f=fname),
                         "power": k, "coef": str(c), "answer": fmt(k, c)})

LATENT = [  # free-form: (id, prompt, power of x involved, what is latent)
    ("cube_volume", "Imagine the volume of a cube whose side is x. Do not write it.", 3, "x^3"),
    ("integral_x2", "Imagine the antiderivative of x^2. Do not write it.", 3, "x^3/3"),
    ("rubik", "Imagine how many small cubes make up a 3×3×3 Rubik's cube. Do not write it.", 3, "27 = 3^3"),
    ("choose3", "Imagine the leading term, for large x, of the number of ways to choose 3 items from x. Do not write it.", 3, "x^3/6"),
    ("square_area", "Imagine the area of a square whose side is x. Do not write it.", 2, "x^2"),
    ("sphere_vol", "Imagine the volume of a sphere of radius x. Do not write it.", 3, "4πx^3/3"),
    ("pizza_sixth", "Imagine one sixth of a pizza. Do not write it.", 0, "1/6, no power"),
    ("two_cubed", "Imagine 2 cubed. Do not write it.", 3, "8"),
    ("neutral", "Imagine a quiet beach at sunset. Do not write it.", 0, "control"),
]
for lid, p, pw, what in LATENT:
    rows.append({"id": f"lat|{lid}", "set": "latent", "group": "latent", "template": "imagine",
                 "prompt": p, "power": pw, "answer": what})

out = sys.argv[1] if len(sys.argv) > 1 else "prompts.jsonl"
with open(out, "w") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(rows), "prompts ->", out)
