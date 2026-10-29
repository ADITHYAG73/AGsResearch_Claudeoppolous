#!/usr/bin/env python3
"""Build the frozen question set for the post-training maths project.

Every question is an expression tree. The tree is rendered three ways from ONE definition, so the
text the model reads, the reference value, and the independent check can never drift apart:

  disp(node)      the text shown to the model, e.g.  sin(cos(40°))
  ev(node, MP)    reference value with mpmath at 40 digits
  ev(node, FL)    the same tree evaluated with Python's own `math` in double precision

The two evaluators share no arithmetic code, and every question must agree between them to a
relative 1e-9 or it is dropped and reported. That is the check on the answer key.

ANGLE RULE (stated in every prompt, enforced here): a number written with a degree sign is in
degrees; any other argument of a trigonometric function is in radians. So sin(cos(40°)) means
"cosine of forty degrees, then the sine of that number taken in radians".

The file is prompt-agnostic and scoring-agnostic: truth is stored to 17 significant digits, so the
scoring rule (decimal places or significant figures) can be chosen without regenerating.
"""
import argparse, hashlib, json, math, os, random
import mpmath as mp

mp.mp.dps = 40
SEED = 20261006

# ----------------------------------------------------------------- expression trees
def num(s):            return ("num", str(s))
def deg(s):            return ("deg", str(s))            # leaf: an angle in degrees
def f(name, a):        return ("f", name, a)
def op(sym, a, b):     return ("op", sym, a, b)

NAMES = {"sin": "sin", "cos": "cos", "tan": "tan", "exp": "e^", "ln": "ln", "log10": "log10",
         "sqrt": "sqrt", "asind": "arcsin", "acosd": "arccos", "atand": "arctan"}

def disp(n, top=True):
    k = n[0]
    if k == "num":
        s = n[1]
        return s if (top or not s.startswith("-")) else f"({s})"
    if k == "deg":
        return f"{n[1]}°"
    if k == "f":
        return f"{NAMES[n[1]]}({disp(n[2])})"
    if k == "op":
        sym, a, b = n[1], n[2], n[3]
        la = disp(a, False) if a[0] != "op" else f"({disp(a)})"
        lb = disp(b, False) if b[0] != "op" else f"({disp(b)})"
        return f"{la}^{lb}" if sym == "^" else f"{la} {sym} {lb}"
    raise ValueError(n)

class MP:                                   # reference arithmetic
    num = staticmethod(lambda s: mp.mpf(s))
    rad = staticmethod(lambda x: x * mp.pi / 180)
    todeg = staticmethod(lambda x: x * 180 / mp.pi)
    fn = {"sin": mp.sin, "cos": mp.cos, "tan": mp.tan, "exp": mp.exp, "ln": mp.log,
          "log10": mp.log10, "sqrt": mp.sqrt, "asin": mp.asin, "acos": mp.acos, "atan": mp.atan}
    pow = staticmethod(lambda a, b: mp.power(a, b))

class FL:                                   # independent check: plain doubles
    num = staticmethod(lambda s: float(s))
    rad = staticmethod(math.radians)
    todeg = staticmethod(math.degrees)
    fn = {"sin": math.sin, "cos": math.cos, "tan": math.tan, "exp": math.exp, "ln": math.log,
          "log10": math.log10, "sqrt": math.sqrt, "asin": math.asin, "acos": math.acos,
          "atan": math.atan}
    pow = staticmethod(lambda a, b: a ** b)

def ev(n, L):
    k = n[0]
    if k == "num": return L.num(n[1])
    if k == "deg": return L.rad(L.num(n[1]))
    if k == "f":
        name, x = n[1], ev(n[2], L)
        if name in ("asind", "acosd", "atand"):
            return L.todeg(L.fn[name[:-1]](x))
        return L.fn[name](x)
    if k == "op":
        a, b = ev(n[2], L), ev(n[3], L)
        return {"+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b,
                "/": lambda: a / b, "^": lambda: L.pow(a, b)}[n[1]]()
    raise ValueError(n)

def depth(n):
    if n[0] in ("num", "deg"): return 0
    if n[0] == "f": return 1 + depth(n[2])
    return max(depth(n[2]), depth(n[3]))

def fmt(x, nd=2):
    """a decimal string with no trailing zeros; integers print as integers"""
    s = f"{x:.{nd}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s

# ----------------------------------------------------------------- families
Q = []
def add(family, sub, node, arg_class, **extra):
    Q.append(dict(family=family, sub=sub, node=node, arg_class=arg_class, **extra))

def build():
    rng = random.Random(SEED)

    # A. trig at integer degrees: the TRIG-01 sweep, unchanged, so old data stays comparable
    for d in range(360):
        cls = "landmark" if d % 90 in (0, 30, 45, 60) else ("mult5" if d % 5 == 0 else "integer")
        for fn in ("sin", "cos", "tan"):
            if fn == "tan" and d % 180 == 90: continue      # poles live in the "undefined" family
            add("trig_int", fn, f(fn, deg(d)), cls, needs_reduction=d >= 90)

    # B. trig variants: SAME generator and seed as TRIG-01b, so the angles are identical
    r2 = random.Random(20260922)
    for d in range(90):
        ab = r2.randrange(1, 100) / 100.0
        for name, ang, cls, red in (("half", fmt(d + 0.5), "fractional", False),
                                    ("dec2", fmt(round(d + ab, 2)), "fractional", False),
                                    ("over360", fmt(d + 360), "integer", True),
                                    ("neg", fmt(-d), "integer", True)):
            for fn in ("sin", "cos", "tan"):
                add("trig_var", f"{name}_{fn}", f(fn, deg(ang)), cls, needs_reduction=red)

    # C. transcendental functions of one number
    def grid(fn, ints, lo, hi, nrand, nd=2, sub=None):
        for i in ints:
            add("transcendental", sub or fn, f(fn, num(i)), "integer")
        for _ in range(nrand):
            add("transcendental", sub or fn, f(fn, num(fmt(rng.uniform(lo, hi), nd))), "fractional")
    grid("exp",   range(-5, 11), -5, 10, 45)
    for h in range(-9, 20, 2):
        add("transcendental", "exp", f("exp", num(fmt(h / 2, 1))), "fractional")
    grid("ln",    range(1, 51), 0.05, 100, 45)
    grid("log10", range(1, 51), 0.05, 100, 45)
    grid("sqrt",  range(1, 51), 0.05, 200, 45)
    for i in range(-10, 11):
        x = fmt(i / 10, 1)
        add("transcendental", "asin", f("asind", num(x)), "landmark" if i in (-10, -5, 0, 5, 10) else "fractional", answer_unit="degrees")
        add("transcendental", "acos", f("acosd", num(x)), "landmark" if i in (-10, -5, 0, 5, 10) else "fractional", answer_unit="degrees")
    for i in list(range(-10, 11)) + [fmt(rng.uniform(-10, 10), 2) for _ in range(20)]:
        add("transcendental", "atan", f("atand", num(i)), "integer" if isinstance(i, int) else "fractional", answer_unit="degrees")
    for a, b in ((2, "0.5"), (2, 10), (10, "0.3"), (3, "0.5"), (2, -1), (5, "1.5"), (10, -2), (2, "0.1")):
        add("transcendental", "pow", op("^", num(a), num(b)), "landmark")
    for _ in range(60):
        a, b = fmt(rng.uniform(1.1, 9.9), 1), fmt(rng.choice([-1, 1]) * rng.uniform(0.2, 3.0), 1)
        add("transcendental", "pow", op("^", num(a), num(b)), "fractional")

    # D. composite: two DIFFERENT functions, f(g(x))
    A40 = [d for d in range(5, 176, 9)]                      # 20 angles, 5..176
    A80 = [d for d in range(-80, 81, 8)]                     # 21 angles, cos > 0
    R = [fmt(x, 1) for x in (0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6, 2.9, 3.3, 3.7, 4.2, 4.8, 5.5, 6.3, 7.1, 8.2, 9.4, 10.5)]
    S = [fmt(x, 1) for x in (-2, -1.5, -1, -0.5, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4)]
    comp = [("exp_sin",  lambda a: f("exp",  f("sin", deg(a))), A40),
            ("exp_cos",  lambda a: f("exp",  f("cos", deg(a))), A40),
            ("ln_sin",   lambda a: f("ln",   f("sin", deg(a))), A40),
            ("sqrt_cos", lambda a: f("sqrt", f("cos", deg(a))), A80),
            ("sin_cos",  lambda a: f("sin",  f("cos", deg(a))), A40),
            ("cos_sin",  lambda a: f("cos",  f("sin", deg(a))), A40),
            ("tan_sin",  lambda a: f("tan",  f("sin", deg(a))), A40),
            ("sin_exp",  lambda x: f("sin",  f("exp", num(x))), S),
            ("cos_ln",   lambda x: f("cos",  f("ln",  num(x))), R),
            ("sin_sqrt", lambda x: f("sin",  f("sqrt", num(x))), R),
            ("exp_sqrt", lambda x: f("exp",  f("sqrt", num(x))), R),
            ("ln_sqrt",  lambda x: f("ln",   f("sqrt", num(x))), R),
            ("sqrt_exp", lambda x: f("sqrt", f("exp", num(x))), S),
            ("log10_exp", lambda x: f("log10", f("exp", num(x))), S)]
    for sub, mk, xs in comp:
        for x in xs:
            add("composite", sub, mk(x), "fractional")
    # a small control: composites that collapse to x if the model simplifies
    for x in R[:12]:
        add("composite", "identity_ln_exp", f("ln", f("exp", num(x))), "simplifiable")
        add("composite", "identity_exp_ln", f("exp", f("ln", num(x))), "simplifiable")

    # E. nested: the SAME function applied 2 or 3 times
    def nest(fn, leaf, k):
        n = leaf
        for _ in range(k): n = f(fn, n)
        return n
    for a in A40:
        for fn in ("sin", "cos"):
            for k in (2, 3):
                add("nested", f"{fn}x{k}_deg", nest(fn, deg(a), k), "fractional")
    for x in R[:14]:
        for fn in ("sin", "cos"):
            for k in (2, 3):
                add("nested", f"{fn}x{k}_rad", nest(fn, num(x), k), "fractional")
    for x in (-2, -1.5, -1, -0.5, 0, 0.2, 0.4, 0.6, 0.8, 1):
        add("nested", "expx2", nest("exp", num(fmt(x, 1)), 2), "fractional")
    for x in (3, 4, 5, 8, 10, 20, 50, 100, 500, 1000):
        add("nested", "lnx2", nest("ln", num(x), 2), "integer")
    for x in (2, 3, 5, 7, 10, 16, 50, 81, 100, 1000):
        for k in (2, 3):
            add("nested", f"sqrtx{k}", nest("sqrt", num(x), k), "integer")

    # F. mixed expressions: several functions joined by arithmetic
    def atom():
        c = rng.choice(["sind", "cosd", "exp", "ln", "sqrt", "log10"])
        if c == "sind": return f("sin", deg(rng.randrange(1, 180)))
        if c == "cosd": return f("cos", deg(rng.randrange(1, 90)))
        if c == "exp":  return f("exp", num(fmt(rng.uniform(-2, 3), 1)))
        if c == "ln":   return f("ln", num(fmt(rng.uniform(1.5, 50), 1)))
        if c == "sqrt": return f("sqrt", num(fmt(rng.uniform(1.5, 90), 1)))
        return f("log10", num(fmt(rng.uniform(1.5, 500), 1)))
    for i in range(240):
        t = i % 4
        if t == 0:   n, sub = op("+", atom(), atom()), "sum2"
        elif t == 1: n, sub = op("*", atom(), atom()), "prod2"
        elif t == 2: n, sub = op("/", atom(), atom()), "quot2"
        else:        n, sub = op("-", op("+", atom(), op("*", num(rng.randrange(2, 10)), atom())), atom()), "three_term"
        add("mixed", sub, n, "fractional")

    # G. undefined over the reals: there is no number to give
    for n, why in ((f("tan", deg(90)), "pole"), (f("tan", deg(270)), "pole"), (f("tan", deg(450)), "pole"),
                   (f("ln", num(0)), "log of zero"), (f("ln", num(-1)), "log of negative"),
                   (f("log10", num(0)), "log of zero"), (f("log10", num(-5)), "log of negative"),
                   (f("sqrt", num(-4)), "sqrt of negative"), (f("sqrt", num(-2)), "sqrt of negative"),
                   (f("asind", num(2)), "arcsin out of range"), (f("acosd", num(-1.5)), "arccos out of range"),
                   (op("/", num(1), f("sin", deg(0))), "division by zero"),
                   (op("/", num(1), f("ln", num(1))), "division by zero"),
                   (f("ln", f("sin", deg(180))), "log of zero"), (f("ln", f("cos", deg(120))), "log of negative"),
                   (f("sqrt", f("sin", deg(200))), "sqrt of negative")):
        add("undefined", why, n, "undefined", undefined=True)

# ----------------------------------------------------------------- evaluate, verify, freeze
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    ap.add_argument("--core-per-sub", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    build()

    rows, dropped, seen = [], [], set()
    for q in Q:
        text = disp(q["node"])
        if text in seen: continue                      # e.g. sin(0°) appears as -0 too
        seen.add(text)
        rec = dict(family=q["family"], sub=q["sub"], expr=text, arg_class=q["arg_class"],
                   depth=depth(q["node"]), needs_reduction=bool(q.get("needs_reduction", False)),
                   answer_unit=q.get("answer_unit"), undefined=bool(q.get("undefined", False)))
        if rec["undefined"]:
            rec["truth"] = None
            rows.append(rec); continue
        try:
            vm = ev(q["node"], MP); vf = ev(q["node"], FL)
        except Exception as e:
            dropped.append((text, f"error {type(e).__name__}")); continue
        if isinstance(vm, mp.mpc) or isinstance(vf, complex):
            dropped.append((text, "complex")); continue
        vmf = float(vm)
        if not math.isfinite(vmf) or (abs(vmf) > 1e6):
            dropped.append((text, f"out of range {vmf:.3g}")); continue
        tol = 1e-9 * max(1.0, abs(vmf))
        if abs(vmf - vf) > tol:
            # near a pole or a zero the two libraries legitimately part ways; such items are unfair
            dropped.append((text, f"evaluators disagree {vmf!r} vs {vf!r}")); continue
        if q["family"] == "trig_int" and abs(vmf) > 1e5:
            dropped.append((text, "pole")); continue
        rec["truth"] = mp.nstr(vm, 17)
        rec["is_zero"] = abs(vmf) < 1e-12
        rows.append(rec)

    # stable ids, and a seeded core subset for the expensive conditions
    rng = random.Random(SEED + 1)
    by = {}
    for i, r in enumerate(rows):
        r["id"] = f"q{i:05d}"
        by.setdefault((r["family"], r["sub"]), []).append(r)
    for key, grp in by.items():
        pick = set(id(x) for x in rng.sample(grp, min(a.core_per_sub, len(grp))))
        for r in grp: r["core"] = id(r) in pick

    path = os.path.join(a.out, "questions_v0.jsonl")
    with open(path, "w") as fh:
        for r in rows: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    json.dump({"file": "questions_v0.jsonl", "sha256": h, "n": len(rows), "seed": SEED,
               "n_core": sum(r["core"] for r in rows), "dropped": len(dropped)},
              open(os.path.join(a.out, "questions_v0.meta.json"), "w"), indent=1)

    print(f"wrote {len(rows)} questions -> {path}")
    print(f"sha256 {h}")
    print(f"core subset: {sum(r['core'] for r in rows)}   dropped: {len(dropped)}")
    fam = {}
    for r in rows: fam.setdefault(r["family"], []).append(r)
    print(f"\n{'family':<16}{'n':>6}{'core':>6}{'subfamilies':>13}   depth range")
    for k, g in fam.items():
        ds = [x["depth"] for x in g]
        print(f"{k:<16}{len(g):>6}{sum(x['core'] for x in g):>6}{len(set(x['sub'] for x in g)):>13}   {min(ds)}-{max(ds)}")
    if dropped:
        print("\ndropped (first 12):")
        for t, why in dropped[:12]: print(f"   {t:<34} {why}")

if __name__ == "__main__":
    main()
