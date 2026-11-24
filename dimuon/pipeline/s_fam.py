"""S1-S6 collection, by trunk 'family', so that only ~2 trunks' save_state checkpoints exist at a
time (the account's free checkpoint storage is capped at ~10 GB, ~6.9 GB of it the owner's;
see the S0 journal entry). Every job is idempotent: re-running this script only does what is
missing. A trunk whose checkpoints were already deleted is re-run (bit-identity checked) by
s_d2.ensure_ckpts, only if some job still needs it.

Usage: s_fam.py
"""
import s_d2 as d

SEQ = d.SEQ
CK = lambda T, k: f"{T}-ck{k}"


def one(text, params, sid, weight=1.0, note=""):
    return [dict(text=text, weight=weight, params=params, sid=sid, note=note)]


class Fam:
    """A trunk (name, rank, beta1, texts) plus jobs that branch from its checkpoints."""

    def __init__(self, T, r, beta1, texts, exp):
        self.T, self.r, self.b, self.texts, self.exp = T, r, beta1, texts, exp
        self.jobs = []          # (k, done_fn, fn)

    def sid(self, k):
        S = d.man()["steps"]
        key = f"{self.T}/step{max(k, 1)}"
        if key not in S:
            return None
        return S[key]["after"] if k > 0 else S[key]["before"]

    def oracle(self, k, text, weight=1.0):
        def done():
            s = self.sid(k)
            return s is not None and d.gkey(s, text, weight, d.SETTING) in d.man()["grads"]
        self.jobs.append((k, done, lambda: d.oracle(CK(self.T, k), text, weight)))

    def chain(self, k, name, steps, kind, exp):
        done = lambda: all(f"{name}/{st['sid']}" in d.man()["steps"] for st in steps)
        self.jobs.append((k, done, lambda: d.chain(name, self.r, steps, exp=exp, kind=kind,
                                                   ckpt=CK(self.T, k))))

    def todo(self):
        return [j for j in self.jobs if not j[1]()]

    def prepare(self):
        ks = sorted({k for k, _, _ in self.todo()})
        if ks:
            d.ensure_ckpts(self.T, self.r, self.b, self.texts, ks, self.exp)

    def run_jobs(self):
        return [fn for _, _, fn in self.todo()]


def fam_S1(r, T=None):
    T = T or f"S1_r{r}"
    f = Fam(T, r, 0.9, [0] * 5, "S1")
    for k in range(6):
        f.oracle(k, 0)
    for k in (1, 2, 5):
        f.oracle(k, 1)
        for t in (0, 1):
            f.chain(k, f"{T}_fresh_ck{k}_t{t}", one(t, d.D(), "step"), "fresh", "S1")
    return f


def fam_S2(r, b):
    f = Fam(f"S2_r{r}_b{b:g}", r, b, SEQ, "S2")
    for k in range(6):
        f.oracle(k, SEQ[k])
    if b == 0.0:   # control: fresh-state step from the same weights, same batch
        for k in range(1, 6):
            f.chain(k, f"{f.T}_fresh_ck{k}", one(SEQ[k], d.D(beta1=0.0), f"step{k + 1}"),
                    "fresh", "S2")
    if b == 0.9:
        r_ = r
        for bb in (0.0, 0.99, 0.9):     # S3
            f.chain(3, f"S3_r{r_}_b{bb:g}", one(SEQ[3], d.D(beta1=bb), "step4"), "opt", "S3")
        f.chain(2, f"S4_r{r_}_fresh_ck2", one(0, d.D(), "zero1", weight=0.0,
                                               note="zero-gradient"), "fresh", "S4")
        for lr in (1e-4, 1e-3, 1e-2):   # S6 (r8 lr1e-3 from ck2 = S0_opt_ck2)
            if not (r == 8 and lr == 1e-3):
                f.chain(2, f"S6_r{r_}_ck2_lr{lr:g}", one(SEQ[2], d.D(lr=lr), "step3"), "opt", "S6")
        if r == 8:
            for lr in (1e-4, 1e-2):     # lr1e-3 from ck3 = S3_r8_b0.9
                f.chain(3, f"S6_r8_ck3_lr{lr:g}", one(SEQ[3], d.D(lr=lr), "step4"), "opt", "S6")
            f.oracle(2, 0, weight=0.0)  # S4: is G = 0 ?
            # extra (S3 control): fresh-state beta1=0 step from state 3 on T2
            f.chain(3, "S3x_r8_fresh_ck3", one(SEQ[3], d.D(beta1=0.0), "step4"), "fresh", "S3")
    return f


if __name__ == "__main__":
    fams = [fam_S2(8, 0.9), fam_S1(8), fam_S1(32), fam_S2(8, 0.0), fam_S2(32, 0.9),
            fam_S2(8, 0.5), fam_S2(32, 0.0), fam_S2(8, 0.99)]
    print("todo:", {f.T: len(f.todo()) for f in fams}, flush=True)
    fams = [f for f in fams if f.todo()]
    if fams:
        d.run([fams[0].prepare])
    for i, f in enumerate(fams):
        jobs = f.run_jobs()
        if i + 1 < len(fams):
            jobs.append(fams[i + 1].prepare)
        print(f"=== family {f.T}: {len(jobs)} jobs", flush=True)
        d.run(jobs)
        d.drop_trunk(f.T)
        print("errors so far", sum(e["kind"] == "error" for e in d.man()["events"]),
              "COUNTS", d.tk.summary(), flush=True)
    print("todo after:", {f.T: len(f.todo()) for f in fams}, flush=True)
