"""Fix: S1 trunks saved ck0..ck4 only (save_last was off), so the state-5 branches failed.
Re-run T0 x5 in a new client, save only the final state, check it is bit-identical to S1 state 5,
alias it as S1_r{r}-ck5, then run the S1 state-5 jobs. Usage: s_fix_ck5.py <rank>"""
import sys
import s_d2 as d
import s_fam as f
r = int(sys.argv[1])
d.chain(f"S1x_r{r}", r, d.seq_steps(0.9, texts=[0] * 5), exp="S1-fix", save_last=True)
m = d.man()
c = m["ckpts"][f"S1x_r{r}-ck5"]
ok = c["sid"] == m["steps"][f"S1_r{r}/step5"]["after"]
print("re-run state 5 bit-identical to S1 trunk state 5:", ok, flush=True)
assert ok
m["ckpts"][f"S1_r{r}-ck5"] = dict(c, alias_of=f"S1x_r{r}-ck5")
d.save_man()
T = f"S1_r{r}"
jobs = [lambda: d.oracle(f.CK(T, 5), 0), lambda: d.oracle(f.CK(T, 5), 1)] + [
    lambda t=t: d.chain(f"S1_r{r}_fresh_ck5_t{t}", r, f.one(t, d.D(), "step"), exp="S1",
                        kind="fresh", ckpt=f.CK(T, 5)) for t in (0, 1)]
d.run(jobs)
d.drop_ckpts([f"S1x_r{r}-ck5"])
print("COUNTS", d.tk.summary())
