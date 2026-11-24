"""Smoke: S2 rank-8 beta1=0.9 trunk (also the S0 trunk) + one O1 oracle at its state 2."""
import s_d2 as d
d.chain("S2_r8_b0.9", 8, d.seq_steps(0.9), exp="S2", save=True)
print(d.oracle("S2_r8_b0.9-ck2", text=1, setting="O1"))
print(d.tk.summary())
