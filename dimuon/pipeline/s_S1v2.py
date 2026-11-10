"""Re-collect the S1 rank-32 family entirely in regime R2 (trunk S1v2_r32), because the original
S1_r32 family straddles the server change (DRIFT-01). Then an R2 canary (s_drift-style)."""
import s_d2 as d
import s_fam as f
d.SETTING = "O1r2"
fam = f.fam_S1(32, T="S1v2_r32")
d.run([fam.prepare])
d.run(fam.run_jobs())
d.drop_trunk(fam.T)
print("todo after:", len(fam.todo()), "COUNTS", d.tk.summary(), flush=True)
