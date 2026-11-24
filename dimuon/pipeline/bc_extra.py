"""bc_extra.py -- BIASCORR-01: extra rank-1 oracles until one reproduces the gradient version seen by
r1_b0 / b0.5 / b0.75 / b0.9 (which neither r1_oracle1 nor r1_oracle2 matched). At most N tries, sequential.
Match = median relF(Horner emulation of g, r1_b0 data) < 1e-6."""
import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bc_collect as C, bc_analyze as A, bc_lib as L

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    C.tk.service(); C.tk.rest()
    U = A.ld("r1_b0", "U")
    for i in range(3, 3 + N):
        name = f"r1_oracle{i}"
        C.job(dict(name=name, kind="oracle", rank=1))
        G = A.ld(name, "G")
        e = float(np.median([np.linalg.norm(L.predict(G[k], 0, "g")[1] - U[k]) / np.linalg.norm(U[k]) for k in G]))
        print(name, "emulation vs r1_b0 median relF", e, flush=True)
        if e < 1e-6:
            break
    print("COUNTS", C.tk.summary())
