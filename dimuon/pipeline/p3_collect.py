"""p3_collect.py -- PREDICT-01 collection driver (Tinker calls). Idempotent: chain()/oracle() skip
anything already recorded in runs/P3/manifest.json.

usage: p3_collect.py PHASE
  trunks   P6_r8 (12 steps, b0.9, texts 0,1,2 x4, ck0..ck11), P2_r32 (T0,T0, 3 zero steps; ck0..2),
           P5_r8_b0 (T0,T0,T1 at b0; ck2, ck3)
  oracles  every oracle needed by P1-P6 on those trunks (+ P5's second recomputation "P3b"),
           P6 fresh twins (ck3/ck7/ck11), P5 fresh steps
  p1a      clip series at step 2 from P6 ck1 (needs predictions written first: p3_predict.py p1)
  p1b      clip thresholds at state 6 (T0) from P6 ck6
  p3       beta' at a zero step from P6 ck2, then a normal T2 step; oracles after the zero step
  p4       lr series (r32 from P2 ck2 on T1; r8 from P6 ck2 on T2); fresh r32 at lr 1e-3/1e-2
  p7       gpt-oss-20b trunk + oracles + beta0 pair          (run in its own process)
  p8       Qwen MLP+unembed r8 trunk + oracles                (run in its own process)
  drift    a new r8 client, step 1 on T0 (regime re-check at the end)
  cleanup  delete every checkpoint of ours still listed
"""
import json, sys, time
import numpy as np
import p3_lib as L
from p3_lib import chain, oracle, run, D, steps_of, man, ck

CYC = [0, 1, 2] * 4
NOW = lambda: time.strftime("%H:%M:%S", time.gmtime())


def trunks():
    jobs = [
        lambda: chain("P6_r8", 8, steps_of(CYC), exp="P6", save=True),
        lambda: chain("P2_r32", 32,
                      [dict(text=0, weight=1.0, params=D(), sid="step1"),
                       dict(text=0, weight=1.0, params=D(), sid="step2")] +
                      [dict(text=0, weight=0.0, params=D(), sid=f"zero{i}") for i in (1, 2, 3)],
                      exp="P2", save=True, skip_save=[3, 4]),
        lambda: chain("P5_r8_b0", 8, steps_of([0, 0, 1], beta1=0.0), exp="P5", save=True,
                      skip_save=[0, 1], save_last=True),
    ]
    run(jobs, workers=3)


def oracles():
    jobs = [(lambda k=k: oracle(f"P6_r8-ck{k}", CYC[k])) for k in range(12)]
    jobs += [lambda: oracle("P2_r32-ck0", 0), lambda: oracle("P2_r32-ck1", 0),
             lambda: oracle("P2_r32-ck2", 1),
             lambda: oracle("P2_r32-ck1", 1), lambda: oracle("P2_r32-ck1", 1, setting="P3b"),
             lambda: oracle("P5_r8_b0-ck2", 1), lambda: oracle("P5_r8_b0-ck2", 1, setting="P3b"),
             lambda: oracle("P5_r8_b0-ck3", 2), lambda: oracle("P5_r8_b0-ck3", 2, setting="P3b")]
    # fresh twins (floor for P6) and P5 fresh steps
    for k in (3, 7, 11):
        jobs.append(lambda k=k: chain(f"P6_fresh_ck{k}", 8,
                                      [dict(text=CYC[k], weight=1.0, params=D(), sid="step")],
                                      exp="P6-fresh", kind="fresh", ckpt=f"P6_r8-ck{k}"))
    jobs.append(lambda: chain("P5_fresh_r32_ck1_t1", 32,
                              [dict(text=1, weight=1.0, params=D(), sid="step")],
                              exp="P5", kind="fresh", ckpt="P2_r32-ck1"))
    jobs.append(lambda: chain("P5_fresh_r8b0_ck2_t1", 8,
                              [dict(text=1, weight=1.0, params=D(beta1=0.0), sid="step")],
                              exp="P5", kind="fresh", ckpt="P5_r8_b0-ck2"))
    jobs.append(lambda: chain("P5_fresh_r8b0_ck3_t2", 8,
                              [dict(text=2, weight=1.0, params=D(beta1=0.0), sid="step")],
                              exp="P5", kind="fresh", ckpt="P5_r8_b0-ck3"))
    run(jobs, workers=4)


def gnorms(key):
    g = man()["grads"][key]
    return g["GA_global"], g["GB_global"], g["G_global"]


P1A_C = [0.7, 0.5, 0.3, 0.1, 0.01]


def p1a():
    # N1 = global norm over ALL LoRA tensors of the step-2 oracle gradient (P6 state 1, T1)
    k1 = man()["steps"]["P6_r8/step2"]["grad_key"]
    NA, NB, N = gnorms(k1)
    print("state1 T1 norms A %.4f B %.4f all %.4f" % (NA, NB, N), flush=True)
    jobs = [(lambda c=c: chain(f"P1a_c{c}", 8, [dict(text=1, weight=1.0, params=D(clip=c * N),
                                                      sid="step2", note=f"c={c} of N_all")],
                               exp="P1a", kind="opt", ckpt="P6_r8-ck1")) for c in P1A_C]
    run(jobs, workers=4)


def p1b_thresholds():
    k6 = man()["steps"]["P6_r8/step7"]["grad_key"]
    NA, NB, N = gnorms(k6)
    return dict(NA=NA, NB=NB, N=N, t1=(NB + N) / 2, t2=float(np.sqrt(NA * NB)), t3=0.5 * NA)


def p1b():
    T = p1b_thresholds()
    print(T, flush=True)
    jobs = [(lambda t=t: chain(f"P1b_{t}", 8, [dict(text=0, weight=1.0, params=D(clip=T[t]),
                                                     sid="step7", note=f"clip={T[t]}")],
                               exp="P1b", kind="opt", ckpt="P6_r8-ck6")) for t in ("t1", "t2", "t3")]
    run(jobs, workers=3)


P3_B = [0.0, 0.5, 0.9, 0.99]


def p3():
    jobs = [(lambda b=b: chain(f"P3_zb{b}", 8,
                               [dict(text=2, weight=0.0, params=D(beta1=b), sid="zero"),
                                dict(text=2, weight=1.0, params=D(beta1=0.9), sid="step3")],
                               exp="P3", kind="opt", ckpt="P6_r8-ck2", save=True, skip_save=[0]))
            for b in P3_B]
    run(jobs, workers=4)
    run([(lambda b=b: oracle(f"P3_zb{b}-ck1", 2)) for b in P3_B], workers=4)


P4_R32 = [1e-3, 4e-3, 5e-3, 6e-3, 8e-3, 1e-2]
P4_R8 = [1e-2, 2e-2, 3e-2, 5e-2]


def p4():
    jobs = [(lambda lr=lr: chain(f"P4_r32_lr{lr:g}", 32, [dict(text=1, weight=1.0, params=D(lr),
                                                                sid="step3")],
                                 exp="P4", kind="opt", ckpt="P2_r32-ck2")) for lr in P4_R32]
    jobs += [(lambda lr=lr: chain(f"P4_r8_lr{lr:g}", 8, [dict(text=2, weight=1.0, params=D(lr),
                                                              sid="step3")],
                                  exp="P4", kind="opt", ckpt="P6_r8-ck2")) for lr in P4_R8]
    jobs += [(lambda lr=lr: chain(f"P4_r32_fresh_lr{lr:g}", 32,
                                  [dict(text=1, weight=1.0, params=D(lr), sid="step3")],
                                  exp="P4", kind="fresh", ckpt="P2_r32-ck2")) for lr in (1e-3, 1e-2)]
    run(jobs, workers=4)


P7_MODEL = "openai/gpt-oss-20b"
P7_TEXTS = [0, 0, 1, 2, 0]


def p7():
    L.set_model(P7_MODEL)
    run([lambda: chain("P7_r8", 8, steps_of(P7_TEXTS), exp="P7", save=True)], workers=1)
    jobs = [(lambda k=k: oracle(f"P7_r8-ck{k}", P7_TEXTS[k])) for k in range(5)]
    jobs += [lambda: chain("P7_b0_cont", 8, [dict(text=1, weight=1.0, params=D(beta1=0.0),
                                                  sid="step3")],
                           exp="P7", kind="opt", ckpt="P7_r8-ck2"),
             lambda: chain("P7_b0_fresh", 8, [dict(text=1, weight=1.0, params=D(beta1=0.0),
                                                   sid="step3")],
                           exp="P7", kind="fresh", ckpt="P7_r8-ck2")]
    run(jobs, workers=4)


def p8():
    L.set_model(train_mlp=True, train_unembed=True, size_mult=8.0)
    run([lambda: chain("P8_r8", 8, steps_of([0, 1, 2]), exp="P8", save=True)], workers=1)
    run([(lambda k=k: oracle(f"P8_r8-ck{k}", k)) for k in range(3)], workers=3)


def drift():
    run([lambda: chain("DRIFT_end_r8", 8, steps_of([0]), exp="drift")], workers=1)


def cleanup(prefix=""):
    names = [n for n, c in man()["ckpts"].items() if not c.get("deleted") and n.startswith(prefix)]
    L.drop_ckpts(names)
    print("deleted", len(names), flush=True)


if __name__ == "__main__":
    ph = sys.argv[1]
    print(NOW(), "start", ph, flush=True)
    globals()[ph](*sys.argv[2:])
    print(NOW(), "done", ph, L.tk.summary(), flush=True)


def p1c():
    """follow-up: determinism + location of the discrete lora_A switch seen in P1a."""
    k1 = man()["steps"]["P6_r8/step2"]["grad_key"]
    N = man()["grads"][k1]["G_global"]
    jobs = [(lambda c=c, tag=tag: chain(f"P1c_{tag}", 8,
                                        [dict(text=1, weight=1.0, params=D(clip=c * N if c < 1 else 0.0),
                                              sid="step2", note=f"c={c}")],
                                        exp="P1c", kind="opt", ckpt="P6_r8-ck1"))
            for c, tag in ((1.0, "c1_rep"), (0.2, "c0.2"), (0.1, "c0.1_rep"))]
    run(jobs, workers=3)
