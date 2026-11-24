"""p3_solo.py -- PREDICT-01 follow-up: everything strictly sequential (one client at a time), to test
whether gradient/step numerics depend on co-scheduling ("clock_cycle:unique" batching).
  part 1: two oracles at P6 state 1 / T1 (settings P3x1, P3x2) and one unclipped step 2 from P6 ck1
  part 2: P2 trunk re-run solo (P2s_r32: T0, T0, zero x3, b0.9) + its two oracles, solo."""
import p3_lib as L
from p3_lib import chain, oracle, D

def part1():
    for s in ("P3x1", "P3x2"):
        oracle("P6_r8-ck1", 1, setting=s)
    chain("P1s_c1", 8, [dict(text=1, weight=1.0, params=D(), sid="step2", note="solo")],
          exp="P1s", kind="opt", ckpt="P6_r8-ck1")

def part2():
    chain("P2s_r32", 32,
          [dict(text=0, weight=1.0, params=D(), sid="step1"), dict(text=0, weight=1.0, params=D(), sid="step2")] +
          [dict(text=0, weight=0.0, params=D(), sid=f"zero{i}") for i in (1, 2, 3)],
          exp="P2s", save=True, skip_save=[2, 3, 4])
    oracle("P2s_r32-ck0", 0, setting="P3x1")
    oracle("P2s_r32-ck1", 0, setting="P3x1")

if __name__ == "__main__":
    L.tk.service(); L.tk.rest()
    part1(); part2()
    print(L.tk.summary())


def part3():
    """solo: oracles for P4 (r32 state 2/T1, r8 state 2/T2); r32 lr pair from P2 ck2; clip t1 vs
    unclipped at P6 state 6 (does N include lora_A?)."""
    import p3_collect as C
    oracle("P2_r32-ck2", 1, setting="P3x1")
    oracle("P6_r8-ck2", 2, setting="P3x1")
    for lr in (1e-3, 1e-2):
        chain(f"P4s_r32_lr{lr:g}", 32, [dict(text=1, weight=1.0, params=D(lr), sid="step3", note="solo")],
              exp="P4s", kind="opt", ckpt="P2_r32-ck2")
    T = C.p1b_thresholds()
    chain("P1bs_none", 8, [dict(text=0, weight=1.0, params=D(), sid="step7", note="solo")],
          exp="P1bs", kind="opt", ckpt="P6_r8-ck6")
    chain("P1bs_t1", 8, [dict(text=0, weight=1.0, params=D(clip=T["t1"]), sid="step7", note="solo")],
          exp="P1bs", kind="opt", ckpt="P6_r8-ck6")


def part4():
    L.set_model(train_mlp=True, train_unembed=True, size_mult=8.0)
    chain("P8_r8", 8, L.steps_of([0, 1, 2]), exp="P8", save=True)
    for k in range(3):
        oracle(f"P8_r8-ck{k}", k)
    L.set_model()
    chain("DRIFT_end_r8", 8, L.steps_of([0]), exp="drift")
