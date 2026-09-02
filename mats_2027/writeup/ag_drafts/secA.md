The NLA paper reports that the activation reconstructor is "only a weak per-claim verifier" of
the descriptions its verbalizer produces. I wanted to know which kind of weak: is there no
per-claim signal in reconstruction error at all, or is there one that is buried in noise? The
difference matters, because a buried signal is a sampling problem someone can pay to fix, and an
absent one is a dead end.

Setup: the official Gemma-3-12B NLA, layer 32, on 6 passages sampled at their last 10 token positions with 4 resamples each — 240 explanations, 2,065 claims, each rewritten out of its explanation and re-scored against the same activation. Δ = mse(claim removed) − mse(intact). A second corpus of 7 biography pages tested transfer.

**What I found**

- **The signal exists and it is weak, now with a number: per-claim AUC 0.535 [0.510, 0.559].**
  The paper never quantifies "weak". Averaging over resamples moves it to 0.615, but that
  interval [0.488, 0.736] still includes chance.
- **The noise is random, not systematic — so averaging is the right lever.** Fitting
  spread(K)² = signal² + noise²/K on K=1 and K=4 predicts K=2 and K=3 to within 0.8%, and two
  independent noise estimates agree. There is a signal floor at 56% of the K=1 spread. **The bottleneck is not noise but recurrence**: only 110 claim-groups appear in at least 3 of 4 resamples, so more resamples would settle this in a day of GPU time.
- **My main hypothesis is dead, and I can say how dead.** I predicted that "false" claims are two populations — faithful readouts the judge mislabels, plus real confabulations — which would make their Δ bimodal. It is one hump (dip test p = 0.992), and planted mixtures at the sizes I predicted are caught 86–100% of the time. A mixture below a fifth would have been missed.
- **Removing a claim improves reconstruction 30% of the time.** The paper never reports this rate.
  Thematic claims carry ~2.7× less reconstruction weight than specific ones — but only after
  controlling for claims that quote the passage's final token; uncontrolled the gap looks 4.5×.
- **Specificity replicates** on a different NLA, corpus and judge: THEME 69.1% / ENTITY 43.8% / DETAIL 36.0% supported, against the paper's 64 / 28 / 24.
- **Confabulation is import, not misreading.** In both corpora, over 90% of false claims naming a
  person name someone absent from the passage entirely. 98% of false claims stay on-topic. And the
  less familiar corpus produced **more** confabulation, not less (63% false vs 50%) — refuting
  predictions I and my agent both wrote down in advance.

**What I checked myself.** I hand-graded 150 claims blind plus 30 retests (96.7% self-consistent, 88.7% agreement with the judge); the paper reports no validation of its own confabulation judge. Several load-bearing claims failed when I checked them: a pre-registered detector rule that
turned out to fire on skew alone, a kill condition no dataset with real signal could have passed,
and a widely-repeated detail in my own notes that did not exist in the data.

**What this is not.** Everything here is black-box. The one causal intervention I attempted never
reached the representation — a control I ran afterwards showed the text edit moved the activation
0.7% as far as a single token step does — so that question is untested rather than answered. The
next step I would take is the paper's own unrun suggestion, best-of-N explanations scored against
the reconstructor, which my existing resamples already support; the causal question needs
activation patching, and that is the first white-box thing I would do.
