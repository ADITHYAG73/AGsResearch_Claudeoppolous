My hypothesis H2: the signal is there, but it is buried in noise. Expressed as an equation,

Δ(observed) = Δ(underlying) + ε

If ε is random, averaging over multiple resamples should neutralise it. If it is systematic, averaging has no effect at all.

My kill condition, written 19 August before the data: if the spread does not shrink when I average, the noise is systematic and averaging cannot rescue it.

**Method**

1. I took the claims that appeared in all four resamples — 31 of them.
2. For each claim, I averaged its Δ over K = 1, 2, 3 and 4 of its resamples. Then I measured the spread: the standard deviation, across those 31 claims, of the per-claim averages. One number per K. It says how far apart the claims sit from each other, not how much any single claim wobbles.
3. The spread shrank: 0.00225, 0.00181, 0.00164, 0.00157 at K = 1, 2, 3, 4.
4. I fitted the model spread(K)² = signal² + noise²/K on K = 1 and K = 4 only, then made it predict K = 2 and K = 3 — values it had never been shown. Both predictions came within 0.8% of what was observed.
5. The noise estimated from that fit is 0.00185; the noise estimated directly from within-claim variation is 0.00212. Two independent routes agreeing to 13%.
6. I concluded that the noise is random and not systematic. The spread shrank toward a floor at 0.00127 — 56% of where it started — and that floor is the genuine between-claim signal, which is the thing I wanted to exist.

**The kill condition was mis-specified, and I want to be explicit about that.**

7. The condition required the spread to fall as one over the square root of K, which on a log-log plot means a slope of −0.50.
8. I observed a slope of −0.262, which on the face of it reads as a partial fail.
9. But a slope of −0.5 only happens if the spread can fall all the way to zero, which is to say if there were nothing but noise. Because there is a real signal floor, the spread flattens onto it and the slope is necessarily shallower than −0.5. The condition was drafted by my agent and I adopted it; neither of us noticed at the time that no dataset containing real signal could ever pass it.
10. So the right test was not the slope. It is whether the variance model predicts data it was not fitted on, which it does, to 0.8%.

Two earlier noise ratios in my notes, 1.41× and 0.12×, used the median within-claim spread. That is the wrong statistic for splitting variance, because the distribution of within-claim spreads is badly skewed. They are superseded by the numbers above.

**Conclusion.** The outcome I was hoping for was a rise in AUC. The unaveraged per-claim AUC was 0.535, with a 95% interval of 0.510 to 0.559 — barely above chance. K-averaged over the matched groups it is 0.615, interval 0.488 to 0.736. It moves the right way, but the interval includes chance, so it is not established.

The reason it is underpowered is not noise. It is recurrence: only 110 claim-groups span at least 3 of the 4 resamples, so there are only 110 claims whose Δ can be averaged at all. A decisive test needs roughly ten times more, which means more resamples per activation — a GPU session I did not run.

H2 is therefore partially supported. The premise was right, the noise is real and random, and averaging shrinks it predictably. The payoff is underpowered, and I could not establish it at this sample size.
