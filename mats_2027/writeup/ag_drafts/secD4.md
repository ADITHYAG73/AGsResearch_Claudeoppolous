My hypothesis H2 : signal is there but its buried in noise (what signal?? delta?)

to express it in the form of an equation,

observed_delta = underlying_delta + noise

if the noise is random, averaging over multiple resamples should neutralise it, otherwise it shall have no effect (in case of systematic noise)

my kill condition : if the spread doesn't shrink upon averaging the noise is systematic  and averaging can't rescue it.

methodology:

1. firstly i took the claims that appeared in all 4 resamples. 
2. for each claim, average its delta over K = 1, 2, 3 and 4 of its resamples. Then measure the SPREAD: the standard deviation, across the 31 claims, of those per-claim averages. One number per K. It says how far apart the claims sit from each other — not how much any single claim wobbles.
3. the spread shrank : 0.00225 → 0.00181 → 0.00164 → 0.00157 (K = 1, 2, 3, 4). These are overall numbers, one per K, across all 31 claims.
4. fit the model spread(K)² = signal² + noise²/K on only K=1 and K=4, then made it predict K=2 and K=3 — values it had never been shown. Both predictions came within 0.8% of what was observed.
5. the noise estimated from that fit is 0.00185; the noise estimated directly from within-claim variation is 0.00212 — two independent routes agreeing to 13%.

I concluded that the noise is random and not systematic.