The paper says the AR "is only a weak per-claim verifier".

I wanted to investigate whether that is because the signal is absent, or because it is buried in noise.

The signal here is Δ(mse). In particular I hypothesise that the observed Δ of a claim is the sum of an underlying Δ and noise:

Δ(observed) = Δ(underlying) + ε

By underlying Δ I mean the Δ of the claim that we could calculate in an ideal scenario, with 100 percent accuracy. ε is everything else: the run-to-run variation that comes from the prose around the claim changing between resamples.

Now either one of two cases is possible.

1. The distribution of underlying Δ is the same for true and false claims, which means they are indistinguishable and no amount of measurement will separate them.
2. The distributions differ, but the accompanying noise is overwhelming — possibly because of the surrounding prose variation across resamples.

There are also two sources of confabulation, and telling them apart is what makes this hard:

1. The residual activation vector genuinely encodes the thing, and the AV faithfully read it out.
2. The AV, being a language model in its own right, made it up.

In essence, a text judge is blind to both the residual activation and the AV's process. The judge in our case (Haiku 4.5) receives only the passage up to the sampled position — the prefix — and the claim, and returns supported, contradicted, or not in text. It can see whether the words are in the passage. It cannot see what the activation contained.

The residual stream activation that was the basis for the AV's explanation is available, in the whole system, only to the AR. So the AR is in principle the only instrument that could tell case 1 from case 2.
