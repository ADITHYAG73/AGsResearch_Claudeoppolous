## Reflections

The practical implication is narrow but usable: **you cannot filter an NLA description
claim-by-claim with the reconstructor as it stands.** An AUC of 0.535 is not a tool. But the
reason it fails is now specific rather than mysterious. The noise is stochastic, so averaging
does help; the limit is that only 110 claim-groups recurred often enough to average over. That is
a sampling problem with a known price — more resamples per activation — not a dead end. Someone
with a GPU budget could settle it in a day.

The confabulation results point somewhere different from where I started. If most false claims
import a name the passage never mentions, then checking descriptions against the *context* is the
wrong shape of defence, because the AV never saw the context. Checking named entities against
what the activation can actually support looks more promising, and cheaper.

With more time, the first thing I would run is one the paper names and did not do. Under
inference-time methods it notes that the pipeline "mostly uses AV outputs and discards the AR",
and that a simple extension is taking a best-of-N explanation scored against AR reconstruction. I
have the resamples and the scoring already. That asks a whole-explanation question rather than a
per-claim one, which sidesteps the recurrence bottleneck entirely — no claim matching is needed to
ask whether the best-reconstructing of four explanations contains fewer false claims than a
randomly chosen one.

After that, in order: whether the AV's confabulated name depends causally on the activation at
that position, which needs patching rather than a text edit; K=12 resamples to decide H2; and a
third corpus at a third level of familiarity to test whether confabulation tracks how well the
model knows the material.

The honest gap is that none of this touched the model's internals. Everything here treats the NLA
as a black box, and the one intervention I attempted never reached the representation. That is
the first thing I would fix.
