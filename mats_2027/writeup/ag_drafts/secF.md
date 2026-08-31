Limitations

1. NLA as a black box. I used no hooks, no internal probing. Throughout this study, I treated the NLA as a black box. I tried to modulate the input to observe a causal effect on the explanations, but my control showed the edit never reached the representation — deleting a sentence ~250 characters upstream moved the layer-32 activation by less than 1% of what moving a single token position does (centred cosine 0.997 vs 0.42). So the intervention was not causal evidence, and I still don't know why the AV does what it does.

2. Judge valiadtion on cricket data (88.7%). The same was not done for the "Savarkar corpus" partly because I have not read the book completely and also due to time crunch.

3. The semantic matcher was never checked by a human. It is an LLM that decides which claims, across the four resamples of the same activation, are "the same claim". Those groups are the entire basis of the K-averaging result in D4 — 110 groups spanning at least 3 of the 4 resamples — and nobody has ever read a group and confirmed it is one claim rather than two.

4. The two hypothesis results are weaker than they look. H2's payoff could only be measured on the 110 claim-groups that span at least 3 of the 4 resamples, so the bottleneck is recurrence, not noise, and the K-averaged AUC's confidence interval still includes chance. H1's verdict is exploratory rather than confirmatory, because the detector rule I pre-registered turned out to be broken and had to be revised after I had already seen the real distribution.

5. I did not fully finish the Savarkar experiment to report the final delta. I took it as far as explanations, claim decomposition and verdicts — enough to compare confabulation rates across the two domains — but never ran the ablation or the AR scoring on it, so there is no Δ on a second domain.

6. The paper's related-vs-unrelated comparison cannot be reproduced on my data. 975 of the 995 false claims are RELATED to the passage, leaving about 20 unrelated ones, several of which look mislabelled. There is effectively no unrelated cell to compare against.

7. The final-token control is a heuristic, not a labelled category. It flags a claim when the last content word of the prefix appears in it, which cannot separate "names the final token" from "happens to reuse that word", and cannot fire at all on the 22 of 60 prefixes that end in a stopword. It rules that rival down, not out.
