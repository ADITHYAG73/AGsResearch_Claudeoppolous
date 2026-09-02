1. **NLA as a black box.** No hooks, no internal probing. I tried to modulate the input to observe a causal effect, but my own control showed the edit never reached the representation — deleting a sentence 218 to 263 characters upstream moved the layer-32 activation less than 1% as far as moving a single token position does. That intervention is not causal evidence, and I still don't know why the AV does what it does.

2. **The judge was validated on cricket only** (88.7% agreement with my blind grading). I did not repeat that for the Savarkar corpus, partly because I have not read the book fully and partly for time — so some of the 13.8-point gap between corpora could be the judge being harsher on unfamiliar Indian names rather than the AV confabulating more.

3. **The semantic matcher was never checked by a human.** An LLM decides which claims across the four resamples count as "the same claim", and those 110 groups are the entire basis of the K-averaging result in D4. I never sat down and read a group to confirm it really is one claim and not two.

4. **Both hypothesis results are weaker than they look.** H2's payoff could only be measured on the 110 groups spanning at least 3 of 4 resamples, so the bottleneck is recurrence rather than noise, and the K-averaged AUC's interval still includes chance. H1's verdict is exploratory rather than confirmatory, because the detector rule I pre-registered turned out to be broken and had to be revised after I had seen the real distribution.

5. **Savarkar stopped at verdicts.** I took it as far as explanations, decomposition and judging — enough to compare confabulation rates across domains — but never ran the ablation or the AR scoring, so there is no Δ on a second domain.

6. **No unrelated cell.** 975 of the 995 false claims are related to the passage, leaving about 20 unrelated ones, several of which look mislabelled. The paper's related-versus-unrelated comparison cannot be reproduced here.

7. **The final-token control is a heuristic**, not a labelled category: it flags a claim when the passage's last content word appears in it, so it cannot separate "quotes the final token" from "happens to reuse that word", and it never fires on the 22 of 60 passages ending in a stopword. It rules that rival down, not out.
