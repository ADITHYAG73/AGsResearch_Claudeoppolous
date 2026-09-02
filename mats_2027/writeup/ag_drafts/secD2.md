The paper calls the AR "only a weak per-claim verifier". The main quantity is

Δ(mse) = mse(ablated explanation) − mse(original explanation)

If Δ is positive, removing the claim worsened the reconstruction, so the claim is load-bearing. If Δ is negative, removing the claim helped, so the claim is not load-bearing.

I set out to verify that "claim" — pun unintended. Simple as the statement sounds, it took me a while to register what it actually means. The original question I set out to answer, along with my favourite knowledge partner in crime (Opus 5), was this: does removing false claims improve reconstruction?

(The question is taken from Neel Nanda's MATS 12.0 admissions doc, under Improved Interpretability Methods: using the activation reconstructor to measure the quality of a description by finding "which claims can be removed and improve reconstruction accuracy".)

And in the process, this is what I found.

| verdict | Δ positive (load-bearing) | Δ negative (removal helped) | total |
| --- | --- | --- | --- |
| TRUE claims | 780 (73%) | 288 (27%) | 1068 |
| FALSE claims | 665 (67%) | 330 (33%) | 995 |

That is 2063 ablations: every one of the 2065 claims except two, which could not be rewritten out of their explanation without changing something else. Verdicts are binary, supported against not supported.

A detector that simply says "positive Δ means the claim is true" is right **53.8%** of the time. That is 780 true claims with a positive Δ plus 330 false claims with a negative Δ, over 2063. Always guessing "true" gets **51.8%**, so the whole of Δ buys me about two points.

About 27% of true claims have a negative Δ, and they are not thereby false claims.

| Δ | level / subtype | claim |
| --- | --- | --- |
| −0.00083 | THEME / format | The text is a sports/cricket article format. |
| −0.00056 | THEME / format | The text is structured as a biographical article listing cricket statistics |

Both are true, both are vague, and both say something the rest of the same explanation already says. Removing one costs the reconstruction nothing, and slightly helps it.

About 67% of false claims have a positive Δ, and they are not thereby true claims either.

| Δ | level / subtype | claim |
| --- | --- | --- |
| +0.00927 | DETAIL / quote | The text contains the final token "Garden Gardens" |
| +0.00910 | DETAIL / date | The text contains the phrase 'By summer 1789' |

These are specific, they are wrong, and the AR still needs them. What I notice about all three is that each one is pointing at the right place in the passage and then getting the content wrong. Across all false claims, the ones that name the final token of the prefix carry a mean Δ of +0.00141 against +0.00038 for the rest — nearly four times as much — and 25.3% of the load-bearing false claims name it, against 11.5% of the ones whose removal helped.

The explanation is written by the AV, which is a language model in its own right. The reconstruction signal is available only to the AR, never to the AV. The AV is handed an activation at a chosen position so that we can get a readout at that position, but it is still a language model doing what language models do.

And that is the whole problem with reading Δ as a truth signal. Δ measures whether the AR needed those words to rebuild the activation, not whether they were true. A vague true claim that the explanation already states three times is not needed even once; a confidently wrong claim that pins down where in the passage the model was reading is needed badly. The two questions come apart, and 53.8% is what that looks like as a number.
