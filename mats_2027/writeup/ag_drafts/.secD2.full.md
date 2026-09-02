"AR is only a weak per claim verifier"

main equation -> delta(mse) = mse(abalated explanation) - mse(original explanation)

if delta(mse) > 0 => removed claim worsened the recosntruction on ablated explanation -> claim is load bearing for reconstruction

if delta(mse) < 0 => removed claim helped in reconstruction -> claim is not load bearing for reconstruction

I set out to verify this "claim" (pun unintended) . in fact , as much as this might be a simple sounding statement, it took me a while to register it.The original question that I set out to finad an answer along with my favourite knowledge partner in crime (claude) was this -> does removing the false claims help in reconstruction better?

(Note this question was directly refered and taken from Neel Nanda's MATS 12.0 admissions doc, "Recommended Research Problems" tab, under *Improved Interpretability Methods* → natural language autoencoders, where he says he is particularly interested in using the activation reconstructor to measure the quality of a description — "which claims can be removed and improve reconstruction accuracy" — to help reduce hallucinations.)

and in the process this is what i found.

|                  | Δ > 0 (load-bearing) | Δ < 0 (removal helped) | total |
|------------------|---------------------:|-----------------------:|------:|
| **TRUE claims**  | 780 (73%)            | 288 (27%)              | 1068  |
| **FALSE claims** | 665 (67%)            | 330 (33%)              | 995   |

n = 2063 valid ablations, cricket, K = 1, verdicts binary (SUPPORTED vs not).
A detector that simply says "Δ > 0 means the claim is true" is right **53.8%** of the time,
against 51.8% for always guessing "true".

about 27% of true claims do contribute to a negative delta but they are not necessariyl false claims .

| Δ | level / subtype | claim |
|---:|---|---|
| −0.00083 | THEME / format | The text is a sports/cricket article format. |
| −0.00056 | THEME / format | The text is structured as a biographical article listing cricket statistics |
| −0.00052 | THEME / format | The text is structured as a sports article format |

All three are true, all three are vague, and all three say something the rest of the same
explanation already says. Removing one costs the reconstruction nothing, and slightly helps.

about 67% of false claims do contribute to a positive delta but they are not necessarily true claims either.

| Δ | level / subtype | claim |
|---:|---|---|
| +0.00927 | DETAIL / quote | The text contains the final token "Garden Gardens" |
| +0.00910 | DETAIL / date  | The text contains the phrase 'By summer 1789' |
| +0.00407 | DETAIL / quote | The text contains the phrase 'During the series against South Africa, the home team, against a touring Indian team, against the series ahead'. |

These are specific, wrong, and load-bearing. Note what they have in common: each one is pointing
at the right place in the passage and getting the content wrong. Across all false claims, the ones
that name the final token of the prefix carry a mean Δ of +0.00141 against +0.00038 for the rest —
nearly 4×, and 25.3% of the load-bearing false claims name it against 11.5% of the ones whose
removal helped.

so thinking about this , as mentioned in the oriignial paper also , the expalantions are made by AV which is a language model. the only ever signal for reconstruction is avaialble only for AR and not AV. AV although is injected with a desired position activation in order for us to get a model read out or explaantion at that position, its alanguage model

and that is the whole problem with reading Δ as a truth signal. Δ measures whether the AR needed
those words to rebuild the activation, not whether they were true. A vague true claim that the
explanation states three times is not needed even once; a confidently wrong claim that pins down
where in the passage the model was reading is needed badly. The two questions come apart, and the
53.8% is what that looks like as a number.

---
NOTE (Claude, 30 Aug): everything above is AG's except — the citation text in the bracket, the two
tables and the three sentences under each of them, and the final paragraph beginning "and that is
the whole problem". Numbers from evidence/D2_confusion.md and SCORE-01b; all recomputed 30 Aug.
AG's own typos left untouched throughout — fix them on your read-through, not before.
