Everything in this write-up comes out of the pipeline in Figure G0. Rather than describe it again, here is one activation carried all the way through, so that the tables in D2 and D5 can be read properly. **This one is chosen, not random** — I looked for an activation with a readable explanation and a mix of true and false claims, because it has to serve as the worked example. The randomly drawn claims are the ones right after the executive summary.

**The passage.** Eden Gardens, Wikipedia. The activation is taken at token position 87, and the prefix — which is all the judge ever sees — ends mid-quote:

> The stadium was established in 1864. The origins of its name are uncertain. According to some, the stadium is named after the Eden Gardens park where it is located, itself named after the Eden sisters, Emily and Fanny, of Lord Auckland, the Governor-General of India (1836–1842). Initially named the 'Auckland Circus Gardens', the park was renamed to the **'Eden Gardens**

**The explanation.** This is what the AV wrote from that single activation vector. It never sees the passage.

> Structured factual/historical article format, with a descriptive entry about a British colonial residence in Kolkata — factual, informative tone.
>
> The sentence "The garden was named 'Elphinstone Gardens', which was later renamed to 'Elphinstone Gardens" sets up a factual conclusion about the name's origin or etymology.
>
> Final token "Gardens" closes a quoted phrase ("was renamed to the 'Elphinstone Gardens"), requiring immediate completion — likely "in the 1800s" or "name by..." or "after the name of..." or "in the estate in 1885," referencing the original naming source.

Look at what it got right and what it got wrong. It knows it is at the end of a quoted park name that is being renamed. It knows the register, the century, the colonial Kolkata setting. And then it fills the name in as **Elphinstone Gardens**, which is not in the passage at all. Right slot, wrong content. This is the failure mode D6 is about, in one line.

**The claims, the verdicts, and Δ.** Stage 3 breaks the explanation into atomic claims, the judge marks each one against the prefix above, and then each claim is rewritten out and the explanation re-scored.

| Δ | level | verdict | claim |
|---:|---|---|---|
| **+0.00849** | ENTITY | false | The text mentions 'Elphinstone Gardens' as a name. |
| +0.00220 | DETAIL | false | The text contains a quoted phrase "was renamed to the 'Elphinstone Gardens". |
| +0.00108 | DETAIL | false | The text contains the sentence "The garden was named 'Elphinstone Gardens'…" |
| +0.00006 | THEME | **true** | The text has a factual, informative tone. |
| −0.00010 | ENTITY | false | The text describes a British colonial residence in Kolkata. |
| −0.00011 | THEME | **true** | The text is in a structured factual/historical article format. |

This one activation shows the whole problem in miniature. **The most load-bearing claim in the explanation is false** — take "Elphinstone Gardens" out and the reconstruction gets substantially worse, because that invented name is carrying the AR's information about where in the sentence the model was. Meanwhile both true claims sit at effectively zero, and one of them is slightly *better* off removed. If you tried to use the sign of Δ as a truth signal here, you would get four out of six wrong.
