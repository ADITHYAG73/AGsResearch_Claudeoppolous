Everything downstream of the judge depends on the judge, so here are six claims drawn **uniformly at random** from all 2,063 graded ablations (`pipeline/random_examples.py`, seed 20260903, no filtering of any kind — whatever came out is what is printed). Judge my labels for yourself.

**1. Eden Gardens, position 88** — prefix ends `...the park was renamed to the 'Eden Gardens'`
Claim (DETAIL/quote): *"The text contains the sentence 'The garden was renamed to the Victoria Gardens but the name was later changed to the Victoria Gardens...'"* — judge: **CONTRADICTED**, Δ = +0.00044.
The AV has the slot right and the name wrong, and invents a quotation around it.

**2. Rahul Dravid, position 138** — prefix ends `...the most balls faced in Test cricket`
Claim (DETAIL/number): *"The text mentions 805 dismissals"* — judge: **NOT_IN_TEXT**, Δ = +0.00189.
A number that appears nowhere, and one of the more load-bearing claims in the whole set.

**3. Rahul Dravid, position 139** — prefix ends `...most balls faced in Test cricket and`
Claim (DETAIL/statistic): *"The text establishes that Dravid holds the record for most Test catches"* — judge: **NOT_IN_TEXT**, Δ = −0.00052.
True of Dravid in the world, absent from this prefix, and its removal *helped* reconstruction.

**4. Rahul Dravid, position 143** — prefix ends `...and the longest time spent`
Claim (ENTITY/person): *"The text mentions Sir Vivian Richards"* — judge: **NOT_IN_TEXT**, Δ = +0.00023.
Another famous batsman who is not in the passage. This is the import failure in D6, caught at random.

**5. Rahul Dravid, position 147** — prefix ends `...the longest time spent batting in Tests.`
Claim (THEME/content): *"The text contains a factual summary of a cricketer with biographical details and statistics"* — judge: **SUPPORTED**, Δ = +0.00008.
True, vague, and worth essentially nothing to the reconstruction — the pattern D5 is about.

**6. French Revolution (the maintainers' example), position 86** — prefix ends `...the National Assembly had abolished feudal privileges across`
Claim (DETAIL/content): *"The text describes a decree eliminating feudal privileges."* — judge: **SUPPORTED**, Δ = −0.00031.
True, clearly supported, and removing it made the reconstruction slightly better.

Five of these six are false, against an overall false rate of 48% — that is small-sample noise, not a second finding. What I would draw your attention to instead is that **the two true claims here sit at +0.00008 and −0.00031, while the two most load-bearing claims are both false.** Six random claims, and the sign of Δ gets four of them wrong.
