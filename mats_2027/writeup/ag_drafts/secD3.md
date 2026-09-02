My hypothesis H1: claims that are marked false are effectively a combination of two different categories.

**Category 1.** The AV had a faithful readout of the activation, but the text judge marked it false because the words are absent from the passage. The kind of claim I had in mind is the one on the French Revolution passage — "The text contains the phrase 'By summer 1789'" — where the phrase is not in the prefix but the period almost certainly is in the activation.

**Category 2.** Genuine confabulations, such as "The text is about Don Bradman" on a passage about Rahul Dravid.

Based on H1, I predicted two bumps in the histogram of Δ for the 975 related-false claims. If the test returned one bump, H1 is effectively ruled out. That kill condition was written down on 19 August, before any of this data existed.

The dip test found no valley between two bumps: it returned p = 0.992 on the 975 claims, ruling out H1.

A null result, though, means either the test is blind or the data genuinely does not carry the hypothesised structure. To check that the test was not blind, I planted fake data that genuinely had two groups, at H1's own predicted mixture size and at the noise level I had measured.

The dip test caught them 26% of the time at a 20% mixture, 86% at a 26% mixture, and 100% at 35% and above. So at or below a fifth the test goes blind.

The test would therefore have found what H1 described. A smaller mixture, below about a fifth, would have slipped past.

The rule I had pre-registered was not actually the dip test. It was ΔBIC greater than 10 — "do two bell curves fit better than one?"

On my data the two bell curves did fit, at +843.4. But my Δ distribution is lopsided (skew 2.63), and a lopsided single hill is fitted better by two bells than by one, whether or not there are two populations underneath.

To check that, I generated data that was one group by construction, lopsided by exactly the same 2.63, and ran the rule on it 200 times. It reported "two populations" in 200 of 200 runs, with a median score of +846.5 — higher than the +843.4 my real data scored. My evidence for two populations was weaker than what a single population typically produces.

Because the rule had to be revised after I had seen the data, H1's verdict is exploratory and not confirmatory. When I was brainstorming with my thinking and experimenting partner (Claude), we did not account for a distribution this lopsided.

Killing H1 does not mean the AR treats Category 1 (activation-true, text-false) and Category 2 (genuine confabulation) claims alike. It shows only that their measured Δ does not split into two groups. This measurement cannot tell them apart at this sample size.
