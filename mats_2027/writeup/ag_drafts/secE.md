The things that were my contributions:

1. I graded 150 claims blind plus 30 retests, 180 items in 33.6 minutes, seeing only the claim and the prefix. 96.7% self-consistent, 88.7% agreement with the judge. That is the human validation for my dataset.
2. I froze the grading conventions the adjudication was then judged against, and the 27 disagreements my blind grading produced turned out to split into two error modes at opposite ends of the specificity scale — I am too generous on quoted strings, the judge is too strict on vague-but-correct descriptions. (The adjudication itself was done by Opus 5, applying my conventions; it is the same model family as the judge, so it is not a neutral referee.)
3. I spotted the Bradman substitution. It needed cricket knowledge, which is my forte, and it is the reason I chose cricket as the corpus.
4. I had a hypothesis that confabulation would increase towards the later token positions of a passage. I read the explanations by eye before any of the data existed and concluded it was not there; the measurement later agreed, at +2.4 percentage points early versus late against a pooled standard error of 2.9, with the sign opposite to my prediction.
5. I found that the relatedness labels did not exist at all, by checking my own mental model of the pipeline against the data. That blocked H1 until it was fixed.
6. I asked where a number in my own notes came from — the 99.94 batting average Opus 5 said the model had blended in — and it was nowhere in the data. It had been carried over from a different run and had propagated into three files. The averages the model actually invented were 95.99 and 51.37, attributed to two batsmen who appear nowhere in the passage.

What I did not do, and would like to explore: no hooks, no internal probing. This was a black box throughout.
