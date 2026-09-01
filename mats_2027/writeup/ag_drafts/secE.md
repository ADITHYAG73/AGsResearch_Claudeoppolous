The things that were my contributions:

1. I  graded 150 claims blind plus 30 retests, 180 items in 33.6 minutes, seeing only the claim and the prefix. 96.7% self-consistent, 88.7% agreement with the judge. A human validation for my dataset.
2. I froze the grading conventions the adjudication was then judged against, and the 27 disagreements my blind grading produced turned out to split into two error modes at opposite ends of the specificity scale — I am too generous on quoted strings, the judge is too strict on vague-but-correct descriptions. (The adjudication itself was done by my agent, applying my conventions; it is the same model family as the judge, so it is not a neutral referee.)
3. I  spotted the Bradman substitution — it needed cricket knowledge , which is my forte.
4. I had an hypothesis that confabulations tend to be more towards the ending positions of sentences. But later i found out it was not the case.
5. I found the relatedness labels didn't exist, by checking my own mental model against the data — that blocked H1 until it was fixed.
6. I asked where a number in my own notes came from — the 99.94 batting average my agent said the model had blended in — and it was nowhere in the data. It had been carried over from a different run and had propagated into three files. The averages the model actually invented were 95.99 and 51.37, attributed to two batsmen who appear nowhere in the passage.


What I did not do and would like to explore :
no hooks, no internal probing. Black box throughout ..(here u know i wonder what is the natural next step in natural language autoencoders..like the official authors what direction did they mention in their original research paper/article worth pursuing.. i would be interested to know that, coz i have forgotten it, and given neel is drifting away from traditional methods like activation patching or something, need to read what his current interests are..we know for sure NLA is his current interest..but which direction moving fwd..what do anthroopic say..that can also in effect determine my next stes right..instead of me writing hooks and internal probing)