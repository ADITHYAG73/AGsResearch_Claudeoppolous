# A forking-tokens experiment with Qwen3.5-9B

I replicated a forking experiment on one of the recent open models, Qwen3.5-9B. On a question
the model is confident about, no token swap changed the final outcome. On a question it was
torn about, six alternate tokens did. And ten free samples at the start told the two questions
apart before I spent anything on the full run.

## What the papers claimed and what was open

The 2024 forking paper by Bigelow et al. ([arXiv 2412.07961](https://arxiv.org/abs/2412.07961))
found that single tokens can flip the answer, on GPT-3.5. The follow-up paper in 2026
([arXiv 2608.19611](https://arxiv.org/abs/2608.19611)), released recently, showed that much
of the jaggedness at low sample counts was sampling noise, on Llama-3-8B. I was curious
whether the effect still replicates on a fairly recent model, and that was my motivation.
Also, neither paper asks which questions fork. I had to ask, because I was bottlenecked by
compute.

## What I did

I chose a fairly recent model, Qwen3.5-9B, with thinking turned off. My thought going in was
that newer models, open or closed, should have fairly aced the benchmark datasets.

I started with row 80 of tinyMMLU. I did not pick it at random: when I analysed the token-level
data the 2026 authors released for Llama-3-8B, row 80 had the clearest fork, a round bracket
versus a square bracket that flips the answer between C and A. On Qwen, row 80 gave the right
answer, and it did not fork. Although I had predicted that, I was still a little disappointed.
I wanted to see the branching for real, and a negative result could not show me that.

So I came up with a fairly simple experiment. Since I was bottlenecked by compute, I had to
choose questions where the model was uncertain. Even a wrong answer does not help if it is
always wrong. Only a question where the model is split between two or three options could be
fruitful. For each of the 100 tinyMMLU questions I recorded the greedy answer once and sampled
10 free answers from the start. Qwen got 14 questions wrong. On 17 it was torn, meaning its
most common answer took 8 or fewer of the 10 samples.

From those I picked row 41, where the ten samples split five A and five C. Then I used a stride
of 4 and swept across the generation: at every fourth token I looked at the next-token
probabilities, and wherever more than one token had at least a 5% chance, I forced each of
those tokens in turn and let the model continue. For row 41 that gave 55 positions and 159
branches. 55 of those branches are the token Qwen actually wrote, which serve as the baseline,
and 104 are alternatives. I used two sampling settings inspired by the papers: 50 continuations
per branch for the full sweep, then 200 at the positions that looked like forks.

## What I found

On row 80, Qwen not only got the answer right, it never forked. 125 alternative tokens at 81
positions, and the largest effect was a TVD of 0.04, which is below the sampling noise at 50
samples. [FIGURE 1]

On row 41 I could clearly see forking at S = 50. Six alternate tokens, sitting at five
positions, changed the distribution of final answers beyond sampling noise, where chance
would give about one. I scaled those positions to S = 200 and all six persisted. [FIGURE 2]

The clearest one is at token 164. After " It", 88% of the continuations end on A. After
" However", it is a coin flip, 53 to 47. At token 52, " Offering" against " The" flips the
majority from C to A, and that flip holds at 200 samples too.

## What this does not show

I have run only two questions in full. The evidence needs more data and more rows. The stride
was four, so the experiment is blind to forks that sit between the sampled positions. The 5%
cutoff means tokens rarer than that never get a chance. I tested one model, with thinking
turned off; more models, and thinking on versus off, would be interesting.

I set a bar of 0.5 for TVD before the run. Qwen's largest fork was 0.35: about a third of the
answers moved. That is far below the bracket fork I found in the authors' released Llama-3-8B
data, where 96% moved.

## What comes next

[AG TO DICTATE]

---
Code, data and the experiment journal: github.com/ADITHYAG73/AGsResearch_Claudeoppolous,
under `forking/`. The experiments were run with Claude (Fable 5.1) writing the scripts and
managing the pods; the questions, choices and this text are mine.
