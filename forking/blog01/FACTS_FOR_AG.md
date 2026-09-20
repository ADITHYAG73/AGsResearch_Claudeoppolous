# Facts for AG to say in his own words (Claude wrote these; do NOT paste them into the post)

LessWrong policy (14 Mar 2026): human text "lightly edited" by an LLM is fine; text
"substantially edited or revised by an LLM" counts as LLM output. Facts "developed with LLM
assistance" are explicitly fine. So: read a fact here, close the file, say it your way.

1. **Why row 80.** Not random. The 2026 authors released their Llama-3-8B data. Analysed
   token by token, row 80 at token 228 is one of the clearest forks: a round bracket ")("
   versus a square bracket ")[" sends the answer to C (196 of 200) or to A (169 of 200),
   TVD 0.96. The PAPER does not report this; it comes from analysing their public data.
   **CORRECTED 2026-09-20 (Claude's error):** I first wrote "THE clearest fork". It is not.
   Re-checked against `runs/2026-09-13_swap/swap_effects_nonletter.csv`: row 50 t=236
   "false"->"true" is larger at TVD 0.990. Row 80 IS the largest among swaps sitting at a
   FLAT pooled-curve position (local change 0.04) — that is the precise claim. The post was
   edited in the LessWrong draft to "one of the clearest forks".
2. **"Changed the final outcome."** Precisely: all six moved the mix of final answers beyond
   sampling noise. The majority answer flips at two positions (token 52 and token 76). At
   token 164 it goes from 88% A to a 53/47 coin flip.
3. **"About one third of the time."** TVD is a share of answers, not time. 0.35 = about a
   third of the answers moved.
4. **The two questions** (optional section). Text and options are in DRAFT.md, copied from
   the dataset. Quoting a dataset is not LLM output. The observation that options A and C
   differ only on Scenario 1 (the sundae) is a fact you can state yourself.
5. **Token 164 context** (optional). Just before it, Qwen wrote that the sundae offer "would
   be considered a minor moral failing or at least socially inappropriate, bordering on
   wrong". That is a quote of the model, fine to include as a quote.
6. **Row 80 numbers.** 125 alternatives at 81 positions, largest TVD 0.04.
7. **Row 41 numbers.** 6 of 104 alternatives at p<0.01 (chance ~1); all six p<=0.001 at S=200.
8. **The 2024 paper names the activation idea** in its discussion: "it might even be possible
   to avoid sampling altogether if hidden activations can be used to predict forking."
   (verified today, PAPER_CHECK.md). Worth crediting, in your words.
9. **2024 was also compute-limited**: ~$500, questions "randomly sampled". They sampled at
   random; you sampled by uncertainty.
