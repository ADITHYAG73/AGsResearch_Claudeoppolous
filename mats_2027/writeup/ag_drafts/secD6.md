While reading the explanations for Rahul Dravid's Wikipedia biography, I noticed the AV kept
writing about Don Bradman. Bradman is in that passage exactly once, and not as its subject: "In
December 2011, he was the first non-Australian cricketer to deliver the Bradman Oration in
Canberra." The records in the passage — most balls faced in Tests, longest time spent batting —
are Dravid's. The AV attached them to Bradman instead, in 21 claims across four token positions,
including the flat assertion "The text is about Don Bradman" and, twice, describing him as Indian.

But Bradman isn't the interesting part here. Across the claims on that passage the AV names
Dravid 27 times, Bradman 21, **Tendulkar 20**, Lara 6, Richards 3, Pietersen 2, Gavaskar 2,
Hobbs 1 — and only Bradman appears in the passage at all. He looked special because he is the one
imported name that happened also to be in the text, so he alone fit a story about misreading what
was there. The rest are simply famous batsmen the model knows, written onto a passage that never
mentions them.

To test whether the confabulation follows the name, I edited that one proper noun and left
everything else untouched: seven conditions — the original, Gavaskar (famous, holds records),
Umrigar (real, far less famous), Thangavelu (a name the model cannot know), a version with no
proper noun, the sentence deleted, and a coherent rewrite — at forty explanations each. Both of us
had written predictions down first; I expected Bradman to vanish when the sentence was deleted and
the Gavaskar condition to reverse the direction. Neither happened. **Deleting the sentence left
Bradman in 25% of explanations against 22.5% in the original, and planting Gavaskar, Umrigar or
Thangavelu produced zero uses of each.**

Before reading anything into that, I checked whether the edit had changed what the AV actually
sees. It had not.

| what was changed in the text | mean centred cosine vs original | worst of 10 positions |
|---|---:|---:|
| one proper noun substituted (3 conditions) | 0.9989–0.9995 | 0.9965 |
| proper noun removed, length kept | 0.9970 | 0.9796 |
| **whole sentence deleted (104 chars, 26 tokens)** | **0.9968** | 0.9875 |
| *one token step along the same passage* | *0.422* | — |
| *a different passage entirely* | *−0.04* | — |

Deleting the sentence moved the activation **0.7% as far as a single token step does**. So the
honest reading is not "planting a name does nothing" — it is that **the intervention never reached
the representation**, and the question is untested rather than answered. The design could not have
worked: I placed the edit far enough upstream that it would not disturb the sampled tokens, which
is exactly why it did not reach them. What the failed control does establish is that at layer 32,
at these positions, the residual stream barely encodes context from 250 characters back — so the
AV cannot be reading "Bradman" out of the activation at all.

If the names come from the model's own knowledge, the amount of confabulation should depend on how
well it knows the material. So I ran the same pipeline on seven random pages of a 2019 biography
of V. D. Savarkar, length-matched, with the Dravid passage re-run as a regression check (cosine
1.000000). Both of us predicted fewer confabulations. **We were both wrong: 63% of claims on the
biography are false against 50% on cricket** (36.7% supported, n=2730, vs 50.5%, n=1668;
intervals disjoint; the gap holds at every claim level).

The two corpora agree on the mechanism. Of the false claims that name a person, **93.6% on
Savarkar and 98.3% on cricket name someone absent from the passage entirely** — Gandhi, Bhagat
Singh and Tilak on the biography; Tendulkar, Dravid and Bradman on cricket. Re-binding a name that
is genuinely present, the Bradman story I started from, is the rare case.

I also labelled all 995 false claims for relatedness, a thing the paper asserts twice without a
number: **975 of them, 98%, are related to the passage.** Given a cricket activation the AV
confabulates cricket. That also costs me the paper's related-versus-unrelated comparison — with 20
unrelated claims, several of them apparently mislabelled, I cannot reproduce it here.

Putting those together — names imported, imports staying in-domain, more of them on unfamiliar
text — the reading I find most plausible is that **the model's own knowledge is the source of the
specifics the AV gets right, not the source of its errors**: it writes to a fixed level of
specificity whatever it is given, and where the activation is thin it fills that budget with the
nearest famous things it knows. This is an interpretation of three results, not a test. The
experiment that would test it is a third corpus at a third level of familiarity, predicting in
advance that confabulation tracks familiarity while the specificity of the claims does not change.
