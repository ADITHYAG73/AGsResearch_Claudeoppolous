The AV imports famous names it knows

while reading the explanations for Rahul Dravid's Wikipedia biography, I noticed the AV kept writing about Don Bradman. Bradman is in that passage exactly once, and not as its subject: "In December 2011, he was the first non-Australian cricketer to deliver the Bradman Oration in Canberra." The records in the passage — most balls faced in Tests, longest time spent batting — are Dravid's. The AV attached them to Bradman instead, 21 claims across four token positions, including the flat assertion "The text is about Don Bradman" and, twice, describing him as Indian — "The text discusses Indian batsman Don Bradman" and "The text is about an Indian cricketer named Don Bradman".

But Bradman isn't the interesting part here. Across the claims on that passage the AV names Dravid 27 times, Bradman 21, Tendulkar 20, Lara 6, Richards 3,Pietersen 2, Gavaskar 2, Hobbs 1. Tendulkar does not appear in the passage at all. Neither do Lara, Richards, Pietersen, Gavaskar or Hobbs. Bradman only looked special because he is the one imported name that happened to also be in the text — so he alone fit a story about misreading
what was there. The rest are simply famous batsmen the model knows, written onto a passage that never mentions them.


I then designed an experiment to test whether the confabulation actually follows the name in the text. If the AV was picking up "Bradman" from the passage, then removing that name should remove Bradman from the explanations, and putting a different famous batsman there should pull the confabulation onto him instead. I edited one proper noun in that sentence and left everything else untouched, giving seven conditions: the original; Gavaskar (famous, holds records); Umrigar (real, far less famous); Thangavelu (a name the model cannot know); a version with no proper noun at all; the sentence deleted; and a rewritten-coherent version. Forty explanations per condition, sampled at the same ten token positions.

Both of us(me and opus5) had written predictions down beforehand. I expected Bradman to disappear when the sentence was deleted, and expected the Gavaskar condition to reverse the direction — Gavaskar's records attached to Dravid. Claude predicted the same direction with a weaker effect.

Neither happened. Deleting the sentence entirely left Bradman in 25% of the explanations, against 22.5% in the original. Planting Gavaskar, Umrigar or Thangavelu produced zero uses of each name in forty explanations. The name in the text was neither necessary nor sufficient.


Before reading anything into that, I checked whether the edit had done anything at all to the
thing the AV actually sees. It had not. When comparing the layer-32 activations between conditions, i observed the edited and unedited versions sit at a mean-centred cosine of 0.997 to 0.9995 of each other.

For scale, on the same axis: moving one token position along the same passage gives 0.422 on
average, and a different passage sits at roughly zero. Per condition, aligned by offset from the
end of the passage (the edits change token counts, so absolute positions are not comparable):

| what was changed in the text | mean centred cosine vs the original | worst of the 10 positions |
|---|---:|---:|
| Gavaskar substituted for Bradman (one word) | 0.9995 | 0.9989 |
| Umrigar substituted (one word) | 0.9989 | 0.9965 |
| Thangavelu substituted (one word) | 0.9993 | 0.9986 |
| proper noun removed, length kept | 0.9970 | 0.9796 |
| **whole sentence deleted (104 characters, 26 tokens)** | **0.9968** | 0.9875 |
| sentence rewritten coherently | 0.9979 | 0.9916 |
| *one token step along the same passage* | *0.422* | — |
| *a different passage entirely* | *−0.04* | — |

Deleting the sentence outright moved the activation **0.7% as far as a single token step does**,
and 3.5% even at the worst of the ten positions. (Cosines are mean-centred on the 60 activations
from the main run, deliberately not on the patch data itself; the raw residual stream is so
anisotropic that everything sits above 0.96 against everything else.)

So the honest reading is not "planting a name does nothing". It is that **at these token
positions the intervention never reached the representation**, and the question I set out to ask is untested rather than answered. The design could not have worked: I deliberately placed the edit far enough upstream that it would not change the sampled tokens, which is exactly why it had no effect on them. A real causal test needs activation patching with hooks, or sampling positions adjacent to the edit.

What the control does establish is worth keeping. At layer 32, at these positions, the residual stream barely encodes context from 250 characters back. So the AV cannot be reading "Bradman" out of the activation — i believe the name has to be coming from the model's own knowledge.
