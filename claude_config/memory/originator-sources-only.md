---
name: originator-sources-only
description: "For social media and announcements, only the originating lab or company's own post counts as a source; never cite a third party's tweet or summary about it."
metadata:
  node_type: memory
  type: feedback
  originSessionId: 5d62c3d5-0568-4062-a81c-214d5eefcc27
  modified: 2026-10-06T07:54:06.408Z
---

When a claim comes from a tweet, blog or announcement, AG accepts only the post from the foundational lab / originator company itself (e.g. @huggingface's own tweet, the lab's own blog or Space). A third party's tweet summarising it is not a source and must not be cited.

**Why:** On 2026-10-06 Claude cited an influencer's tweet about a Hugging Face post; it was both a third-party source and about a *different* HF post from the one AG meant. AG: "only source tweet from the foundational lab and originator company is important and shall be considered." Extends [[check-before-you-claim]].

**How to apply:** Search results that surface a commentator's post are a lead only. Follow it to the originator's own page and read that (for a JS-rendered Hugging Face Space, pull the repo files via the HF API rather than trusting a fetch of the empty shell). If the originator's source can't be reached, say so instead of citing the commentary.
