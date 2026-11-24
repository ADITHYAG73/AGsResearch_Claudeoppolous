---
name: latent-cube-experiment
description: "AG's new interp idea (2026-10-29) — make Qwen3.5-4B \"imagine\" the second Taylor term of sin x and look for x³/6 in its hidden states; backlog includes reading Anthropic's Biology of an LLM paper."
metadata:
  node_type: memory
  type: project
  originSessionId: 5d62c3d5-0568-4062-a81c-214d5eefcc27
  modified: 2026-10-29T18:07:26.959Z
---

**The idea (AG's, 2026-10-29):** prompt Qwen3.5-4B to *imagine* (not write) the second Taylor term of sin x; check whether −x³/6 / x³ is represented in hidden states at the last prompt token; compare with other prompts that need a cube latently to find a shared "cube" direction. Separate from the fine-tuning paper ([[posttrain-math-project]]) — AG said keep them apart until he says otherwise.

**Prompt set drafted with AG:** same x³/6 by other routes (eˣ 4th term, sinh 2nd term, C(x,3) leading term); cube without /6 (cube volume x³, ∫x² = x³/3, tan 2nd term x³/3, Rubik's 27 = 3³); /6 without cube (one sixth of a pizza); controls (cos 2nd term −x²/2, sin 1st term x). Composition tests to separate recall from composing (computed exactly 2026-10-29): sin(2x) 2nd term −4x³/3; sin(x/2) 2nd term −x³/48; sin(x)·eˣ = x + x² + x³/3 + 0 − x⁵/30. AG's own caveat: these may also be on the internet.

**AG's self-caught problem:** a shared representation does not by itself show "thinking vs memorising" (he memorised the series too). Reframed: test composition of memorised pieces into unseen results. Prior work on latent multi-hop for words: Yang et al. arXiv 2402.16837; Anthropic "Biology of an LLM" (Dallas→Texas→Austin, read only via secondary summaries so far).

**Backlog (AG asked):** read Anthropic's "On the Biology of a Large Language Model" (2025) properly — transformer-circuits.pub / anthropic.com/research/tracing-thoughts-language-model — and check whether our idea overlaps.

**Next step agreed:** a detailed one-page view of Qwen3.5-4B internals (not an abstract block diagram), then design the GPU run with [[overcollect-on-gpu]].
