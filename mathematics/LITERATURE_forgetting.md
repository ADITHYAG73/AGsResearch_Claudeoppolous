# Prior work: does fine-tuning on unrelated data make a model forget mathematics?

Searched 2026-10-05, before any money or design. Two independent passes: Claude (WebSearch + the
arXiv abstract/full-text pages) and Codex CLI 0.159.2 (`codex exec`, live web access confirmed in
its own first line). The two passes were not shown each other's results. Codex's raw answer is
quoted at the end. "Verified" = the abstract or full text was opened this session; it says what
the paper reports, not that the result was reproduced.

## AG's idea, as stated
Benchmark mathematical ability broadly (expressions, composite and nested functions), then
fine-tune for a long time on unrelated data and track what happens to the mathematics. Sharpened
in discussion: do memorised FACTS and PROCEDURES fade at different rates, and how fast does the
ability return on brief retraining (his abacus analogy).

## What is ALREADY DONE

| claim | paper | models | what it shows | verified |
|---|---|---|---|---|
| Maths drops after fine-tuning on other tasks | TRACE, Wang et al. 2023, arXiv 2310.06762 | llama2-chat 13B (others) | "llama2-chat 13B on gsm8k dataset declined precipitously from 28.8% to 2% after training" on its 8-task sequence | abstract page |
| Same, up to 33B | Dong et al. 2023, arXiv 2310.05492 | LLaMA 7B / 13B / **33B** | code→math→general sequential SFT: "Sequential training and mixed sequential training preserve general ability while losing too many specialized abilities" | full text |
| Same, small models | Luo et al. 2023, arXiv 2308.08747 | BLOOMZ, mT0, 1b–7b | forgetting of domain knowledge, reasoning (incl. MathQA), reading; worsens with scale in that range | abstract page |
| Different things forget at different rates | Chang et al. 2024, arXiv 2406.11813 | OLMo 1B / 7B | "the forgetting in compositional generalization is slower (the decay constant a is smaller) than in memorization and semantic generalization" — decay constants 0.25 vs 0.20 (OLMo-7B, mid) | full text |
| Forgetting is uneven across tasks | Zhang, Choshen, Andreas 2024, arXiv 2409.02228 | Llama 2 7B, GPT-J, GPT-2 | forgetting by random-label fine-tuning generalises for some tasks and not others | abstract page |
| "Forgetting" is often loss of task alignment, not knowledge | Zheng et al., ICLR 2025, arXiv 2501.13453 ("spurious forgetting"); Jiang et al. 2024, arXiv 2406.12227 | not stated in abstracts | performance drops "reflect a decline in task alignment rather than true knowledge loss"; instruction-following "the main contributor to forgetting" | abstract pages |
| Forgotten ability comes back with little retraining | Xu et al., ICML 2026, arXiv 2505.16831; plus Lynch 2402.16835, Fan 2502.05374, Qian 2505.09500, Yoon 2602.03379 (Codex-verified) | mostly 7–8B | "original behavior is easily restored through minimal fine-tuning... information is merely suppressed, not genuinely erased" | abstract page (first); Codex (rest) |
| LoRA forgets less than full fine-tuning; rank controls it | Biderman et al. 2024, arXiv 2405.09673 | — | "LoRA better maintains the base model's performance on tasks outside the target domain" | search summary of abstract only |
| Forgetting vs number of update steps | Kalajdzievski 2024, arXiv 2401.05605 | Llama 2 Chat 7B, LoRA | forgetting as a function of steps and trainable parameters | Codex only |
| Removing a maths skill on purpose | Li, Sun, Weng, NAACL 2025, arXiv 2503.21730 | not stated | training-free "skill unlearning" of math-solving, coding, comprehension | abstract page |

## What was NOT found (the possible contribution)
A study that, **inside mathematics and in one model**, separates recalled values from procedure
using a decomposition established beforehand, tracks both densely through carefully screened
non-mathematical fine-tuning, and then compares their relearning against a matched control.
Neither pass found a full match. **That is a search result, not proof of absence.**

## Three things this changes
1. **Scale is not a novelty argument.** Dong et al. already go to 33B. "We did it on 27B" buys
   nothing by itself; the decomposition is the contribution.
2. **The published evidence points BOTH ways, which is the best reason to run it.**
   - Chang et al.: verbatim memorisation decays FASTER than compositional generalisation →
     predicts the recalled fifth decimal dies first and the rough calculator survives.
   - Spurious-forgetting / instruction-vector line: what fades is task alignment, knowledge stays
     → predicts stored values survive and the procedure/usage degrades. AG's abacus analogy, and
     the prediction Claude gave from it, sit on this side.
   Mapping either paper onto "recall vs procedure in trigonometry" is Claude's inference; their
   constructs are not ours (Chang: newly injected fictional facts during continued pretraining).
3. **Two design requirements the literature imposes.** (a) A "savings" claim needs a matched
   never-learned or initial-learning control — fast recovery alone does not show it (Codex's point).
   (b) "Unrelated" needs an operational definition: Yoon et al. 2026 report that syntactically
   similar data without the target content can restore forgotten outputs.

## Source judged NOT reliable
arXiv 2601.18699, "Mechanistic Analysis of Catastrophic Forgetting in Large Language Models During
Continual Fine-tuning", single author. v1 (Jan 2026) lists GPT-5.1, Claude Opus 4.5 and Gemini 2.5
Pro with "estimated" parameter counts and ~35,000 H100/H200 GPU-hours; the current abstract lists a
different model set and "mid-2026". Mechanistic analysis of closed-weight models by one author at
that compute is not credible. Not used. A reviewer may still know it exists.

## Gaps in this search
Question 6 (anything on trigonometric evaluation, recall vs computation) was shallow: only
arXiv 2510.26285, already known to the project, surfaced. Not a clean negative.

---

## Codex's independent answer, verbatim (2026-10-05)

WEB ACCESS: YES (I fetched live pages)

**The broad experiment is not novel. The precise facts-versus-procedures hypothesis may still be.** I found mathematical forgetting after other-task fine-tuning, different decay rates for memorization versus compositional generalization, and recovery after unlearning. I did **not** find a paper combining all your proposed conditions.

Below, **FULL** means the complete proposed experiment, not merely one component. Years refer to first arXiv submission. Every entry is **VERIFIED**: I opened its abstract page and inspected relevant full-text sections. Verification establishes what the paper reports; it does not independently reproduce its results.

**A. Mathematical degradation after fine-tuning on other data**

1. **[How Abilities in Large Language Models are Affected by Supervised Fine-tuning Data Composition](https://arxiv.org/abs/2310.05492)**  
   **First author:** Guanting Dong. **Year:** 2023. **arXiv:** 2310.05492. **VERIFIED.**  
   **Models:** LLaMA, **7B, 13B, 33B**.  
   **Measured:** GSM8K mathematical reasoning, code generation, and general instruction-following performance under different data mixtures and sequential training orders, including mathematical training followed by general ShareGPT training.  
   **OVERLAP = PARTIAL.** This directly anticipates losing mathematical ability through subsequent general fine-tuning, including at 33B, but does not separate mathematical facts from procedures or measure their forgetting and recovery curves.  
   **Important qualification:** ShareGPT contains some mathematics and code; the paper also investigates filtering these out, but its main sequential-training results should not be described as a strictly math-free interference experiment. Evaluation uses final checkpoints after three epochs. [Experimental details](https://arxiv.org/html/2310.05492v3)

2. **[An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning](https://arxiv.org/abs/2308.08747)**  
   **First author:** Yun Luo. **Year:** 2023. **arXiv:** 2308.08747. **VERIFIED.**  
   **Models:** BLOOMZ **1.1B, 1.7B, 3B, 7.1B**; mT0 **1.2B, 3.7B**, as reported; LLaMA and Alpaca **7B**; BLOOM comparisons.  
   **Measured:** Changes in domain knowledge, reasoning—including **MathQA**—and reading comprehension across sequential fine-tuning on simplification, empathetic dialogue, question generation, explanation generation, and headline generation.  
   **OVERLAP = PARTIAL.** It measures mathematical performance during unrelated-task fine-tuning, but uses task-boundary checkpoints and smaller models, without a facts/procedures decomposition or relearning phase.  
   **Important qualification:** MathQA is evaluated as zero-shot multiple-choice accuracy; that is weaker evidence about execution of arithmetic procedures than controlled carrying/sign-rule tests. [Methods and models](https://arxiv.org/html/2308.08747)

**B. Differential forgetting of memorization, knowledge, and skills**

3. **[How Do Large Language Models Acquire Factual Knowledge During Pretraining?](https://arxiv.org/abs/2406.11813)**  
   **First author:** Hoyeon Chang. **Year:** 2024. **arXiv:** 2406.11813. **VERIFIED.**  
   **Models:** OLMo **1B and 7B**, using intermediate pretraining checkpoints.  
   **Measured:** Stepwise acquisition and forgetting of injected fictional knowledge, separately testing exact-sequence memorization, semantic generalization, and compositional generalization, with fitted decay constants.  
   **OVERLAP = PARTIAL.** This is the strongest conceptual overlap: the authors report that compositional generalization generally forgets more slowly than memorization, but their composition tests combine factual statements rather than execute mathematical algorithms.  
   **Why it matters:** You cannot safely claim that comparing memorization and generalization decay rates is new; the narrower mathematical procedure comparison remains distinct. [Results, especially Table 2](https://arxiv.org/html/2406.11813). I also inspected the [author-linked repository](https://github.com/kaistAI/factual-knowledge-acquisition/).

4. **[Unforgettable Generalization in Language Models](https://arxiv.org/abs/2409.02228)**  
   **First author:** Eric Zhang. **Year:** 2024. **arXiv:** 2409.02228. **VERIFIED.**  
   **Models:** Llama 2 **7B**; additional experiments with GPT-J **6B** and GPT-2 **124M**.  
   **Measured:** Within-task and cross-task forgetting after randomized-label fine-tuning across 21 tasks, including MathQA, plus example-level forgetting/relearning order and residual task information in linear probes.  
   **OVERLAP = PARTIAL.** It studies unequal susceptibility of skills to forgetting and includes a fine-tune–forget–fine-tune sequence, but does not compare mathematical factual recall against algorithmic procedures under ordinary unrelated-data training.  
   **Why it matters:** Both heterogeneous forgetting and a subsequent relearning phase already exist; moreover, degraded outputs can coexist with recoverable internal task information. [Experiments, §§3–7](https://arxiv.org/html/2409.02228)

5. **[Scaling Laws for Forgetting When Fine-Tuning Large Language Models](https://arxiv.org/abs/2401.05605)**  
   **First author:** Damjan Kalajdzievski. **Year:** 2024. **arXiv:** 2401.05605. **VERIFIED.**  
   **Model:** Llama 2 Chat **7B**, fine-tuned with different LoRA configurations.  
   **Measured:** Forgetting as a function of update steps, fine-tuning loss, and trainable parameter count, principally using output-distribution changes, with additional knowledge, reasoning, and safety evaluations.  
   **OVERLAP = PARTIAL.** It anticipates measuring forgetting trajectories against training duration, but does not isolate arithmetic or compare factual and procedural decay and recovery.  
   **Qualification:** Its reported scaling relationship is not an established scaling law for mathematical accuracy. [Methods and results](https://arxiv.org/html/2401.05605v1)

**C. Relearning and reversal of unlearning**

6. **[Eight Methods to Evaluate Robust Unlearning in LLMs](https://arxiv.org/abs/2402.16835)**  
   **First author:** Aengus Lynch, with equal-first-author collaborators. **Year:** 2024. **arXiv:** 2402.16835. **VERIFIED.**  
   **Models:** Llama 2 Chat **7B** and its Harry Potter-unlearned “Who’s Harry Potter” derivative.  
   **Measured:** Recovery of Harry Potter knowledge through multiple interventions, including LoRA fine-tuning on **800 or 8,000 sentences**, alongside trivia tests and latent-knowledge probes.  
   **OVERLAP = PARTIAL.** It directly tests recovery after apparent forgetting using limited retraining, but concerns deliberate factual unlearning rather than mathematical interference and differential recovery rates.  
   **Qualification:** The paper reports that 8,000 sentences bring the unlearned model close to the original baseline; it does not establish complete recovery from just a handful of examples. [§3 and Appendix A.2](https://arxiv.org/html/2402.16835)

7. **[Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware Minimization Perspective and Beyond](https://arxiv.org/abs/2502.05374)**  
   **First author:** Chongyu Fan. **Year:** 2025. **arXiv:** 2502.05374. **VERIFIED.**  
   **Models:** Zephyr **7B-beta**, Llama 3 **8B**, news-fine-tuned Llama 2 **7B**, and Harry Potter-fine-tuned ICLM **7B**.  
   **Measured:** Recovery after unlearning on WMDP and MUSE while varying relearning epochs and sample counts, including **20-example** experiments and fine-tuning on forget-unrelated datasets.  
   **OVERLAP = PARTIAL.** It directly anticipates recovery curves and limited-data retraining, but does not study mathematical facts versus procedures after prolonged ordinary fine-tuning.  
   **Why it matters:** “Forgotten ability returns after a little fine-tuning” is already an established experimental topic. [Experiments](https://arxiv.org/html/2502.05374); [author-linked repository inspected](https://github.com/OPTML-Group/Unlearn-Smooth).

8. **[Layered Unlearning for Adversarial Relearning](https://arxiv.org/abs/2505.09500)**  
   **First author:** Timothy Qian. **Year:** 2025. **arXiv:** 2505.09500. **VERIFIED.**  
   **Models:** Zephyr **7B-beta**; synthetic logistic-regression and small attention-only transformer experiments.  
   **Measured:** Recovery on held-out portions of forgotten datasets after retraining on other portions, tracking accuracy over epochs on WMDP, MMLU subsets, and historical-event years.  
   **OVERLAP = PARTIAL.** It tests how relearning generalizes beyond the examples rehearsed—including numerical factual answers—but does not compare those facts with mathematical procedures under incidental forgetting.  
   **Why it matters:** A recovery study should distinguish reacquiring rehearsed answers from recovering a broader capability. [Experimental protocol](https://arxiv.org/html/2505.09500); [author-linked synthetic repository inspected](https://github.com/12tqian/layered-unlearning).

9. **[Rethinking Benign Relearning: Syntax as the Hidden Driver of Unlearning Failures](https://arxiv.org/abs/2602.03379)**  
   **First author:** Sangyeon Yoon. **Year:** 2026. **arXiv:** 2602.03379. **VERIFIED.**  
   **Models:** Llama 2/Chat **7B**, Zephyr **7B-beta**, Llama 3 Instruct **8B**, and an additional TOFU-fine-tuned Phi-1.5 experiment—**1.3B** according to its [official model card](https://huggingface.co/microsoft/phi-1_5).  
   **Measured:** Forgotten-information recovery across unlearning and relearning checkpoints, comparing topically related data with syntactically similar data that omit the target information.  
   **OVERLAP = PARTIAL.** It closely anticipates checkpoint-dependent recovery and shows that apparently unrelated data can restore factual outputs, but its forgetting is deliberate unlearning and it does not test mathematical procedures.  
   **Why it matters:** “Unrelated” needs an operational definition: topical separation alone does not rule out structural overlap that affects recovery. [Methods and results](https://arxiv.org/html/2602.03379)

**What appears already done:** Mathematical degradation after other-task fine-tuning, including at **33B**; forgetting curves versus training steps; unequal forgetting across capabilities; slower decay of compositional generalization than memorization in a factual-knowledge setting; and recovery after unlearning through limited retraining. The papers above rule out claiming any of those broad observations as your novelty.

**What appears open:** A controlled, tool-free study in the same tens-of-billions-parameter model that separately measures **mathematical fact recall and execution of mathematical procedures**, follows both densely through prolonged, carefully screened non-mathematical training, and compares their recovery curves afterward. I found no **FULL** match, but that is a search result, not proof of absence.

**Blunt assessment:** “Fine-tune away from math and see whether math comes back” is too incremental by itself. The defensible contribution is a convincing separation of facts from procedures and their forgetting/relearning dynamics. For a claim of **savings**, rapid recovery alone is insufficient: you need a matched initial-learning or never-learned control showing that prior learning actually reduced the retraining required.