# CLAUDE.md — Research Context

> Root of the repo. Edit anything wrong or stale — this file is mine, not fixed.
> Update **Section 9: Current State** at the end of every session. That is what makes
> picking this up again after three weeks cheap.

---

## 1. Who I am

- Based in India (Tamil Nadu, Kumbakonam).
- BE, Instrumentation & Control Engineering (2020).
- M.Tech, Data Science & Analytics (2023).
- ~11 months at MBRDI (first corporate role, limited learning).
- **Karini AI** (startup), July 2024 – Jan 2026 — where the bulk of my real learning happened.
- Active Kaggle competitor.

I am a committed student of mathematics — not a trained pure mathematician, and I don't
claim to be. My work sits in applied mathematics. I don't accept a result I can't explain
mechanistically, and I'd rather understand the underlying structure than take a working
black box on trust.

## 2. Where I want to go

Long-term: research at a frontier lab, Anthropic specifically, in interpretability or
alignment. Medium-term: a funded PhD (Australian universities with a living stipend have
been the main route considered — revisit whether that's still best). Near-term:
**produce legible public research output**, because that is the actual bottleneck.

Verified against primary sources (Aug 2026):

- Anthropic careers page: about half their technical staff had no prior ML experience;
  about half have PhDs; some never went to college. Independent research, blog posts and
  open-source contributions belong at the top of a resume.
  → https://www.anthropic.com/careers
- Same page: people with an engineering background should **apply as an engineer** — they
  perform better in the interviews, and Anthropic papers have engineers as authors, often
  first author. **Given my Karini AI profile, Research Engineer is the door, not
  Research Scientist.**
- Neel Nanda, "How To Become A Mechanistic Interpretability Researcher" (Sept 2025):
  mech interp is learnable on your own with short feedback loops and *modest compute*.
  Recommended progression: learn the ropes (≤1 month) → 1–5 day research mini-projects →
  1–2 week sprints.
  → https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher

Caveat worth holding: Neel has publicly moderated his optimism about the most ambitious
version of mech interp, favouring a layered "Swiss cheese" safety picture over any single
solution. Source is a podcast summary, not a paper — weight accordingly, but know it.

## 3. What I actually know (use this; don't re-explain basics)

Strong, from production experience:
- Production GenAI end to end — agentic systems, LLM integrations, conversational backends.
- AWS. Agents and workflows. LangGraph, deep agents.
- Multiple foundation model families and their SDKs.
- Agentic RAG, graph RAG. Evals.
- NLP-heavy. RL background; TRL, HuggingFace ecosystem.

Weaker / self-taught: diffusion models (interest, not depth).

Comfortable with dense technical material and jargon across ML, RL, biology.

## 4. How to work with me — epistemic rules

I care about this more than about being encouraged.

- **No sycophancy.** Don't soften a real problem. Disagree when I'm wrong.
- **No hallucination.** If you don't know, say so. Distinguish *sourced fact* from
  *your inference* explicitly, in the same sentence.
- **Primary sources only** for research claims: the official paper, the official repo,
  the official docs, the authors' own writing. Third-party blogs are acceptable for
  applied AI, not for research claims.
- **Never adopt a term I use without checking it.** If I use unfamiliar vocabulary,
  look it up or ask. Repeating my words back manufactures fake shared understanding.
  (This already happened once — I typo'd a project name and it got echoed back to me
  three times as though it were real.)

## 5. Research interest

Mechanistic interpretability. Aware SAEs came under scrutiny after DeepMind's 2025
negative results, and that transcoders and circuit-tracing / attribution methods have
gained ground. Reference frame: Sharkey et al., "Open Problems in Mechanistic
Interpretability" (arXiv 2501.16496).

Adjacent: RLHF, AI for science, hallucination detection in agentic tool chains,
uncertainty propagation.

## 6. Prior work, and where it stopped

**Natural Language Autoencoders (NLAs)** — Anthropic / Transformer Circuits, May 2026.
Fraser-Taliente, Kantamneni, Ong et al. Forked `kitft/natural_language_autoencoders`.

What the method actually is (checked against the paper, because I had this wrong):
- An NLA is a pair of modules: an **activation verbalizer (AV)** mapping an activation to
  a text explanation, and an **activation reconstructor (AR)** mapping that text back to
  an activation. Round trip: activation → text → reconstructed activation.
- AV and AR are **initialised as copies of the target model** and **jointly trained with
  reinforcement learning** to reconstruct residual-stream activations.
- **Not SFT.** I had planned "SFT on the AV," which is not the paper's procedure. The RL
  training loop is the real thing and is where the debugging pain lives.

Links: https://transformer-circuits.pub/2026/nla/ ·
https://github.com/kitft/natural_language_autoencoders

**Repo layout (verified 2026-08-22):**
- `../natural_language_autoencoders/` — my fork (`ADITHYAG73/...`, upstream `kitft/...`),
  a **sibling of this repo, NOT inside it.** Holds the pod skill, `tutorials/`, Objective 2 draft.
- `reference_repos/EasyNLA/` — Celeste's training framework (`asherps/EasyNLA`, MIT). Read-only.
- `reference_repos/nanoNLA/` — predecessor (`ceselder/nanoNLA`), marked superseded by EasyNLA.

What I did: got inference working on small Gemma-2 and Qwen models, debugging issues one
at a time. Then targeted Gemma 4 and hit the per-layer embedding architecture on the
smaller variants — not a dense network, needed rethinking.

**It stalled because I ran out of RunPod credit, not because I ran out of ideas.**

An old Claude Code session in VSCode holds the full Gemma inference debugging history.
That is a blog post that already mostly exists and costs zero GPU-hours. See Objective 2.

## 7. Constraints (real ones)

- **Self-funded compute.** No institutional credits. RunPod, billed by the GPU-hour.
- No PhD, no academic affiliation, no supervisor.
- India-based; most programs are US/UK. Visa support matters.

## 8. Operating principles — read every session

These exist because I have a documented pattern of starting strong and not finishing.

**a. Scoping, not discipline.** My projects die from completion conditions outside my
resource envelope ("release a trained checkpoint" when I can't fund the GPU hours). No
willpower closes a funding gap. State the completion condition before starting and check
it's reachable with resources I have *today*. If not, shrink it.

**b. The GPU is a batch service, not a workspace.** Never think on the meter. Write the
script offline → dry-run on dummy examples locally → spin up → run → dump artifacts to the
network volume → **terminate**. Analysis happens off-GPU. Persist environment and data on
a RunPod network volume (per GB-month) so cold start is minutes.

**c. Cache activations.** They're the expensive artifact and they're idempotent. Once on
disk, weeks of analysis run on CPU. Never regenerate what's on the volume.

**d. Free compute first.** Colab (free GPU — Neel's own recommended environment).
Kaggle notebooks (verify current free GPU-hour quota myself). **TPU Research Cloud** —
free Cloud TPUs, application-based, open to non-academics, in exchange for publishing the
work. → https://sites.research.google/trc/ — apply; it's an afternoon.

**e. I own the research questions.** Claude writes scripts, debugs CUDA, manages pods,
handles plumbing. I decide what's worth measuring and what a result means. If I start
passively deferring on direction, say so.

**f. Finish small things.** A written-up negative result beats an unfinished ambitious one.

## 9. Current objectives

**Objective 1 — MATS Winter 2027, Neel Nanda stream. Deadline: Fri 4 Sept, 11:59pm PT.**
The application is a weekend research task. Fresh directory, nothing carried over. The task
itself is the finished artifact — even with no reply, I end with completed research work.
No reply is the modal outcome given volume and says close to nothing about me.
(Exploration: 28 Sept – 30 Oct, remote, $4.2k stipend + $500 compute. Research phase:
19 Jan – 10 Apr, Berkeley, J1 visa + travel support. Other streams open late August —
apply to those too.) → https://www.matsprogram.org/apply

**Objective 2 — extract the Gemma per-layer-embedding debugging writeup.** From the old
VSCode session. Zero GPU cost. Ship this week.

**Objective 3 (candidate) — train and submit an NLA to Neuronpedia.** Neuronpedia lists
contribution routes that don't require deep experience, including submitting your own
trained NLAs. Completion condition sized to my resources, public artifact at the end.
→ https://www.neuronpedia.org/blog/nlas

**Not now:** restarting the Gemma 4 NLA training run. Over-scoped, already beat me once.

## 10. Current state

*(update every session — date, what changed, what's blocked, what's next)*

- **2026-08-17** — Repo initialised. Objectives set. Nothing run yet.
- **2026-08-18** — Read the Neel Nanda MATS 12.0 admissions doc end to end (all six
  tabs) plus the Airtable form. Notes in `mats_2027/neel2027.md`. Key corrections to
  earlier assumptions:
  - **Deadline is Sept 4, but extensions are available to Sept 11.** §9 had it as hard.
  - **Project need not be interpretability** — "interpretability and non-interpretability
    projects are both fine, so long as I think it's interesting." His interests have
    broadened to model forensics, model biology, science of post-training, alignment
    training, science of generalization.
  - **Two of his listed disqualifiers hit the prior NLA plan:** "only studying old models
    (GPT-2, Pythia, Gemma 2)" and "SAE hill-climbing / basic science of SAEs". He is
    explicitly pessimistic about ambitious interpretability (complete reverse-engineering),
    which is closer to the §5 Sharkey et al. framing than to what he wants.
  - **His current recommended models:** Qwen 3.5/3.6 dense (4B/9B/27B) as defaults;
    deepseek v4 flash 0731 for a highly capable target; Gemma 3 + Gemma Scope 2 for SAEs.
    Internals via `nnsight` or raw PyTorch hooks.
  - **He recommends against Colab** — rent a cloud GPU. Contradicts §8d. RunPod
    recommended, vast.ai noted as notably cheaper; preemptible cheaper still.
  - **The form's summary questions are the primary filter**, read before any write-up.
    Write-up is a Google Doc, anyone-with-link, first 1–3 pages an exec summary,
    max 600 words, graphs expected.
  - **Time accounting:** 20h + 2h for exec summary and form. Reading this admissions doc,
    GPU setup, breaks, training wait time and filling the form do NOT count. Paper reading
    chosen for the project DOES, capped at ~5h by his advice. A doomed project may be
    abandoned and the timer reset.
  - **"Sanity-check your agent" is the single most weighted piece of advice in the doc.**
    Design and baselines must be mine; every load-bearing number needs an independent
    check I wrote myself (if Claude computes both the number and the check, the check is
    worthless — correlated errors). Last round, agentic LLM users were accepted at ~3x the
    rate of writing-polish-only users, so agentic use is table stakes and the variance is
    in question choice and verification. Raw/LLM-sounding prose in the form or exec
    summary is a stated significant negative signal.
  - **Q8 (evidence you can do good research) explicitly bans citing the application
    project.** The two live blog posts (free norm, flash attention) are the answer —
    he names blog posts as valid non-standard credentials.
  - Decision taken: **learn raw PyTorch hooks myself.** Small (a page of API), and it
    falls inside general learning time so it costs zero of the 20 hours. Claude still
    writes the experiment harness; I write the verification code.
  - Downloading his ~600k-token mech interp context bundle from Drive into the repo.

  **Blocked on nothing. Next: read the Recommended Research Problems tab properly and
  narrow to three candidate questions, filtered by "can I get real traction on this in
  20 hours on one rented GPU" — not by which area matters most.**

- **2026-08-18 (later)** — Downloaded his context bundle to
  `mats_2027/Mech Interp Context Docs/` (24MB, includes the admissions doc as .docx,
  which is the cleanest source for its text). Read the **Recommended Research Problems**
  tab properly from the .docx. **This reverses the entry above:**
  - **NLAs are explicitly on his excited list.** Under "Improved Interpretability
    Methods" he names J-Lens and natural language autoencoders, and says he is
    "especially curious about natural language autoencoders". My earlier claim that the
    NLA work sat in a deprioritized area was wrong — it came from a summarizer model
    that dropped this section. The Gemma-2 point stands but is moot (see next).
  - **A trained NLA is available: Celeste's (his scholar) Qwen 3.6 27B NLA**, open
    source + Neuronpedia demo. So the §6 blocker — RunPod credit for joint AV/AR RL
    training — is off the critical path. Inference only, which §6 already has working.
  - **His named direction:** use the activation reconstructor to measure description
    quality — find which claims can be removed to *improve* reconstruction accuracy, as
    a way to reduce hallucination in descriptions. He says this was only briefly explored.
  - Caveats: he is "moderately more cynical" about meta-models than he was; and he heads
    the list with a warning that these ideas are NOT filtered for 20-hour feasibility.
    Also "do not trust LLM time estimates, in my experience they're super off."
  - Other live candidates: red-teaming the NLA against baselines; eval-awareness probe
    false-positive base rates (he asks this near-verbatim; suggests Nemotron 49B);
    J-Lens (open lenses on Qwen 3.5 4B through deepseek v4 flash); model diffing.

  **Leading candidate: reconstruction-guided hallucination detection in NLA descriptions
  on Celeste's Qwen 3.6 27B NLA. Baselines needed: random claim removal, length-based
  removal, LLM judge. Next: pin the hypothesis down myself before any code.**

- **2026-08-18 (later still)** — Located the concrete artifacts and the prior work.
  - **Celeste's NLA = `ceselder/qwen3.6-27b-nla-rl` on HF.** Qwen3.6-27B, **layer 42**
    residual stream. Contains `av_base/` (base + warmstart LoRA merged),
    `av_rl_adapters/` (GRPO-RL LoRAs every 100 steps to 600), and crucially
    **`ar_reconstructor/`** (43-block backbone + value head). Call path: explanation text
    → reconstructor → value head → predicted activation → FVE. Built with
    [EasyNLA](https://github.com/asherps/EasyNLA) (sits on nanoNLA). License "other" —
    read before publishing. **She recommends step 300 (75.6% FVE)**, not final (~78%).
  - **The reconstruction metric is FVE** (fraction of variance explained). She notes FVE
    is NOT comparable across base models — keep all numbers inside this one NLA.
  - **Compute:** 27B bf16 ≈ 54GB weights + AR + activations → wants a single 80GB card.
  - **Prior work is the NLA paper's "Characterizing NLA confabulations" section.** Their
    method: Opus 4.6 NLA on pretraining-like text, **Haiku 4.5 extracts verifiable claims
    and judges validity/specificity** (judge reliable but miscategorizes sometimes).
    Result: Theme 64% true, Entity 28% true, Detail 24% true; higher-level claims more
    often supported, false claims usually somewhat relevant. They explicitly flag that
    since verifiable claims can't be taken literally, the same worry applies to
    **unverifiable claims about model cognition** — and offer no method for those.
  - **The gap (this is the project):** the paper's method needs ground truth, so it only
    works on claims about input context. Neel's suggested reconstruction-delta method
    needs no ground truth, so it could extend to unverifiable claims — but is unvalidated.
    The paper's verifiable claims are a **labelled test set for a label-free metric.**
  - **Baselines, one handed over free by their own data:** claim type alone predicts
    falsity (Theme 64% vs Detail 24%), plus random removal and length-based removal.
  - **Limitation to bank for Q6:** their numbers are for the Opus 4.6 NLA; this would be
    an open Qwen3.6-27B NLA at ~75% FVE. Transfer untested.

  **Next: I pin down the hypothesis and its kill condition myself before any code.
  Working version to pressure-test — reconstruction-delta on claim ablation ranks
  confabulated claims as well as or better than an LLM judge with access to the input
  text. Clock not started.**

- **2026-08-18 (correction, read the NLA paper source directly)** — Fetched
  `transformer-circuits.pub/2026/nla/index.html` and grepped it. Two corrections:
  - **Terminology:** the paper uses *confabulation* and *hallucination*
    interchangeably for the same phenomenon — the section "Characterizing NLA
    confabulations" is referred to in-text as measuring hallucination rates. Neither term
    is defined by whether the model knows the claim is false. The axis they use is
    **verifiability**: false claims about the input context are checkable; claims about
    model processing are not. Operational definition to use: **confabulation = verifiably
    false claims about the target model's input context.**
  - **The paper ALREADY RAN the claim-ablation experiment** — subsection "Using the AR as
    a verifier". They delete individual claims and measure Δ reconstruction **MSE**
    (note: MSE here, FVE for training). Findings: removing *true* claims hurts MSE more
    than removing false ones; removing context-relevant false claims hurts more than
    unrelated ones; trends hold in aggregate but are **noisy per transcript, so the AR is
    only a weak per-claim verifier.** They also report two further weak signals:
    specificity (theme > entity > detail) and **recurrence across token positions** (run
    the NLA over the last 10 tokens, count how often a final-token claim appears earlier).
    They state the results cover claims about input text only, not model cognition.
  - **So the naive project is done.** What remains open, and is now the actual gap:
    (a) the AR is a *weak* per-claim verifier — nobody has tried to make it strong
    (ablation design: delete vs replace, control for claim length/bullet structure,
    average over paraphrases); (b) **they never combine the three signals** into one
    predictor, and their single-signal results are the baselines to beat; (c) Neel's
    phrasing ("removed to *improve* reconstruction") is stronger than the paper's finding
    ("false claims hurt *less*") — whether removal ever genuinely improves reconstruction
    looks open; (d) transfer to the open Qwen3.6-27B NLA is untested.
  - Also: paper's Future Work suggests penalizing context-inconsistent statements during
    NLA RL — relevant prior-art-of-intent, and out of scope (no training).

- **2026-08-18 (NLA method details, verified against the paper)** —
  - **AV architecture:** same architecture as target M. Fixed prompt instructing it to
    describe an activation; the activation is injected **at a special token between
    `<concept>` tags** (`<concept>A</concept>`), replacing that token's embedding.
    Activations are **L2-normalized then scaled by a constant α** (Appendix heuristic:
    ~75th-percentile activation norm at the layer; the AV is sensitive to α). Output
    wrapped in `<explanation>` tags, sampled at **T=1**.
  - **AR architecture:** target model **truncated to its first ℓ layers**, plus a
    **learned affine map** on the layer-ℓ activation at the **final token**.
    *Cross-check that holds:* Celeste's card says `ar_reconstructor/` = 43-block backbone
    + value head at layer 42 → 43 blocks for ℓ=42, value head where the paper puts the
    affine map. Paper and artifact agree.
  - **The AV never sees the input context** — only the single activation. The paper leans
    on this (it is "structurally less able" to add inferences beyond the activation).
    Confabulation instead comes from **"excessive expressivity"**: the AV is a full LLM
    and can infer beyond what the activation stores.
  - **Three different "50"s in the paper, none a signal threshold:** (a) eval-awareness
    protocol samples **50 random response tokens**, flags the transcript if *any* is
    graded aware — they call this aggregation **arbitrary** and the metric **uncalibrated**,
    relative-only; (b) recurrence analysis uses the **last 10 tokens**; (c) the PastLens
    baseline reproduces the preceding N tokens, **N ~ U[1,50]**.
  - **No claim found that early token positions carry less signal.** The early/late
    finding in the paper is about **layers**: midpoint-layer NLAs surface reward-model
    sycophancy terms, later-layer ones don't (consistent with Lindsey et al.). Layer 42
    of 64 in Celeste's NLA is ~2/3 through, matching "middle-to-late layer".
  - **Steganography check (useful for ablation design):** they paraphrase explanations and
    shuffle bullet points and measure FVE change; find little evidence of steganography.
    Justifies averaging ablation results over paraphrases to cancel formatting noise.

- **2026-08-19 — judge reliability is an unvalidated dependency (project-critical).**
  The confabulation section's ground truth comes from **Haiku 4.5**, used both to extract
  verifiable claims and to judge validity/specificity. The paper's only statement about it:
  generally reliable, miscategorizes some of the time. **No agreement number, no
  hand-grading reported.**
  - Contrast, same paper: the **eval-awareness grader** was validated (2 authors hand-graded
    **186** explanations, **97% agreement**); the **auditing-report grader** was calibrated
    (2 authors hand-graded **22** reports, rubric tuned then frozen). The standard exists in
    the paper and was not applied to the confabulation judge.
  - Why Haiku is a defensible choice anyway: verification (is this claim supported by this
    text?) is far easier than the AV's generation task, and it runs at scale cheaply. The
    paper's own judge choices track task difficulty — they used **Opus 4.7** as judge for
    the subtler realism-edit comparison. Grader snapshot in the embedded data is
    `claude-haiku-4-5-20251001`.
  - **Risk to our project:** those verdicts are the ground truth for validating a label-free
    metric. If judge error rate varies **by claim level**, it contaminates the specificity
    signal — signal and label would share a common cause. That is a confound, not noise.
  - **Cheapest first experiment, API-only, no GPU:** re-judge a stratified sample with a
    stronger model, hand-label some myself, measure agreement per claim level. Doubles as
    the "look at your own data" evidence Neel explicitly rewards.

- **2026-08-19 (effect size — the ablation figure is weaker than it looks).**
  - **Units differ, no inconsistency:** the 64/28/24 chart = % of claims that are true; the
    0.25/0.09 chart = **percentage points of FVE lost when one claim is deleted**. Small is
    expected — one bullet out of a ≤256-token explanation reconstructing a 5120-dim vector.
  - **Error-bar overlap (read off the figure, approximate):** Theme true-vs-false 6% of the
    narrower bar; Detail 62%; Entity 80%; **Related-vs-Unrelated (0.14 vs 0.06) = 100%** —
    the Unrelated interval ~[0.00,0.21] fully contains the Related mean. That panel's
    headline is close to unsupported by its own figure.
  - **The error bars are undefined.** No SE, no CI, no n stated for that figure anywhere in
    the text. (They DO define SE for the reward-steering figure and give 95% CIs in the
    realism-edits table — so the standard exists in the paper and wasn't applied here.)
  - **The key distinction:** detecting a *mean* difference across many claims (Task 1 — they
    did this) vs classifying *an individual* claim (Task 2 — needs the distributions to
    separate). **Our project is Task 2; their evidence is Task 1.** That gap is the opening.
  - **Back-of-envelope sizing** (11 fennec claims, ΔFVE ≈ Δmse% × (1−FVE), FVE assumed 0.78
    from a *different* NLA — order-of-magnitude only, NOT a measurement):
    per-claim spread −0.57 to +0.23pp ≈ **4.7× the ~0.17pp effect**; **d ≈ 0.71**;
    **per-claim AUC ≈ 0.69** (chance 0.50).
  - **Lever + prediction:** averaging Δmse over K paraphrases/resamples shrinks independent
    noise as 1/√K → d grows as √K → AUC 0.84 at K=4, 0.93 at K=9. Justified by their
    steganography result (AR responds to meaning, not surface form).
  - **KILL CONDITION (clean):** if variance does NOT fall as ~1/√K, the noise is systematic
    rather than stochastic, averaging cannot rescue it, and the approach is dead — which is
    itself a fast, reportable negative result.
  - **Hypothesis reframed:** the open problem is **signal-to-noise, not existence**. Target
    metric = per-claim AUC. Still mine to write, including the bet on whether the noise is
    stochastic or systematic.

- **2026-08-19 — FIRST HYPOTHESIS (mine, written out) + its rivals.**
  - **H:** *The AR's "related false claims still matter" effect exists because a substantial
    fraction of related-false claims are NOT confabulations — they are faithful readouts of
    the activation that the text-based judge mislabels as false.* (= the 2x2 top-right cell.)
  - **It follows from the paper's own stated principle:** "claims that don't reflect the
    activation should, in theory, contribute little to reconstruction." Run backwards:
    related-false claims DO contribute → they DO reflect the activation → activation-true,
    text-false. Example: the model reading "Annals of the Jos" plausibly represents the
    Joseon Annals *including its Korean name*; the AV reads that out; the text-judge, seeing
    no such string, calls it false.
  - **Consequence if true:** the AR's apparent weakness is substantially a **labelling
    artifact**, not a method failure. Predicts related-false claims behave like TRUE claims
    on activation-grounded measures and like FALSE claims only on text-grounded ones —
    a check the paper never ran.
  - **RIVALS (must be killed first — two are simpler than mine):**
    1. **Lexical overlap.** The claim carries words/concepts ("Joseon", "Korean historical
       records") that ARE in the activation; deleting it deletes that content regardless of
       whether the assertion was encoded. Claim as *vehicle*, not as truth.
    2. **Coherence.** The AR is an LM reading prose; excising a topically-related bullet may
       hurt coherence, while an unrelated bullet is a non-sequitur whose removal helps flow.
    3. **Length / position artifacts.** Related-false claims may just be longer, or sit
       elsewhere in the bullet list.
  - **Discriminating controls, cheapest first:** (a) **length matching** — if length explains
    it, stop; (b) **delete vs replace** with neutral filler of matched length — if the effect
    vanishes it was coherence/length; (c) **lexical control** — rewrite a related-false claim
    keeping its vocabulary but flipping the assertion; effect survives → Rival 1, effect
    collapses → my H.
  - **Self-check flagged:** I have said I *want* the related/unrelated result to be wrong, and
    this hypothesis conveniently makes it "not wrong, just mislabelled". Comfortable landing
    → be maximally suspicious. Run the length control first.

- **2026-08-19 — NORMALIZATION QUESTION RESOLVED (read the EasyNLA source). Big correction.**
  **EasyNLA does NOT use the paper's injection scheme.** From `nla/injection.py`
  (`karvonen_inject_in_residual`, cited to Karvonen et al. 2025 Activation Oracles eq. 1),
  called via `nla/utils/hooks.py::register_karvonen_hook`:

      h'_p = h_p + ||h_p|| * v / ||v||

  | | paper | EasyNLA / Celeste |
  |---|---|---|
  | operation | **replace** the embedding | **add** to the residual |
  | where | embedding layer (0) | **output of transformer block 1** (`layer_idx=1`) |
  | scale | fixed constant **α** | **‖h_p‖** at that position, computed live |
  | pre-norm | unit L2 then ×α | **raw vector in**; the hook normalizes |

  - **So `norm: none` is correct and REQUIRED** — store raw, pass raw. The hook divides by
    ‖v‖ itself. Docstring is explicit: *"Vectors should be RAW (no injection_scale
    normalization) — this function does its own norm match."*
  - **The paper's α heuristic has NO analogue here. Do not apply it** — pre-scaling then
    gets renormalized against a different quantity. Silently wrong, looks like a result.
  - **Supersedes my earlier explanation** that the activation replaces the special token's
    embedding at layer 0 — that is the *paper's* mechanism, not Celeste's.
  - **`mse_scale` defaults to `sqrt(d_model)`** when absent from the sidecar (it is absent).
    Per config comments that scales BOTH prediction and gold to L2 √d before MSE ⇒ **the
    loss is direction-only; magnitude is normalized away.** FVE here = directional
    agreement. **Confirm directly** — a CLI override would not appear in `run_config.yaml`.
  - **Free smoke test:** injection module says a failed/mispositioned injection makes the
    model see the literal marker char and **output Chinese**. AV emitting CJK ⇒ injection
    broken. Zero-cost day-zero check.
  - **Marker neighbours explained:** the left/right check exists to **reject false positives**
    where the marker char appears in ordinary response text. Use the **id (158983)**, not the
    glyph — the code comment shows a different character than the sidecar does.
  - **Do not reimplement injection.** The repo calls it "the most correctness-critical path
    in NLA" and ships unit tests (`tests/`). Use their function.
  - **Created `mats_2027/hypotheses.md`** — hypothesis tracker (H1, H2, rivals, kill
    conditions, standing methodological commitments).

- **2026-08-19 — the OFFICIAL Anthropic release exists and I already forked it.**
  `kitft/natural_language_autoencoders` (Kit Fraser-Taliente, paper co-first author),
  **Apache-2.0**, 924 stars, last push 2026-08-02. My fork `ADITHYAG73/...` is 23 ahead,
  2 behind. **The 2 upstream commits are RL/KL-only (`--kl-loss-type k2`, comment trims) —
  they do not touch inference.** Update branch; NEVER "Discard 23 commits" (that is the
  Gemma debugging history = Objective 2).
  - **Eight released checkpoints (AV+AR × 4 base models):** Qwen2.5-7B-Instruct L20/28
    d=3584 · Gemma-3-12B-IT L32/48 d=3840 · Gemma-3-27B-IT L41/62 d=5376 ·
    Llama-3.3-70B-Instruct L53/80 d=8192. Collection: `kitft/nla-models`.
  - **`kitft/nla-inference`** — standalone inference-only package (single-file actor client,
    no training deps, SGLang `input_embeds`, examples with per-token MSE). Apache-2.0.
  - **CONFIRMED OFFICIALLY (closes the open item):** "Both vectors are L2-normalised before
    comparison, so MSE(reconstructed, original) = 2(1 − cos) measures **direction agreement
    only**." So FVE/Δ are directional; magnitude is invisible. State in the write-up.
  - **Injection divergence confirmed from both sides:** official = *inject the vector as a
    single token embedding into a fixed prompt* (replacement, paper-faithful);
    EasyNLA/Celeste = norm-matched **addition** at block 1 (Karvonen). Different mechanisms.
  - Official README also documents "model-specific scale factors, the **Gemma √d embed-scale
    gotcha**" — directly relevant to my prior Gemma debugging (Objective 2).
  - **STRATEGIC OPTION (mine to decide):** develop/debug on **Qwen2.5-7B** (fits one
    mid-range card, repo I already know) → run the headline on a **current** model
    (official Gemma-3-27B, and/or Celeste's Qwen3.6-27B). Running **both** gives
    cross-implementation replication across 2 base models and 2 injection schemes —
    robustness Neel rewards, and unreported either way.
    Caveat: Qwen2.5 is 2024-era; not on Neel's banned list (GPT-2/Pythia/Gemma 2) but in its
    spirit — so 7B is for development, not for the headline. Gemma 3 IS on his recommended
    list. Licence: official = Apache-2.0 + base licence (cleaner to publish than "other").

- **2026-08-19 — official sidecars fetched; direction-only MSE confirmed NUMERICALLY.**
  | | Qwen2.5-7B L20 | Gemma-3-27B L41 |
  |---|---|---|
  | d_model | 3584 | 5376 |
  | **injection_scale** | **150.0** | **60000.0** |
  | mse_scale | 59.8665 | 73.3212 |
  | marker char / id | ㈎ / 149705 | ㈜ / 246566 |
  | left / right neighbour id | 29 / 522 | 236813 / 954 |
  - **√3584 = 59.8665, √5376 = 73.3212** → `mse_scale` IS √d_model in both official
    checkpoints ⇒ **direction-only MSE, confirmed from shipped artifacts** (3rd independent
    source: code defaults, README prose, now the sidecars).
  - **`injection_scale` is explicit and per-model: 150 vs 60000 — a 400× difference.** This
    is the "Gemma √d embed-scale gotcha" (Gemma scales embeddings by √d). Two rules:
    (a) no need to compute the paper's 75th-percentile α — the sidecar gives it;
    (b) **NEVER port a scale across models.**
  - Celeste's sidecar has **no** `injection_scale` — because the Karvonen path derives scale
    live from ‖h_p‖. Divergence now confirmed from the config side too.
  - Official AV prompt = identical to Celeste's, incl. "2-3 text snippets" (paper's Opus NLA
    used 4-5). AR prompt identical too. Celeste evidently reused the official template.
  - **CLOCK BOUNDARY (my call, written down):** running the maintainers' shipped example =
    setup, free. **The clock starts when cricket data enters the pipeline.**
  - **Start order:** (1) update fork branch [green button — NEVER "Discard 23 commits"];
    (2) `kitft/nla-inference` on Qwen2.5-7B, reproduce their `examples/` per-token MSE —
    this is the end-to-end smoke test for injection + scaling + AR; (3) verify √d myself
    (first entry in the verification log). **Then** start the clock and build the corpus +
    blind-labelling harness BEFORE any explanations exist.

- **2026-08-19 — old NLA session "lost" — NOT lost. Prior work recovered.**
  Only the **conversation transcripts** were pruned, by Claude Code's **local**
  `cleanupPeriodDays` (unset ⇒ default 30 days). Anthropic does not delete transcripts
  remotely for inactivity. Everything load-bearing survived because it was externalized:
  - **`../natural_language_autoencoders/.claude/skills/nla-runpod-inference/SKILL.md` (375
    lines)** — battle-tested A-Z recipe: pod deploy, cache redirection, dep install, AV
    server launch, 3-level smoke-test ladder, AR round-trip scoring, a **separate complete
    Gemma-3 recipe (verified 2026-06-07)**, "Key facts (don't rederive these)",
    troubleshooting appendix (error → cause → fix).
  - **`tutorials/`** — `round_trip.py`, `launch_av.sh`, `steer.py`, `setup_pod.sh`,
    `pod_datagen_setup.sh`, `peek_row.py`.
  - **`tutorials/gemma4_onboarding.md` (18.5KB, 8 sections)** + `gemma4_block_diagram.png`,
    `gemma_arch_compare.png`, `print_arch.py`, `drill_arch.py` → **this IS Objective 2**,
    already a draft with figures.
  - 7 memories under `~/.claude/projects/-Users-...-natural-language-autoencoders/memory/`.
  - **ALREADY DONE: Gemma-3-12B-L32 verified end-to-end 2026-06-07, AV→AR round-trip
    cos ≈ 0.997, planning finding reproduced.** Same model I re-derived today from cost
    arithmetic. The "day-zero smoke test" is already passed once and written up.
    Qwen2.5-7B-L20 verified end-to-end 2026-05-31.
  - **Version trap (per model):** Qwen needs **transformers 4.57.6** (sglang pulls 5.x →
    `apply_chat_template` returns BatchEncoding not list[int] → marker scan finds 0 matches
    → AssertionError). Gemma-3 needs **transformers 5.3.0** + a `return_dict=False` patch
    (already committed to the fork). **Separate environments.**
  - Proven stack: torch **2.9.1+cu128**, sglang **0.5.10.post1** (`--attention-backend
    triton --sampling-backend pytorch` on Blackwell sm120), CUDA 12.8. Deploy with
    **≥60–100GB disk**; `HF_HUB_DISABLE_XET=1`, `HF_HUB_ENABLE_HF_TRANSFER=0`;
    passphrase-less SSH key added BEFORE pod boot; gated HF token for Gemma.
  - **Open follow-up recorded there:** NLA steering not faithfully replicated — AR-vector
    steering gave a negative result (2026-06-07); next move is `--source-text` activation
    PATCHING, plus an AV-readback of the injected vector.
  - **Working-preference discrepancy to resolve:** the recorded memory says I *rejected*
    Claude SSH-driving the pod and want ONE command at a time in my own terminal. Confirm
    which mode for this project — it changes GPU cost.
  - **TODO: set `cleanupPeriodDays` to 365 in `~/.claude/settings.json`.**

- **2026-08-19 (CORRECTIONS — the CODE outranks the paper. Read `stage0_extract.py`.)**
  - **RETRACTED, and I was right the first time:** `_MIN_POSITION = 50` **IS REAL.** It is in
    the official Anthropic repo, `nla/datagen/stage0_extract.py:35`:
    `_MIN_POSITION = 50  # need enough left-context for the activation to be meaningful`.
    Extraction only samples positions `i >= 50`; docs with no valid positions past 50 are
    skipped. Also documented in the repo's `CODEBASE_MAP.md` and my own SKILL.md
    ("earlier tokens decode to noise"). **Claude told me twice this was a conflation of
    three unrelated 50s — that was wrong.** The paper does not mention it; the code enforces
    it. **Lesson: for what the system DOES, the code is authoritative, not the paper.**
  - **RETRACTED: `chunk_size` is DOCUMENTS, not tokens.** `--chunk-size` = "docs per
    extraction call — also the parquet write granularity". Nothing to do with token windows.
  - **RETRACTED: positions are sampled RANDOMLY, not taken from the end.**
    `positions_per_doc: 10` = 10 positions drawn by `rng.sample()` from all positions
    `>= _MIN_POSITION` that are not special tokens. (The paper's *recurrence analysis* uses
    the last 10 tokens — a different procedure. Claude conflated the two.)
    RNG is seeded on `(seed, doc_id)` so a doc gets the same positions regardless of corpus
    slicing/ordering — parallel runs on disjoint slices merge cleanly.
  - **Stage-0 row schema (this is "one row of the dataframe" — my tabular instinct is RIGHT
    here):** `n_raw_tokens` int64 · `detokenized_text_truncated` str · `activation_vector`
    FixedSizeList(float32, d_model) **RAW** · `activation_layer` int64 · `doc_id` str.
    One row = one (document, position) = one NLA input. Gemma-3-12B ⇒ 3840 floats.
  - **`detokenized_text_truncated` IS the prefix** — the "ledger" from the cricket story as a
    column. Grading task = `(detokenized_text_truncated, claim) → verdict`.
  - **Confirmed:** vectors stored raw by design — "Data-gen never normalizes — that's a
    training-time decision." Explains `norm: none` in both implementations.
  - **FVE needs a corpus baseline — a single round trip gives MSE and cosine only.**
    My own `tutorials/round_trip.py` prints `mse_nrm` and `cos`, never FVE. EasyNLA ships
    `scripts/compute_fve_baseline.py` for the population statistic. Expect cos ≈ 0.9+ on a
    good reconstruction; do NOT go looking for ~75% from one passage.
  - Schema war story worth heeding: they use `FixedSizeList` because a uint32 byte-offset
    overflow at 4 GiB "silently corrupted ~40% of the 100k RL run" — silent corruption at
    scale, uncatchable by reading output.

- **2026-08-20 — FIRST POD SESSION. Pipeline verified, cricket adopted. ~$1.23, 53 min.**
  Full detail in `mats_2027/experiments.md` → **SMOKE-01**. Headlines:
  - **Round trip reproduces exactly: cos 0.997 on their example** — identical to my
    2026-06-07 result, on a different machine/GPU arch/fresh install. Pipeline is sound.
  - **CRICKET IS IN-DISTRIBUTION: cos mean 0.996 (0.994–0.997) vs their 0.997.** The
    objection that killed it was about Ultra-FineWeb's *contents*, not the model. Settled
    by measurement. **Cricket adopted as the corpus.**
  - **Confabulations appeared in the first six activations, and I verified them by hand:**
    the Laxman passage (Dravid x0) was described as *"a cricket match involving Dravid"*;
    the Dravid passage (Tendulkar x0) as *"factual prose about Sachin Tendulkar"* — with a
    correct Dravid claim in the very next sentence. Plus invented quotes. That is the
    paper's 64/28/24 pattern, live, on my own data.
    → **Vindicates the cricket choice:** I caught the player swap instantly; a grader
    without cricket knowledge plausibly would not.
  - **A100-SXM4-80GB (Ampere) was the right call** over the Blackwell card I used in June —
    zero CUDA issues, and cheaper. torch 2.9.1+cu128 / transformers 5.3.0 / sglang
    0.5.10.post1 all as recorded in the skill.
  - **Claude's "injected vector is comparable to an ordinary token" was REFUTED by
    measurement: the ratio is 1357.7x.** Its attention-domination mechanism story is also
    wrong — that demo had no layer norm, and Gemma pre-norms each position before Q/K/V.
    The *value* of injection_scale is right; the *explanation* was not.
  - **p75 heuristic validated:** real layer-32 norms 64.5k–78.9k, injection_scale 80000
    sits at the top of that range.
  - **K=8 resampling: the AV is very stable at T=1** (sd/mean of mse 6–7%, explanation
    lengths within ~15%). **BUT this measures the wrong quantity for H2** — H2 needs the
    *paired* Δ variance (full vs claim-ablated, same explanation), where resampling noise
    largely cancels. Measuring that properly is the first job of Session B.
  - Correction: **~4.0 chars/token** for entity-dense text, not 4.9.
  - Artifacts: `mats_2027/runs/2026-08-20_smoke/` (208 KB). **Pod terminated, 0 running.**

  **Clock still NOT started.** Next, all free of the GPU: build the cricket corpus, write
  the blind-labelling harness, dry-run the analysis on fake data. Then Session B (AR-only,
  fits a 24 GB card) for ablations.

- **2026-08-21 — BIG DAY. Pipeline complete end to end; simulation caught a design flaw;
  measured the number that fixes it. Two pod sessions, ~$1.08 total. Balance $8.78.**
  Full detail in `mats_2027/experiments.md` → SCOUT-01, PILOT-03, SIM-01, NOISE-01.
  - **Corpus built:** 144 Wikipedia cricket passages, reproducible (seeded, revision-pinned),
    one paragraph per article, `article_chars` as a memorisation proxy (252 → 76,370).
    `mats_2027/corpus/cricket_passages.jsonl`. Cricket killed as an Ultra-FineWeb *filter*
    (0.00% in 4 shards) but adopted as a Wikipedia corpus (cos 0.996 in SMOKE-01).
  - **Blind grading harness built and leak-tested** (`mats_2027/harness/`): the file the
    grader loads contains ONLY claim_id/claim/prefix; verdicts write to disk per keypress
    via `serve.py`. Retest block for self-agreement (grading solo).
  - **Stage 3 decomposition prompt FROZEN** (structured outputs, Haiku 4.5). The paper
    never published theirs — checked paper, appendix, official repo. **User caught two
    things:** claims need explanations to exist (a sequencing gap in my plan), and
    "sets up a concluding statement" is a LAST-POSITION artefact → forward-looking claims
    are now excluded (they describe the future of the text; ungradable against the prefix).
    ~8–9 claims/explanation, not 2–3 → ~52k claims for Session A → LLM judge + human
    validation is mandatory.
  - **SIM-01 — the analysis, tested on planted worlds BEFORE real data:** H2 code works
    (slope −0.46 vs −0.50). **H1 bimodality test CANNOT separate the worlds at the assumed
    noise (4–5× effect), even at K=64.** The 0.8 noise was a ratio extrapolated from 11
    claims in one transcript — never measured. Failure mode #1 caught for free.
  - **NOISE-01 — measured it directly** (A40, $0.23, 6 pilot activations × K=8 → 401 claims
    → 447 AR scorings): **paired-Δ noise/effect ≈ 1.4×, not 4–5×.** Design stands.
    Caveats: n=6 recurring claims, 3–5 resamples each, exact-text matching, last-position
    only, prose-damage rival uncontrolled.
  - **Unexpected:** **18.8% of single-claim removals IMPROVE reconstruction** — Neel's exact
    phrasing, which the paper never reported. And the same sentence ("mentions Dravid"),
    true in one passage and confabulated in another, gets identical AR weight (≈0), while
    the confabulated Tendulkar claim is load-bearing. First real numbers in the 2×2.
  - **Corrections to me today:** "streaming is free on disk" (xet cached 1.5 GB); chars/tok
    is 4.0 not 4.9; the 0.8 noise figure was mine, not the paper's, and was wrong by ~3×.
  - Tooling: `runpod_balance.sh` (read-only key) for live balance; `cleanupPeriodDays=365`.

  **Next: (1) widen recurrence matching + last-10-positions extraction for the 6 pilot
  passages to firm up the 1.4×; (2) re-verify `noise_analysis.py` line by line with the
  user; (3) THEN Session A. Clock: today ≈ 3h on-task (corpus, prompt tuning, SIM, NOISE).**

- **2026-08-22 (04:00–05:15 IST, POS-01) — last-10-positions data collected. $0.40.**
  - **60 activations** (6 pilot passages × last 10 positions, all ≥50) + **240 explanations**
    (K=4, T=1) in `mats_2027/runs/2026-08-22_pos10/`. 0 no-tag, 0 CJK. Pod terminated.
  - Regression vs SMOKE-01 vectors: cos 0.99999+, **not bit-identical** (A100 vs A40 bf16).
    Harmless, recorded as such.
  - **User's hypothesis (not yet H3, their call):** text-false rate rises toward the final
    position. Mechanism story Claude offered was retracted — unsourced. It is a measurement.
  - **Key consequence:** position is a potential confound for H1's bimodality test — claims
    pooled across positions could show two humps *by position*, not by truth. The position
    check is a precondition for H1, not a detour. Re-run SIM-01 at the measured 1.4× noise
    with n≈2k claims before attempting H1.
  - Pod scripts rescued from `/tmp` → `mats_2027/pipeline/pod/` (README maps each to its run).
  - TODO recorded in experiments.md: **batch the AV client and AR scorer** before Session A
    (serial = ~12 h for 5,760 explanations; batched ≈ 2 h). Verify batched == serial on a subset.
  - Fork path corrected: it is `../natural_language_autoencoders/`, a sibling of this repo.
  - Still owed: `noise_analysis.py` walkthrough with the user (the 1.4× is unverified by them);
    `.gitignore` + `git init`; H1 "Predicts" rewrite.

  **Next session: read Laxman across positions first (no numbers) → Stage 3 → text verdicts →
  position check → only then H1.**

- **2026-08-23 — Δ EXISTS. Four experiments closed 22nd, AR scoring closed 23rd. $0.34.**
  Full detail in `mats_2027/experiments.md` → SOURCE-01, DECOMP-01, JUDGE-01, ABLATE-01,
  SCORE-01. Repo now public: **github.com/ADITHYAG73/AGsResearch_Claudeoppolous**.
  - **Specificity REPLICATES** on a different NLA/base model/corpus/positions and an
    independently written judge prompt: THEME 69.1% / ENTITY 43.8% / DETAIL 36.0% supported
    (paper 64/28/24). First real result of the project.
  - **The "more confabulation at later positions" idea is DEAD on this data**: +2.4pp early
    vs late, pooled SE 2.9, n=2065, and the sign is opposite to the prediction. AG called
    this from eyeballing the Laxman explanations before any data existed.
  - **Δ for all 2065 ablations** (`runs/2026-08-23_ar/deltas.parquet`). **30% of single-claim
    removals IMPROVE reconstruction** (NOISE-01 saw 18.8% at n=399 — but ablation method AND
    positions both changed, so not attributable). **DETAIL Δ = 4.5× THEME Δ**, the opposite
    of what the paper's specificity result predicts — consistent with H3 (redundancy), not
    a test of it.
  - **SOURCE-01: the paper's confabulation pipeline recovered from its own HTML.** Recurrence
    matching is an LLM call, not string matching; ablation is a REWRITE, not a deletion; the
    prompts were never published. Our `norm()` regex is the crude version — that is why
    NOISE-01 had only 6 recurring claims out of 401.
  - **GAP FOUND (AG, by checking his own mental model against the labels): relatedness labels
    do not exist.** 995 claims are false and none has been asked DIRECT/ADJACENT. **H1 is not
    testable until Stage 5 runs** (~$1, no GPU). H2 is unaffected.
  - **H1 detector decision is PARKED** in `hypotheses.md` with a hard constraint: decide it
    BEFORE looking at the real related-false Δ distribution. SIM-02 showed the frozen rule
    (`dBIC>10 AND dip p<0.05`) scores 0/40 at the measured noise; `dBIC>10` alone scores
    40/40 with 0 false positives on a Gaussian AND a skewed null, at K=4, n=2065. Also found:
    **more averaging can make it WORSE** — at low noise the skew stops being masked and ΔBIC
    reads it as bimodality (92% false positives at ratio 1.0, K=8).
  - **THE DAY'S LESSON, and it cost the morning:** removing `sglang[all]` from `pod_setup.sh`
    "because the AR needs no server" broke the pod — sglang was silently upgrading torchvision
    to 0.24.1 to match torch 2.9.1. Symptom was `std::bad_alloc` on IMPORT with no traceback.
    Three causes were proposed from knowledge and all three were wrong; `faulthandler` and a
    diff against the last working setup.log settled it in minutes. **Do not "optimise" a
    recorded recipe.** Guard now written into `pod_setup.sh`.
  - Batched AR scoring was BUILT and then NOT USED: serial is 0.07 s/item (2305 in 135 s), so
    batching saves ~80 s and adds 2.5e-4 bf16 noise to a Δ whose effects are ~1.1e-3. bs=1 is
    bit-exact with the official `NLACritic.score()`.

  **Next: (1) Stage 5 relatedness (~$1, no GPU) — unblocks H1; (2) settle the parked detector
  decision BEFORE looking at related-false Δ; (3) AG's 150+30 blind grading to validate the
  judge; (4) `noise_analysis.py` walkthrough — still unverified by AG.**

- **2026-08-25 — H1 TESTED AND KILLED. Relatedness done. Zero GPU spend.**
  Full detail in `mats_2027/experiments.md` → REL-01, H1-01. Balance unchanged at $7.98.
  - **REL-01: 98% of false claims are RELATED** (975 of 995), 2% unrelated, ~$1, no GPU.
    This **quantifies a claim the paper asserts twice without a number** ("even false claims
    are usually somewhat related to the context rather than fabricated wholesale"). Given a
    cricket activation, the AV confabulates cricket.
    → **Consequence: the paper's related-vs-unrelated Δ contrast is NOT reproducible on this
    corpus** — 20 unrelated claims, ~6 of them mislabelled. State it as a limitation.
    → Simplified to binary after a 3-way pilot failed (returned ADJACENT 20/22 including
    cases AG ruled DIRECT, and was self-inconsistent on the same claim across resamples).
    The deciding argument: **H1 never needed the split.**
    → Negative control run BEFORE spending: 7 synthetic out-of-domain claims → 7/7 UNRELATED.
  - **H1-01: H1 is NOT SUPPORTED.** Related-false Δ (n=975, K=1) is unimodal, dip p=0.992,
    with **76–100% power** across H1's own predicted 26–42% mixture range. AG's kill condition
    from Aug 19 met exactly.
  - **The ΔBIC rule AG signed the same morning turned out to be broken, and it was Claude's
    error.** It fired (+843.4) — but a single skewed hump matched to the real data produces
    dBIC>10 in **200/200** draws, median +846.5. The real value sits BELOW the null's median.
    Internal control: TRUE claims, which H1 says nothing about, score +2467. **SIM-02's
    skewed null used skew ≈1.0; the real data is 2.63 (related-false) and 5.63 (true).**
    Re-run at the real skew the rule false-positives at 99–100% for every affordable K.
    Hartigan's dip — the detector Claude argued to DROP — has 0% false positives throughout
    and was right all along.
  - **H1's verdict is EXPLORATORY, not confirmatory**, because the rule had to be revised
    after the real distribution was seen. That is the pre-registration's own clause applying
    to Claude's mistake.
  - **H3 (redundancy, AG's) is now the live alternative** — it would produce exactly this null
    whether or not a mixture exists. A dead H1 does not show the AR treats these claims alike,
    only that their measured Δ does not separate.
  - Still parked: **MATCH-01** (semantic matcher, 2 pilots, over-merges; verifier-pass fix
    designed not built). It blocks H2's kill condition, ablation-fidelity verification, and
    H3's redundancy score.

  **Next: (1) AG's 150+30 blind grading — turns five provisional results into solid ones;
  (2) the matcher (verifier-pass design) — unblocks H2/H3; (3) start the write-up.**

- **2026-08-25 (later) — JUDGE VALIDATED, MATCHER FIXED, H2 ANSWERED. Zero GPU spend.**
  Full detail in `mats_2027/experiments.md` → JUDGE-02, MATCH-02, NOISE-03, H2-01.
  - **JUDGE-02 — AG graded 150 blind + 30 retests in 33.6 min.** Self-consistency **96.7%**
    (the paper's own hand-validation standard was 97% by two authors). Agreement with Haiku
    **88.7%** on the binary scale. **The S/C/N three-way scale was my design error** — Stage 3
    phrases claims as "The text mentions X", a META-claim, for which CONTRADICTED is
    structurally almost unreachable (Haiku used it on 1.9% of claims). 10 of 27 disagreements
    were pure C-vs-N confusion. **All analysis now uses binary SUPPORTED/FALSE**, as the paper
    does.
  - **Adjudicating all 27 disagreements found two error modes on DIFFERENT claim types:**
    AG is too generous on `quote` claims (Haiku right); Haiku is too strict on `format`/`genre`
    claims (AG right — third independent sighting of that Haiku bug). **They sit at opposite
    ends of the specificity axis, so both COMPRESS the gradient — the true THEME→DETAIL gap is
    probably larger than either measured.** The specificity replication survives and likely
    understates.
  - **AG's Bradman observation validated and extended into a characterised failure mode.**
    "Bradman" IS in the Dravid prefix — as the *Bradman Oration* Dravid delivered. The AV
    re-binds Dravid's records to Bradman. **21 claims across 4 positions**, including "The text
    is about Don Bradman". It took cricket knowledge to spot; the corpus choice paid off.
    **[CORRECTED 2026-08-31, after AG asked where the 99.94 was attached.]** Two errors here,
    both Claude's: (a) "blends in Bradman's real 99.94 average" is a CROSS-RUN CONFLATION — 99.94
    never occurs in this data; the invented averages are 95.99 → Brian Lara and 51.37 → Kevin
    Pietersen. 99.94 occurs twice in the project, both in PATCH-01. (b) Bradman is NOT special:
    on the Dravid passage the AV names Dravid 27, **Bradman 21, Tendulkar 20**, Lara 6, Richards
    3, Pietersen 2, Gavaskar 2, Hobbs 1 — and Tendulkar is absent from the passage entirely.
    So this is **IMPORT from parametric knowledge**, the same failure SAVARKAR-01 measured;
    re-binding a name that IS present is the rare special case, and that is also why PATCH-01's
    DELETE condition changed nothing — the prefix name was never the mechanism.
  - **MATCH-02 — the matcher works.** Fix was a **second pass** (audit each group in
    isolation), not a better prompt. 115 groups with ≥3 resamples vs 22 by regex (5.2×).
  - **NOISE-03 — H2's kill condition NOT met, and the condition was mis-specified by me.**
    The noise is **stochastic**: spread(K)²=signal²+noise²/K fitted on K=1,4 predicts K=2,3
    out-of-sample to 3%, and two independent noise estimates agree. The slope is −0.273 not
    −0.50 because there is a **signal floor at 55% of the K=1 spread** — which is what we want.
    **Also: NOISE-01's 1.41× and NOISE-02's 0.12× used the MEDIAN within-group sd, the wrong
    statistic for a variance decomposition. Do not quote them.** RMS-based: 0.5–1.4×.
  - **H2-01 — the AR's per-claim AUC is 0.535 [0.510, 0.559]**, which quantifies the paper's
    unquantified "weak per-claim verifier". K-averaging → 0.615 but the CI includes chance.
    **H2 partially supported**; bottleneck is recurrence (115 of 2065 claims), not noise.

  **Next: (1) the write-up — this is now the binding constraint; (2) optional: AG spot-checks
  ~20 matcher groups, the only independent check that component has; (3) optional: more
  resamples per activation would decide H2, but that is a pod session.**

- **2026-08-28 — PATCH-01 analysed. The intervention never reached the representation.**
  Full detail in `mats_2027/experiments.md` → PATCH-01 R2–R7. ~$1 (Haiku); no GPU today.
  - **Stage 3: 2603 claims from 280 explanations, 0 parse problems.** (First attempt died
    280/280 on HTTP 401 — the Anthropic key had expired. Nothing charged, nothing partial
    written; AG rotated the key.)
  - **Behavioural result: the confabulation does NOT follow the planted name.** Planting
    Gavaskar/Umrigar/Thangavelu produced **0/40** uses of each. Deleting Bradman did not
    reduce Bradman (25% vs 22.5%). Cricket legends absent from every prefix appear in ~45% of
    explanations in **every** condition.
  - **BUT THE CONTROL FAILED, AND THAT IS THE REAL RESULT.** On mean-centred activations, the
    edit moves the representation by 0.997–0.9995 cosine, against **0.42 for a single token
    step** and **0.01 for a different passage.** Deleting 26 tokens ~250 characters upstream
    changes the layer-32 activation **less than 1% as much as moving one position.**
  - **So the seeding hypothesis is UNTESTED, not refuted** — AG's predictions, mine, and the
    JUDGE-02 R6 mechanism story all remain unexamined. I was about to report them falsified;
    the control is the only reason that did not happen. JUDGE-02 R6 has been corrected in
    place.
  - **What IS established:** at layer 32, at these positions, the residual stream barely
    encodes context from ~250 characters back. Consequence: **the AV cannot be reading
    "Bradman" out of the activation** — the name must come from parametric knowledge. That
    coheres with REL-01's 98% RELATED: confabulations stay in-domain because they are drawn
    from domain knowledge, not copied from context.
  - **Why the design could not have worked:** the edit was placed upstream deliberately so it
    would not change the sampled tokens — which is exactly why it had no effect. **A causal
    test needs activation patching with hooks**, or sampling positions adjacent to the edit.
    Text-level intervention at that distance cannot reach the representation.

  **Next: (1) the write-up — 7 days to Sept 4 and it has not started; (2) optional: activation
  patching, which would be the first hooks AG writes himself; (3) parked: more resamples to
  decide H2.**

- **2026-08-28 (later) — SAVARKAR-01: domain transfer. Both predictions refuted. $0.45 + ~$2.**
  Full detail in `mats_2027/experiments.md` → SAVARKAR-01. 7 random pages of a 2019 Penguin
  biography, length-matched to cricket, Dravid canary at cos=1.000000, watchdog clean.
  - **The AV confabulates MORE on the biography: 63% false vs 50% on cricket** (ALL-level
    CIs disjoint). AG and I both predicted fewer. **Parametric knowledge is the source of the
    correct specifics, not of the errors** — where the activation is thinner the AV fills the
    same specificity budget with the nearest famous entities it knows (Gandhi, Bhagat Singh,
    Tilak), which are wrong. Interpretation, not a test.
  - **>90% of false person-claims name someone absent from the passage, in BOTH domains**
    (Savarkar 93.6%, cricket 98.3%). Re-binding a present name — the Bradman story — is the
    rare case. The dominant failure is IMPORT. Coheres with PATCH-01 and REL-01.
  - AG's p449 prediction: right entity (Gandhi, 13 false claims), wrong mechanism (invented
    Gandhi facts, no re-binding). Fails by its own stated condition. His Afghanistan-window
    guess held.
  - Position flat on a second corpus.
  - Caveat: judge validated only on cricket; part of the gap could be judge harshness on
    unfamiliar names. AG has no domain knowledge here — the point of the design, but it means
    a blind sample would be a weaker standard.

  **Next: WRITE. 7 days to Sept 4. Nothing else goes near a GPU.**

- **2026-09-01 — ALL ELEVEN SECTIONS DRAFTED. Write-up assembled. Zero GPU spend.**
  `mats_2027/writeup/DRAFT.md` (6,525 words) built by `writeup/compile.py` — deterministic,
  re-runnable, strips Claude provenance blocks and AG's `<notes to self>`, places G0–G5.
  - **Written 30 Aug–1 Sep:** B, C, D1–D6, E, F, F2, A. E is AG's alone. A is Claude's draft
    (582 w, under the 600 limit) and still needs AG's voice — it is the section Neel reads first.
  - **Three corrections found by AG asking, not by review:** (a) the **99.94** batting average
    never existed in the data — a cross-run conflation that had propagated into experiments.md,
    CLAUDE.md and D6; the real invented averages are 95.99→Lara and 51.37→Pietersen. (b) **Bradman
    is not special** — on the Dravid passage the AV names Bradman 21, **Tendulkar 20** (absent from
    the passage entirely), Lara 6, Richards 3; so the phenomenon is IMPORT, not re-binding, which
    also explains why PATCH-01's DELETE condition did nothing. (c) E claimed AG adjudicated the 27
    judge disagreements — **Claude did that**; corrected.
  - **Two numbers were not reproducible and now are.** `pipeline/noise_fit.py` rebuilds H2's
    variance decomposition (the out-of-sample fit existed only as an in-session calculation):
    spread 0.00225→0.00181→0.00164→0.00157, fit on K=1,4 predicts K=2,3 to **0.8%**, noise 0.00185
    vs 0.00212 from within-claim RMS, signal floor **56%**. `pipeline/patch_control.py` rebuilds
    PATCH-01's control and reproduces the recorded 0.4221 — it also documents the two traps:
    align by **offset from the end** (absolute pos is not comparable across edits) and centre on
    the 60 POS-01 activations (pooling all 130 moves the token-step reference 0.42→0.25).
  - **115 vs 110 was a conflation:** 115 groups have ≥3 member CLAIMS; **110 span ≥3 distinct
    RESAMPLES**, which is the quantity H2 needs. Corrected in D4, F and experiments.md.
  - **New finding, SCORE-01b:** the DETAIL/THEME Δ gradient falls **4.5× → 2.7×** once claims
    naming the passage's final token are removed, and ENTITY vs DETAIL becomes indistinguishable.
    Script `pipeline/final_token_control.py`. Never quote the 4.5× alone.
  - **G3 power settled at 400 draws:** 86–100% across H1's 26–42% range (the 76% was 80 draws).
  - **The paper's Future Work names AG's next step:** "best-of-N NLA explanation against AR
    reconstruction" — inference-time, whole-explanation, needs no recurrence matching, and the
    resamples + scorer already exist. Their legibility section also independently reports that
    NLAs "often repeat the same content on multiple bullet points", which corroborates H3.
  - Figures: G0 pipeline flowchart + G1–G5, plus teaching figures T1 (binning) and T2 (two bells).

  **Next: (1) TRIM — 6,525 words against a 2,500–3,000 target, mostly D6 (1,460), C (777),
  D2 (686); (2) AG's voice pass, especially A, D5, D6, F2 which are Claude's prose; (3) typo pass
  — Claude has deliberately not touched AG's typos; (4) Google Doc, anyone-with-link;
  (5) THEN the application form questions (Q8 cannot cite this project).**

- **2026-09-02 — WRITE-UP FINISHED, PUBLISHED, SHARED. Paper scrutiny found two errors. Zero GPU.**
  **FINAL DOC (anyone-with-link, Viewer):**
  https://docs.google.com/document/d/1H8bNinfS9vIK_NFSSVyHKUttA20XIlQoPZatC4rhdhU/edit
  - **AG read the whole draft twice and left 52 comments.** All addressed. The biggest theme
    (~9 comments) was VOICE — he could tell exactly which sections Claude wrote, and flagged
    only those (A, D2, D4, F, F2), never the ones he wrote. A was rewritten in his register
    (602 words, under the 600 cap); audit-report phrasing removed throughout.
  - **PAPER SCRUTINY (he asked for it; done in the browser against the live article).** All 11
    claims we make about the paper verified. `writeup/evidence/PAPER_SCRUTINY.md`. Two findings:
    (a) **64/28/24 is real but printed inside their FIGURE** (`png/img_18fcfc16e92031e0.png`) —
    it appears **0 times** in their prose. Now stated as such.
    (b) **WE WERE WRONG:** our claim that the paper asserts the related/unrelated split "without
    a number" is false. The same figure gives it — of FALSE claims, related share is
    **83% theme / 70% entity / 91% detail, ≈80% overall.** D6 now compares our 98% against
    their ~80% (corpus breadth explains the gap) instead of claiming we filled a gap.
  - **APPENDIX added — one activation end to end**, which answers his "what was the prefix,
    what was the explanation, what was the claim". Eden Gardens pos 87: prefix ends
    `...renamed to the 'Eden Gardens`; the AV writes **'Elphinstone Gardens'** — right slot,
    wrong name — and that false claim is the single most load-bearing in the explanation
    (Δ=+0.00849) while both true claims sit at ~0. Four of six wrong if Δ is read as truth.
  - **His three factual challenges, all checked:** 53.8% = 780 true with Δ>0 + 330 false with
    Δ<0, over 2063 (shown as arithmetic now); the −0.50 slope is the exponent on a log-log
    plot; and **"250 characters" was a rounded midpoint — the measured range is 218–263**,
    now stated. He was right to challenge it.
  - **Both models credited: "Opus 5 / Fable 5"** (6 places). Neel's doc is cited as a real
    **hyperlink on the anchor text** to the Recommended Research Problems tab
    (`?tab=t.knytn7x826kv`), not a pasted raw URL — he rejected the first attempt, correctly.
  - Repetition cut (judge-validation claim was in A, C and D1 — now D1 only). Equations on
    their own lines. All underscores stripped for the Docs import (`writeup/export_for_docs.py`
    asserts zero, because markdown import mangles `injection_scale` into `injection\_scale`).
  - Six figures re-inserted at full PNG quality via clipboard paste; 0 markers remain.
    Superseded Docs trashed. Outline sidebar on. **Sharing set to anyone-with-link.**
  - **Process note worth keeping:** patching the live Doc through the browser was the wrong
    tool and failed on an encoding bug in front of AG. Fixing the SOURCE and republishing is
    the reliable path; the browser is only for what the API cannot do (inserting images,
    setting link-sharing, making hyperlinks).

  **Next: the application form Qs — Neel reads these FIRST and uses them as a preliminary
  filter. 20+2h accounting to be answered honestly (the clock was never formally started).
  Q8 cannot cite this project: use the two live blog posts (free norm, flash attention).**

- **2026-09-03 — APPLICATION SUBMITTED. Zero GPU spend.**
  - Neel's format re-checked against the admissions doc: graphs in the exec summary are expected,
    and randomly selected examples must follow it. G4 and G5 now also sit in A; a "Randomly
    selected examples" section (6 claims, seed 20260903, `pipeline/random_examples.py`) follows
    A; the appendix discloses that its Eden Gardens example was chosen, not drawn.
  - All 12 Airtable questions captured in `writeup/FORM.md` with the source material for each.
    AG wrote the answers himself in the form (Q2 used a Claude draft as-is, after the
    LLM-prose risk was flagged once). Q8 cites only the two live blog posts (both 200).
  - Doc `1H8bNinfS9vIK_NFSSVyHKUttA20XIlQoPZatC4rhdhU` confirmed to open with no Google login;
    sharing set to anyone-with-link, Commenter. Submitted with giridharanadithya@gmail.com.
  - `mats_2027/app_answers.md` is gitignored (personal).

  **Next: nothing owed to the application. Open items, in order of value: (1) HF token rotation
  (leaked into a log 2026-08-22, still not rotated); (2) matcher spot-check of ~20 groups — the
  one component with no independent check; (3) Objective 2, the Gemma per-layer-embedding
  write-up; (4) Savarkar Δ / more resamples if a pod session is ever justified.**

- **2026-09-06 — NEW DIRECTION SCOPED: Forking Paths. Zero GPU spend. Nothing committed.**
  Read both papers in full: Bigelow et al. 2024 (arXiv 2412.07961, GPT-3.5, S=30) and
  Goodfire's "Forking Fast" 2026 (arXiv 2608.19611, Llama-3-8B + R1-distill, S=200/1000).
  - 2024 asks *do forking tokens exist*; 2026 asks *how cheaply can the curve be measured*.
    2026's own result — noise is exactly multinomial, jaggedness at S=30 is largely dice —
    partly undercuts 2024's odd-token forks, and nobody has re-examined them at high S.
  - **Open-ended outcomes are named by both papers and run by neither.** That is AG's
    two-year-old question. Both S/N and the open question are recorded in
    `forking/ROADMAP.md` (six open items, three sized projects, free pre-GPU work).
  - AG forked `ericb-goodfire/forking-fast` → `IamAGP/forking-fast`; cloned to
    `reference_repos/forking-fast` (gitignored). Sampler is plain transformers with a
    model-path flag; outcome extraction is one regex (`answers.py`). **No LICENSE file.**
    2026 paper's Appendix H: research and draft produced by their agent Silico.
  - The 2024 code exists after all: `github.com/ebigelow/forking-paths` (cited in the
    fork's config) — not yet opened. Claude had said it was only promised; wrong.
  - Teaching artifact for Figure 1 (bars → area chart, S slider):
    https://claude.ai/code/artifact/e0d185c3-1ba3-4c87-b4bd-4a7592c847e7
  - Claude's recommendation: project B (high-S replication of odd-token forks on Qwen 3.5 9B)
    first, then A (open-ended forks). **AG has not chosen yet** — reading the plan first.
    No email to authors before a finished result (AG's call, stated).

  **Next: AG internalises ROADMAP.md and asks questions; then the free CPU work — reproduce
  the §3.3 noise slope from `data/s1000`, open the dashboard, open the 2024 repo.**

- **2026-09-07/08 — COMPASS WRITTEN. FIRST RESULT: 2026 NOISE LAW REPRODUCED. Zero GPU.**
  - `forking/COMPASS.md`: 6 established claims, 10 gaps with sources, 6 pointed questions
    (falsifiable sentence, yes/no meaning, cheapest test, model, cost). Current models verified
    on HF 2026-09-07: Qwen3.8-27B (Aug 2026, dense, Apache 2.0), Qwen3.6-27B, Gemma-4-31B/12B.
    **Recommended order changed:** Q3 first (per-token swap effects o_{t,w} from the shipped
    2026 stores, zero GPU — the quantity 2026 dropped and 2024 cared about), then Q1/Q2 with
    activations cached, then Q4 (probe predicts forks). AG excited by Q4, holding back by choice.
  - 2024 authors' code cloned: `reference_repos/forking-paths` (GPL-3, ~1.9k lines, OpenAI API,
    S=30). Has an open-ended StoryCloze path drafted and never run — strengthens G1.
  - **NOISE-LAW-01 (`forking/experiments.md`):** row 39, S=1000, 344 positions, independent
    code. Slope −0.483 (paper −0.490, theory −0.5); ratio to multinomial null 0.99–1.02 through
    S=250. **Reproduced.** AG's independent hand-check still owed.
  - Found in passing: position 268 branches "(" vs "√" give clearly different final-answer
    distributions on 1000 rolls each — a live per-token swap effect (Q3 material).
  - Goodfire: Anthropic put $1M into a $50M Series A, not a sponsor; not in Series B.
  - Artifacts: comparison table https://claude.ai/code/artifact/a0541af0-962e-4eb8-ac02-716543c6d4bf ;
    15-step noise-law derivation https://claude.ai/code/artifact/e0d72b51-56e9-4d6d-a883-52e292fe046c
  - **Process note:** AG skimmed the 15-step page and got lost. Wrong dose, Claude's error.
    Agreed method: **one step per sitting, hand calculation on paper, then stop.** Steps 1–3 next.
  - Nothing committed. `forking/` (COMPASS, ROADMAP, experiments.md, pipeline/, runs/) and
    CLAUDE.md are uncommitted.

  **Next: AG does steps 1–3 of the derivation page by hand. Then Q3 on the shipped stores.**

- **2026-09-13/14 — SWAP-01 DONE (zero GPU); QWEN-01 KILLED with no output. $1.13 spent.**
  Full detail `forking/experiments.md`.
  - **SWAP-01 (compass Q3), on the shipped 2026 stores, Llama track:** 13,149 non-letter token
    swaps; **7.2% significant at p<0.01** (chance 1%); **83 swaps with effect ≥0.2 sit where
    the pooled curve is flat, in 41 of 100 questions** — per-token forks the 2026 charts cannot
    show. Row 80 t=228 ")(" vs ")[" flips C↔A (196/200 vs 169/200), effect 0.96, pooled change
    0.04. Odd tokens fork slightly more than words (punct 8.5%, digits 12%, words 6.9%).
  - **DeepSeek track is unusable for per-token answer swaps: 46.9% of outcomes are "Other"**
    (1536-token cap, thinking never finishes; answered_rate ≈0.12). Their Fig 10 caption
    acknowledges it. Usable for what THEY measured (noise law, smoothing); not for ours.
  - **QWEN-01:** Qwen3.5-9B (newest dense <27B; 3.6/3.8 ship 27B only), thinking off, row 80,
    S=200, only-forks → **108 forking positions, 280 branches**. Laptop smoke on 0.8B passed
    (checkpoint deleted after, per AG). A40 $0.49/hr, EU-SE-1. Kernel install needed the
    official setup.py recipe (CUDA_HOME + --no-build-isolation). HF `generate` ran 2 h 07 min
    without finishing; their code logs nothing inside a question and writes once at the end;
    ptrace blocked so no stack dump. **AG chose kill, no rerun.** Nothing survived.
  - **Rule adopted (AG, now global ~/.claude/CLAUDE.md §7 + memory `gpu-run-preflight`):**
    before ANY billed run, answer in the journal from reading the code: progress logs?
    incremental checkpoints? memory trajectory? what survives a kill? plus a cost line
    (tokens × rate → hours → $). Claude read run.py before launch and flagged none of it.
  - Parked rerun plan (AG's call): vLLM/SGLang, S=50 stride 4 first, per-branch log +
    checkpoint (adapter patched for the log), gen_batch ≤100; est. <20 min.
  - Goodfire authors checked: credible (Geiger, McGrath, Lubana, Bigelow); resampling-based
    CoT interp is a live cluster incl. Neel's Thought Anchors / Thought Branches. Behavioural,
    not mech interp — activations-predict-forks (Q4) is the bridge and nobody has done it.
  - Still uncommitted: `forking/` and CLAUDE.md. HF token still not rotated.

  **Next: AG's call. Options on the table: write up SWAP-01 (real result, zero GPU, ready);
  or the rerun with visibility. Nothing owed tonight.**

- **2026-09-15 — QWEN-02 DONE. Clean negative on a current model. $1.03.**
  Full detail `forking/experiments.md` → QWEN-02.
  - **Docs-first, per AG:** vLLM API from source (SamplingParams, LLM.generate/chat,
    TokensPrompt, prefix caching, install page), Qwen3.5 support confirmed in vLLM 0.29.0
    (`models/qwen3_5.py` + registry). Sampler rewritten on vLLM as OUR code with the paper's
    semantics (`forking/pipeline/fpa_vllm.py`): per-chunk progress lines, per-branch jsonl
    checkpoints, `--resume`, cost line printed before generation. Laptop dry-run on a fake
    vLLM passed, incl. resume. Pre-launch checklist (rule 7) answered in the journal first.
  - **Three of my mistakes caught by smokes, not by the run:** torchaudio CUDA mismatch from
    an extra pip index (fix: let vLLM's wheel pick torch); vLLM engine spawn needs
    `if __name__ == "__main__"` (the official example had it; I dropped it); the paper's
    400-token budgets truncate Qwen mid-answer and their `option X` regex then reads running
    commentary as the verdict (97/97 in smoke2). Probe: Qwen's greedy answer = 1,111 tokens.
  - **Main run:** stride 4, S=50, budgets 1,500, 206 branches at 81 positions, 7.03M tokens,
    78.6 min at ~1,500 tok/s. 94.6% explicit answers, 0 "Other".
  - **RESULT: no fork.** 125 alternative tokens, largest swap effect TVD 0.040, below the
    S=50 noise floor (0.06–0.08). B (correct) ≥96% everywhere. Llama flipped C↔A at 0.96 on
    the same item. Hypothesis NOT met. Open readings: easy item / sub-5% forks / MMLU
    contamination. Cheap next: greedy pass over all 100 tinyMMLU, pick Qwen's wrong ones.
  - Pod terminated, zero pods, balance $53.23. Still uncommitted: `forking/`, CLAUDE.md,
    global rule 7. HF token still not rotated.

  **Next (AG's call): (1) write up SWAP-01 + QWEN-02 together — the contrast IS the result;
  (2) or first the 100-item greedy pass (~$0.10) to find items Qwen is uncertain on.**
- **2026-09-15 (later) — SCREEN-01: Qwen3.5-9B on all 100 tinyMMLU items. $0.34.**
  Greedy accuracy **86/100** (not saturated). **17 items torn at t=0** (top answer ≤80% of 10
  samples), 7 of them also wrong; 7 further items confidently wrong. Cleanest fork candidate:
  **row 41** (A5/C5, no truncation, 480-token answer); then row 91 (A4/B3/C3, wrong), row 60.
  Two more infra lessons journaled: serial per-question vLLM calls are 5× slower than
  chunked batches; killing a vLLM parent leaves the EngineCore holding the GPU — kill the
  pids nvidia-smi lists and confirm 0 MiB before relaunch. Pod terminated, zero pods,
  balance $52.89. Day total $1.37.
  **Next: full stride-4 run on row 41 (~$0.25) — the real second test of the hypothesis on an
  item Qwen is torn on. Then write up SWAP-01 + QWEN-02 + (row 41).**
- **2026-09-15 (night) — QWEN-03: Qwen3.5-9B FORKS on a torn item (row 41). $0.79.**
  Row 41 (moral_scenarios, o_0 = 30A/20C): 159 branches at 55 positions, S=50, then S=200
  at the 5 hit positions. **6 swaps significant at p<0.01 (chance ≈1), all six confirmed at
  S=200 (p ≤ .001).** Strongest: t=164 "It"→"However", A 88% → 53/47 (TVD 0.35). Clean
  verdict flip that holds at S=200: t=52 "Offering"→"The", C 72% → A 55%. TVD ≥0.5 still
  not reached (max 0.35), so the strict sentence fails while the per-token claim holds.
  Contrast set is now complete: Llama row 80 (0.96 fork) · Qwen row 80 (none) · Qwen row 41
  (six, two verdict flips). Infra lessons: pin `allowedCudaVersions=["13.0"]` (a driver-570
  host killed a pod); `--positions` was missing from fpa_vllm.py (added). Zero pods, balance
  $52.10. Day total $2.16.
  **Next: WRITE. SWAP-01 + QWEN-02 + QWEN-03 + SCREEN-01 is a complete story with figures
  to make. AG's one-sentence claim first, in his words. Then LessWrong.**

- **2026-09-17→19 — MATHEMATICS thread opened (AG's curiosity, his direction). Zero GPU spend.**
  Session "MATHEMATICS". Full detail `mathematics/experiments.md`; memory `mathematics-curiosity-thread`.
  - **Two process failures by Claude, both called out by AG:** (1) first answer on LLM arithmetic
    given from recall, no tools, no citations, one arXiv ID wrong (helix paper is **2502.00873**,
    not 2502.00817), and it asked AG to verify a claim it could verify itself; (2) the RunPod
    skill (plugin 1.2.0, Aug 19) was trusted over the live docs and **missed Global Volumes
    (beta, launched 2026-09-14)** — AG found it in the console. Rule: for RunPod features,
    search docs.runpod.io; the skill lags by weeks.
  - **Verified (tools, primary sources):** FrontierMath is Epoch AI's; its harness gives the
    model a Python sandbox, so Astra's 97.6% on Tier 4 (**43 problems** after the June 2026
    update) is agentic. IMO gold 2025 was tool-free (Noam Brown, Dwarkesh 2026-09-17). OpenAI's
    Navier–Stokes claim (2026-09-08) is Lean-certified per OpenAI, disputed on priority by
    Buckmaster, and confirmed by no independent mathematician as of the coverage read.
    Mechanism papers stop at two-operand arithmetic on ≤8B models (Nanda 2301.05217, Zhong
    2306.17844, Stolfo 2305.15054, Nikankin 2410.21272, Levy–Geva 2410.11781, Kantamneni–Tegmark
    2502.00873, Biology paper addition section). **Nothing mechanistic exists on olympiad or
    proof-level math on any model.**
  - **Reframe that landed:** three mechanisms get conflated — (1) arithmetic inside one forward
    pass (what interp studies; weak past a few digits), (2) chain of thought as scratchpad (the
    text is the calculator; this is the IMO result), (3) external tools (FrontierMath). AG's
    "what is in its brain" is (1); the capability jump he is amazed by is (2).
  - **Candidate question (Claude's, not yet AG's):** does the Kantamneni–Tegmark helix survive
    in a 2026 RL-trained model? Replication code exists (`subhashk01/LLM-addition`).
  - **Serverless assessed and parked:** a custom worker CAN run hooks (flash class Endpoint,
    load once per worker, activations to a volume, ~10 MB payload cap), but it is ~1.7× the
    pod rate per working second and every wake reloads 55 GB on the meter. Right tool for
    sampling runs; wrong tool for interactive interp. Two-track design recorded in the session.
  - **Model facts (HF config):** Qwen3.8-27B = `Qwen3_5ForConditionalGeneration`, 64 layers,
    hidden 5120, 3 linear-attention : 1 full-attention, Apache 2.0, not gated. Official
    transformers doc: load text-only with `Qwen3_5ForCausalLM`; `fla` + `causal_conv1d`
    required or the DeltaNet layers silently fall back to slow PyTorch. Gemma 4 31B/12B are
    the second base model (Apache 2.0, not gated).
  - **MATH-SMOKE-01 written and dry-run PASSED on the laptop** (Qwen3.5-0.8B, exit 0, 278 s):
    `mathematics/pipeline/pod/{pod_setup_q38.sh,smoke_q38.py,run_smoke.sh}` — arch dump, one
    GSM8K question thinking-on, residual stream at 5 layers via hooks, four files written
    incrementally. Rule-7 preflight + cost line (~$0.70–0.95, A100) in `experiments.md`.
    Weights to go on a **Global Volume** (`/workspace-global`, $0.09/GB/mo, ~$5/mo for 27B),
    venv in the container. Stock was "Low" everywhere on the 19th; A100 SXM still deployable.
  - Hub banner "Qwen3.8-27B is live on Runpod Serverless": not in the public-endpoints API
    catalog (only Qwen3-32B-AWQ, Kimi K3, Granite there) — so probably a Hub serverless repo
    (a prebuilt worker to deploy in your own account), not a pay-per-token endpoint. Unverified;
    AG to click it.
  - Nothing committed. `mathematics/` untracked. Zero pods, zero volumes, balance $52.08.

  **Next (AG's call, tomorrow): read the two scripts; decide global volume yes/no, the question,
  the layers; say go → MATH-SMOKE-01 (~30 min, ~$1). Then read Kantamneni–Tegmark and write the
  one-sentence question. HF token STILL not rotated.**

- **2026-09-19 (evening IST) — MATH-SMOKE-01 DONE. Qwen3.8-27B looked at on a GPU. $0.39.**
  Full detail `mathematics/experiments.md` → MATH-SMOKE-01 run log. Balance $52.08 → **$51.69**.
  - AG chose **pod + global volume** over serverless. Checked first: serverless cannot mount a
    global volume (docs: container disk / network volume / S3 only), and the Hub banner
    "Qwen3.8-27B live on Serverless" is a **llama.cpp Q8_0 worker** — quantized, no hooks.
    The RunPod API cannot create or attach a global volume, so **AG deploys in the console,
    Claude drives over SSH.** AG also created an S3 API key (in `.env`); it covers network
    volumes only and is unused.
  - **Result:** A100 PCIe, EU-RO-1. 8 min from deploy to results. Download 52 GB in 245 s
    (**212 MB/s from HF**), load 11 s (page cache), smoke 48 s, peak VRAM 54 GB. Loaded with
    `Qwen3_5ForCausalLM`; 26.9 B params, 64 layers at `model.layers`, 3 linear : 1 full attention.
    GSM8K item 0 answered correctly (18) in 161 tokens — **too easy to elicit a long chain.**
    Residual stream cached at layers 0/15/31/47/63, [278, 5120], mean ‖h‖ 14 → 401.
    Artifacts: `mathematics/runs/2026-09-19_smoke/` (16 MB, incl. `pip_freeze.txt`).
  - **Global volume measured:** mounts at `/workspace-global` as `fuse.geesefs` (object storage
    behind FUSE — never put an HF cache or a venv on it). Write 244 MB/s; **read-back 141 MB/s,
    byte-identical** — slower than HF. 52 GB of weights now sit on `like_aquamarine_bandicoot`
    (~$4.70/mo) with a `DONE` marker; next session's setup restores from it and prints the real
    speed. Delete it if it loses again. Its real use: gated/private weights, fine-tunes, cached
    activations.
  - **Claude's errors tonight:** (1) copied the QWEN-01 setup without the causal-conv1d fix its
    own journal records (`CUDA_HOME` + `--no-build-isolation`); fla shared the pip line and died
    with it. Repaired in parallel with the download, now folded into `pod_setup_q38.sh`.
    (2) Did not record the opening balance; AG supplied it.
  - **What AG actually wants, in his words:** the barrier to research should not be infra;
    pod → weights → server every time is the frustration. Tonight's timings say weights are not
    the slow part (4 min); installs, kernel builds and servers are. **Fix = baked Docker image
    + saved template + one launcher taking a model name.** None of it needs a pod.
  - Nothing committed. `mathematics/` untracked, CLAUDE.md modified. Zero pods.

  **Next (no GPU): Dockerfile + GitHub Actions amd64 build → public image → template via API →
  launcher. Then a harder math item than GSM8K-0, and AG's one-sentence question after reading
  Kantamneni–Tegmark (2502.00873). HF token STILL not rotated.**

- **2026-09-19 (night) → 09-20 — INFRA FINISHED; FIRST STEPS ON THE ACTUAL QUESTION. $0.39 for the night.**
  Full detail `mathematics/experiments.md` → IMAGE-TEST-01, TRACE-01, TRACE-02, PARABOLA-01.
  Balance $51.63 → **$51.24** (settled; RunPod posts a pod's last minutes ~5 min late — never quote
  a closing balance straight after terminate). Zero pods.
  - **AG's objective, in his words, and it does not change:** understand how models do mathematics
    with NO tools / sandbox / agentic setup — "that was, that is, and that will always be" the goal.
  - **Infra, done and proven:** custom image `ghcr.io/adithyag73/interp-pod:torch28-tf5170`
    (repo **github.com/ADITHYAG73/interp-pod-image**, GitHub Actions build, `verify.py` gates
    versions + model classes for Qwen3.5/Qwen2/Gemma3/Gemma4/Olmo3). **One image DOES cover all
    these families for hooks work** — Claude's earlier "No" was stale journal evidence, corrected
    by a live check. Templates: `8miemycg5h` interp custom image (**default; create → results in
    3 min 58 s, zero installs, activations bit-identical to the stock-image run**), `31hq5fjfvy`
    stock-image fallback, `mf96kxlalr` vLLM server (**never deployed, untested**). Claude creates
    pods from templates via the API; the console is only needed for the global volume.
  - **Global volume `like_aquamarine_bandicoot`** still holds 52 GB. HF download was 50–245 s
    depending on data centre; volume read-back was 141 MB/s. Balance drifted a few cents with no
    pod; the API cannot show global-volume billing. **Check console Billing; likely delete.**
  - **UserPromptSubmit hook installed** (`~/.claude/hooks/epistemic_reminder.sh`): injects AG's
    standing rules on prompt 1 and every 5th. Confirmed firing. AG asked for it because he is
    tired of retyping "don't be a sycophant / don't hallucinate".
  - **TRACE-01/02 (free, Neuronpedia Circuit Tracer via API, key in `.env`):** graphs for
    36+59 on qwen3-4b and gemma-2-2b. Paper's `calc:` format gives Qwen only 25% on the right
    digit; `Question: … Answer: ` gives 81% (Gemma 87.5%). Both models: late-layer "say 9"
    output features visible; **no lookup / magnitude / carry feature found; ~24% of Qwen's graph
    is unexplained error.** We can see the model DECIDE, not HOW. Tools in `mathematics/pipeline/`.
    Circuit Tracer only exists for gemma-2-2b, gemma-3-4b-it, qwen3-4b, qwen3-1.7b (needs
    per-model transcoders) — that is why Qwen3-4B, not a retreat from Qwen3.8.
  - **PARABOLA-01 — AG's first own prompt, verbatim, Qwen3.8-27B, $0.25:** the model caught BOTH
    ambiguities unprompted (vertex direction; scale), wrote h²+k²=4 in thinking then dropped it,
    used the (y−k)²=4p(x−h) template with **p** not AG's **y²=4ax**, reconstructed "y²=4x" as
    what AG imagined, answered y²=−4(x−2) and y²=−4(x+2). 951 tokens, [1035, 5120] at 5 layers on
    disk, unexamined. **The five layers (0/15/31/47/63) were Claude's placeholder default, not a
    research choice — AG called that out.** All 64 is ~680 MB and trivial.
  - **Candidate experiment for AG's "does it visualise" question (Claude's proposal, his call):**
    Qwen3.8 is natively vision-language — compare internal state for the WORDS "parabola opening
    rightwards" vs an IMAGE of one, vs left-opening controls.
  - **Is addition solved? No** (checked): helix/Clock (2502.00873) vs bag of heuristics
    (2410.21272) vs Anthropic's lookup+magnitude vs "The Shape of Addition" (2606.03645, found,
    unread) — unreconciled, all ≤8B or one closed model, two operands 0–99, one forward pass.
    The Biology paper gives NO reason for choosing 36+59; it ran all 10,000 pairs.
  - **THE REAL PROBLEM, stated by AG twice tonight: "I still do not understand even one bit."**
    Claude explained results from a field whose basic objects (feature, layer, attribution) were
    never established with him — the Sept-7 "wrong dose" error again. **Agreed fix: read
    Kantamneni–Tegmark together, one piece per sitting, by hand, his pace. First sitting =
    Figure 1; first exercise = do 36+59 on the T=10 clock on paper.** He said "not now".
  - **Claude's errors tonight (all journaled):** answered a Neuronpedia question from the
    screenshot instead of the question; invented "a question you wrote down years ago" (AG never
    said years — conflated with the forking-paths note); zsh word-splitting sent host+port as one
    argument (watchdog self-test caught it); `--answer "-4a"` parsed as a flag and the completion
    grep missed the lowercase `error:` (a 1-second local parse check now runs first); quoted a
    closing balance before billing settled.
  - Nothing committed in this repo: `mathematics/` untracked, CLAUDE.md modified. HF token STILL
    not rotated.

  **Next: AG rests. Then, his call and in this order of value: (1) the paper, Figure 1, by hand;
  (2) his one-sentence question; (3) look at the parabola/GSM8K activations already on disk —
  free; (4) console Billing check → delete the global volume if it is the leak.**

- **2026-09-20 — MATHEMATICS: instrument built, AG driving it himself, prior work found. $0.44 for the day.**
  Full detail `mathematics/experiments.md` → NUMBERS-01, PROBE-01, and the **"WHERE THIS STANDS"**
  section at the end, written for a cold pickup. Balance $51.24 → **$50.86** (settled). Zero pods.
  - **AG's backward-learning principle RECOVERED** from the 2026-09-13 transcript and saved as
    memory `backward-learning`: concrete runnable thing first, then walk back from what he cannot
    explain. Claude had just written an 8-sitting forward reading plan for Kantamneni–Tegmark —
    the THIRD forward reading plan in two weeks, wrong every time. Method now: he looks, he asks,
    Claude answers one level down and stops. **It worked.** He went from "I still do not understand
    even one bit" in the morning to deriving, from a +1.00 he read off a plot, that tokenisation
    must precede the embedding lookup — and then demanding evidence for it before accepting it.
  - **Two artifacts he can drive:** model anatomy https://claude.ai/artifact/Uki6ToP96u5h6mjRrubyyQ
    (built from `arch.txt` + the official `modeling_qwen3_5.py`); the number instrument
    https://claude.ai/artifact/H7du5JUmGXjGpmGbemmo1k (90 numbers × 65 depths, pin a pair, per-pair
    depth curve). He found two real bugs in it: hovering was destroyed by moving to the slider, and
    "read in 36+" read as if 36 were fixed. Both fixed.
  - **NUMBERS-01 ($0.19, 7 min, no bugs):** 90 numbers 10–99, read at the SECOND digit token, all 65
    depths, two contexts. **PROBE-01 (free):** first digit is at chance (11.1%) at the embedding and
    **100% from block 1** — a linear-attention block, three before any full attention. AG predicted
    "linear" in advance. **Claude will not let that stand as support:** 100% at the very first block
    is the signature of the width-4 `conv1d` copying, not of assembly. Presence ≠ assembly ≠ use.
  - **Claude discarded its own first probe run in front of him** — its sanity control read 0.0% at
    every depth (impossible; split held out the test classes). Re-ran with a control that can pass,
    a shuffled control, and 8 splits. Right call, and he saw it made.
  - **CLAUDE WAS WRONG about novelty and corrected itself the same day.** Told AG "nobody has looked
    at digit-tokenised models", then on his instruction checked and found TWO papers that cover it:
    **arXiv 2510.26285** (ACL 2026, §4.1 is our exact question, 99% at offset −1, MIT code at
    `prompteus/numllama`) and **arXiv 2606.03645** (ICML 2026, carry geometry on Qwen3-4B/8B).
    **Verified from config.json:** every model in both is standard attention; the hybrid begins at
    Qwen3.5. So the gap is real but narrow — a transfer test, not a discovery.
  - **Probe caveat:** Claude wrote its own linear probe, NOT the paper's sinusoidal one, so our
    numbers are not comparable to their 99%. Use theirs (MIT repo) if comparability matters.
  - AG's own words on leaving: *"research is a rat hole… I'm not knowing where we are going."* Half
    justified — a day produced an instrument, real understanding, a confirmed-but-uninformative
    transfer, and one sharp question. Not a result yet.

  **Next, one decision: Qwen3-8B (standard) vs Qwen3.5-9B (hybrid), same probe, every block. If the
  standard model also hits ~100% at block 1, the linear-attention reading is dead — which is why it
  is the test worth running. ~$0.50. Queued behind it: PARABOLA-02 (words vs image) and e^x.
  HF token STILL not rotated.**

- **2026-09-20→22 — FIRST BLOG POST PUBLISHED. Rejected by LessWrong; shipped on my own site.
  Zero GPU spend.** Full detail `forking/experiments.md` → BLOG-01.
  **Live: https://adithyag73.github.io/first_principles/forking-tokens/** — third article on
  `first_principles`, after flash-attention and free-normalization. Commit `442e045`.
  - **The post is SWAP-01 + QWEN-02 + SCREEN-01 + QWEN-03 as one claim:** on a question
    Qwen3.5-9B is confident about nothing moves; on one it is torn about, six alternate tokens
    move the final answer and two flip the verdict. Dictated by me; Claude proofread only. All
    six claims about the two papers verified against live arXiv HTML (`blog01/PAPER_CHECK.md`).
  - **A Claude fact was wrong and the paper check caught it:** row 80 is *one of* the clearest
    forks in the Llama data, not the clearest — row 50 t=236 "false"→"true" is 0.990 vs 0.960.
    Row 80 is the largest among positions where the *pooled* curve is flat. Fixed everywhere.
  - **LessWrong rejected it with a template** (verified: 8,108 rejected posts carry the same
    text at lesswrong.com/moderation). Its one real point is Claude's error, not mine: the post
    never says why forking matters. Claude proposed the four-section shape and omitted the
    motivation section. **Still not written** — it needs my words, between "What the papers
    claimed" and "The two questions".
  - **Infra:** the active `gh` account on this machine is `IamAGP` (the forking-fast fork) and
    it 403s on `ADITHYAG73/first_principles`. `gh auth switch --user ADITHYAG73`, push, switch
    back. Both accounts are in the keyring.
  - Editor lesson: LessWrong's editor pastes markdown literally and drops list items; retyping
    as plain paragraphs and verifying character-for-character against the dataset was the only
    reliable path.

  **Next: (1) the motivation paragraph, in my words, then re-publish; (2) my hand check at
  t=164 (`out/s200/row041_branches.jsonl`, expect 175/25 and 105/92) — the only load-bearing
  number in the post I have not verified myself; (3) MATH-SMOKE-01 is still queued and read;
  (4) HF token STILL not rotated.**

- **2026-09-22/23 — TRIG-01: AG's own experiment. TWO MECHANISMS SEPARATED. ~$1.0–1.3.**
  Full detail `mathematics/experiments.md` → TRIG-01, TRIG-01b. Balance $50.59 → **$49.63**
  (provisional; RunPod posts the last minutes late). Pod terminated, **zero pods confirmed**.
  **AG's design, his words:** sweep the whole cycle in degrees, answers to exactly five decimal
  places, "let's use a calculator as well for the ground truth because mathematics is verifiable
  rewards". His framing set the analysis: he can do 0/30/45/60/90 and ASTC from memory but
  "can never do this in between things like 37 degree".
  - **A (thinking off, 1080 asks, 51 s):** **68.9% exact to 5 dp** — sin 86.9 / cos 80.8 /
    tan 38.8. **100%** on his memorised angles, **67.5%** on the in-between. Format 99.6%.
    Clean **quadrant gradient** (sin 98.9→96.7→76.7→75.6; tan 92.2→39.3→6.7→16.9).
  - **B (thinking on): 100.0% (213/213)**; paired on the same asks **87.8% → 100%**.
    The traces show it **states** sin 35° = 0.573576436 and rounds (no derivation), but for
    tan(95°) applies tan(90+x)=−cot(x), recalls tan 5°, inverts, and even starts a Taylor series.
    **Confound, stated not fixed:** `enable_thinking=True` ALSO injects an unrequested
    "Reasoning effort xhigh / think carefully" system message — B is CoT **plus** that.
  - **TABULATION GRADIENT (free, from data already collected):** mult-of-15 **94.3%** >
    mult-of-5 **84.7%** > even **67.6%** > odd **60.9%**; mult-of-5 vs not = 87.9 vs 64.2,
    **z = 6.68**, tan gap **+37 pts**. No computation account makes sin(35°) easier than sin(34°).
  - **TRIG-01b — the dissociation, and it CORRECTED Claude's own "it's mostly recall" reading:**
    `neg` (−d) **98.5%** · `int` 95.9% · `half` 70.7% · `dec2` (d+0.ab) **3.0%** ·
    `over360` (d+360) **4.4%**. But `dec2` is **75.6% at 1e-3** (computed, imprecise) while
    `over360` is 4.4% at *every* tolerance (value simply wrong). **85.7% of `over360` errors are
    an EXACT 5 dp entry for a DIFFERENT angle** (sin(361°)→sin(39°); tan(365°)→tan(45°)) vs
    **5.0%** for `dec2`. → **Retrieval gives the 5th decimal; the model's own calculator is worth
    ~3 decimals; `sin(−d)=−sin(d)` is applied perfectly; reduction mod 360 is absent.**
  - **Four checks that earned their place.** (1) The activation hook was saving the **last
    GENERATED** token, not the last prompt token (off by 3.08) — silent, caught only by an
    independent prompt-only forward pass; fixed to prefill-only, re-verified at 0.001.
    (2) Printing the two chat templates (not just asserting they differ) found the B confound.
    (3) Truncation flagged and excluded — B's lone "error" was tan(95°) hitting the 2048 cap after
    reaching −11.430052; A checked and **0/1080 truncated**. (4) **Pad control:** 87/90 identical
    answers at batch 1 vs 48, accuracy 77.8% either way — the 3 differences are all tan, all
    already wrong. Activations remain caveated (residual not padding-invariant, max 2.73).
  - **Claude's errors:** double-launch (two runs; the duplicate OOMed and died, no data lost);
    preflight memory estimate **wrong by 23 GB** (said ~57, actual 80.4 of 81.9) — the
    incremental-checkpoint rule, not the estimate, is what made it harmless; a "right magnitude,
    wrong quadrant sign" story generalised from the 8 largest errors that collapsed to **9.6%**
    when tested; budgeting from the catalog $1.19/hr when the pod bills **$1.59/hr**.
  - **PROBE-TRIG (free, CPU, on the cached activations):** the tenths digit of the answer is
    readable from the LAST PROMPT TOKEN — **20.8% at depth 0 (= the majority-class rate, i.e. no
    information) rising to 63.9% at depth 56**, against a **20.6%** majority baseline (NOT the
    10% the script first printed — digit 9 is 222 of 1078). **So the value is largely fixed in
    the forward pass before a character is emitted.** It does NOT show it "knew better than it
    said": true and said digits agree on 80.3% of rows, and the wrong-only subset is too small to
    train on. Caveat: these activations came from batches of 48 and the residual is not
    padding-invariant — re-measure at `--batch 1` (~90 s) before quoting in a write-up.
  - `.gitignore` now excludes `mathematics/runs/**/*.npy` (718 MB > GitHub's 100 MB limit).

  **Next: (1) AG reads the dissociation and decides what it means — the `dec2` "~3 decimals of
  real calculator" is the thread nobody has pulled; (2) the probe on the cached activations (free,
  CPU) asks whether the answer digit is present BEFORE generation; (3) still queued: PARABOLA-02,
  e^x, Qwen3-8B-vs-Qwen3.5-9B; (4) HF token STILL not rotated.**

- **2026-09-22/23 (night) — dashboard published; TRIG-02 started and STOPPED by me for a clean
  morning restart. Pod terminated, zero pods confirmed.**
  - **Dashboard (all 2,736 asks, model vs calculator vs error, filterable):**
    https://claude.ai/artifact/9gFKBqkC8Je2b55pukMdpb — source in `mathematics/dashboard/`.
    Artifacts cannot offer file downloads (sandbox blocks them), so there is no Excel export;
    raw JSONL is in `mathematics/runs/2026-09-22_trig/`.
  - **My question, answered: it is NOT a calculator.** Thinking off, ~1 answer in 3 is wrong, and
    it fails in a way arithmetic never does — sin(361°) returns sin(39°) exactly.
  - **TRIG-02 (does the scratchpad fix it) reached 24/345 and was stopped.** Early signal only,
    too small to quote: thinking got `dec2` angles right where thinking-off scored 3.0%, but ran
    to the 2048-token cap in 4 of the first 6, and `sin(4.14°)` came back **2.56** — outside
    sine's range. **Non-termination is a distinct failure mode from being wrong**, and the morning
    design separates them.
  - **MORNING PLAN (in `mathematics/experiments.md`): full 1° sweep, thinking ON, TWO passes** —
    pass 1 all 1080 at a 512-token cap; pass 2 re-asks only the truncated rows at 4096 in their
    own batches. Pairs within-angle against condition A already on disk, and the tabulation
    gradient under thinking falls out of the same data for free. ~50 min ≈ $1.30 (treat as a
    floor). Keep the variants question for a separate session — it is the expensive half.
  - **Three of Claude's errors tonight, all journaled:** (a) the "preflight memory estimate wrong
    by 23 GB" self-criticism was ITSELF wrong — KV at batch 24×2048 is **1.61 GB**, the original
    estimate; 80.4 GB was `nvidia-smi` **reserved**, and the same run with
    `expandable_segments:True` sits at **52 GB**, measured. (b) `pkill -f run2.sh` over SSH killed
    its own shell silently (the pattern matched the ssh command line), leaving an orphan and
    losing the script that held the remaining stages — diagnose by PID. (c) cost estimate wrong in
    the same direction twice: a generation-time estimate must be anchored to the HARDEST inputs,
    and a batch costs as much as its slowest member.
  - `mathematics/` is committed (`a5c6bac`); activation tensors stay gitignored (718 MB > GitHub's
    100 MB limit, regenerable in ~90 s).

  **Next (morning): the two-pass full thinking sweep, then the extensive discussion I asked for.
  HF token STILL not rotated.**
