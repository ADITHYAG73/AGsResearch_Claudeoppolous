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
