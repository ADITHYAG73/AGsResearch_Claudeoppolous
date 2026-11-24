# Global working preferences (apply to every project)

1. **Official sources only when in doubt.** Consult the **official GitHub repo** and **official
   documentation**. NEVER rely on third-party blogs (Medium, etc.).
2. **Debug, never assume.** Diagnose with real evidence; **leave debug logs** while investigating.
3. **Confirm, then fix — following OOP + SOTA (state-of-the-art) practices**, backed *only* by the
   official GitHub repo and official docs.
4. **Research tasks: official arXiv papers are mandatory.** Read the paper; if it names a GitHub
   repo, use that repo too. Don't do research off blogs.
5. **Never be a sycophant; never hallucinate.** Push back honestly when warranted; ground every
   claim in a verifiable official source.
6. **Vet source quality (web search).** Don't just confirm a source is *official* — verify it's
   **authentic, actively maintained, and from a reputable source** (top AI labs / well-regarded
   maintainers). Double-check claims against the current state of the art via web search.
7. **Rented GPU = lab practice, every time.** Before any billed run (sampling, finetuning, RL,
   agentic finetuning, anything): read the code path that will run and write four answers
   into the experiment journal BEFORE the machine exists — (1) timestamped progress line per
   unit of work? (2) results written to disk incrementally, not once at the end? (3) does
   working memory grow through the run, and what is the peak vs the card? (4) if killed at
   80%, what survives? Wrap third-party research code to add logging and checkpointing; never
   assume a lab's release has them. Red flags are raised while READING, not after launch.
   Run an idle watchdog alongside every pod. Terminate and confirm zero pods at the end.
   (Adopted 2026-09-14 after a silent, uncheckpointed 40-minute run on a billing A40.)
8. **Claude in Chrome is a first-class tool, not an afterthought.** It sits alongside web
   search, GitHub search, docs search and arXiv. When those don't fit or can't reach the
   content (client-side-rendered pages, logged-in sites, Twitter/X posts and threads, Gmail,
   API billing/balance dashboards, anything WebSearch/WebFetch returns empty or blocked for),
   open the site in Chrome yourself: navigate, scroll, click through links, read the page.
   NEVER answer "I couldn't fetch it", "it's only rendered client-side", or "open it and tell
   me what it says" — reach for Chrome first. Use the lighter tools when they work; Chrome is
   the fallback that must actually be used, not skipped. (Adopted 2026-10-09.)
9. **Every rented machine feeds the personal compute dashboard.** The user keeps one public
   compute dashboard across ALL projects (RunPod, AWS EC2, Kaggle; repo `compute-dashboard`,
   GitHub Pages). It reads the providers' APIs, so usage is captured automatically; the only
   per-session duty is naming: name every pod/instance `<project>-<purpose>` (e.g.
   `gemma4-exp052-sft`, `ntl-v42-coverage`) and tag EC2 with `Project=<project>`, so the
   per-project breakdown stays clean. Never put project data, task content or secrets on the
   public page: aggregates only (hours, GPU type, region, cost, carbon estimate).
   (Adopted 2026-10-29.)
10. **Never truncate. Not Claude's decision to make.** Do not cut, shorten, preview or cap
   anything on your own: no `[:60]` log previews, no `head`/`tail`/`cut -c` on output you then
   reason from, no low generation caps (`max_new_tokens`) that stop a model mid-answer, no
   summarised tool results passed off as the whole. Print, save and read things in full. If
   something is genuinely too large to show at once, read ALL of it in pieces and say so — never
   draw a conclusion from a partial view. A model's generation runs until it stops on its own; a
   safety cap must be far above any realistic answer and every hit must be reported, not hidden.
   (Adopted 2026-10-29, after Claude told the user a model "never answered Paris" from a
   60-character log preview when the full reply did answer Paris.)
