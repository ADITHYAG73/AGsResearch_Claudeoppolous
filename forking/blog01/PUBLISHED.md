# Where this post lives

**Live:** https://adithyag73.github.io/first_principles/forking-tokens/
**Published:** 2026-09-22, third article on AG's own site.

## Source of truth

The HTML is in the site repo, not here:

    ~/Desktop/PythonProjects/blog_drafts/first_principles/forking-tokens/index.html
    ~/Desktop/PythonProjects/blog_drafts/first_principles/forking-tokens/figures/

Repo `https://github.com/ADITHYAG73/first_principles.git`, branch `main`, GitHub Pages
deploys on push. Commit `442e045`.

**Push note:** the machine's active `gh` account is `IamAGP` (the forking-fast fork). That
account gets 403 on this repo. Push with `gh auth switch --user ADITHYAG73`, then switch
back. Both accounts are in the keyring.

The prose is `POST_AG_WORDS.md` — AG's dictated words only, which is the version that went
to LessWrong. The site version is the same text with citations as real links, the two
questions as quoted cards, and the two figures placed inline.

## LessWrong

Submitted 2026-09-20, rejected with a templated message. Template confirmed via
lesswrong.com/moderation (8,108 rejected posts carry it). The one substantive point in it —
the post never says *why* forking matters — is real and is **not** addressed in this
version. It was my structural error: I proposed the four-section shape and left the
motivation section out of it. AG has not dictated one yet; when he does it belongs between
"What the papers claimed and what was open" and "The two questions".

## Not carried over

- The "why this matters" section above.
- AG's owed hand check at token 164 (count A/C over the 200 continuations in
  `out/s200/row041_branches.jsonl`; expect 175/25 for "It" and 105/92 for "However").
