---
name: mathematics-curiosity-thread
description: AG's standing research objective (how LLMs do mathematics with no tools) — his queued experiments, the infra that exists, and how to teach him.
metadata:
  type: project
---

AG's objective, stated 2026-09-20 and not to be re-litigated: understand how models do mathematics with NO tools, sandbox or agentic setup. "That was, that is, and that will always be" the goal until answered. He owns the questions ([[one-experiment-at-a-time]] applies).

**Live state (2026-09-20):** read `mathematics/experiments.md` → the **"WHERE THIS STANDS"** section at the end first; it is written for a cold pickup and names the one decision waiting (Qwen3-8B standard vs Qwen3.5-9B hybrid, same probe, every block). Prior work EXISTS for the representation question (arXiv 2510.26285 §4.1, 99%; arXiv 2606.03645) — Claude wrongly said it was open ground and corrected this on AG's instruction; always check novelty before designing. **Also queued by him:** (A) PARABOLA-02, does Qwen3.8-27B "visualise": words vs image of a right-opening parabola, left-opening controls; (B) EXP-01, can models compute e^x and how. Neither is scoped; scope WITH him before any pod.

**Infra exists, do not rebuild:** image `ghcr.io/adithyag73/interp-pod:torch28-tf5170` (repo ADITHYAG73/interp-pod-image), RunPod template `8miemycg5h`, create-pod via API, ~4 min to results, ~$0.15–0.25 a run. Free route for small models: Neuronpedia Circuit Tracer via API (`mathematics/pipeline/np_graph.py`, key in `.env`).

**Why:** three days went to infrastructure before any step on the question; he noticed and said so.

**How to apply:** He said twice "I do not understand even one bit" — results were explained in a vocabulary never established with him. Teach by his backward method ([[backward-learning]]): concrete runnable thing first, then walk back from what he cannot explain; define a word before using it; NO forward reading plans. Kantamneni–Tegmark arXiv 2502.00873 is a reference to dip into, not a syllabus (Claude read it in full 2026-09-20; text in `mathematics/papers/`). Verify before claiming ([[check-before-you-claim]]); before any billed run follow [[gpu-run-preflight]] and [[gpu-idle-watchdog]], record the opening balance, and run a 1-second local parse check of the exact command line. Never quote a closing balance until ~5 min after terminate.
