---
name: gpu-idle-watchdog
description: Never leave a RunPod pod unwatched — start the idle watchdog alongside every GPU session, because orchestration bugs hang silently and bill indefinitely.
metadata:
  type: feedback
---

Whenever a RunPod pod is running, **start `mats_2027/pipeline/pod_watchdog.sh` in the
background alongside it**. It polls GPU utilisation and shouts if the card sits idle while a
pod is billing.

**Why:** on 2026-08-27 an orchestration script hung on an ssh line that backgrounded the
sglang server (`setsid nohup ... &` inside the ssh command — the server inherited the
connection's file descriptors so ssh never returned). The script never reached its own
health-check loop, which had a 15-minute cap. **With no timeout on that line it would have
hung indefinitely**, billing $0.44/hr. It was caught only because AG watched the RunPod
console and asked why the GPU had been at 0% for a while. His words: *"what if I had not
asked the question — then we would have sat forever, the mere thought scares the shit out
of me."*

This was the THIRD scripting bug in the project to waste idle GPU time (a truncated SSH key
passed from a `head -c 40` preview; a poll loop comparing `grep -c` to 1 when `set -x` prints
the line twice; this launcher). Individually cheap, together a pattern: remote-orchestration
scripts get written once and not reasoned about.

**How to apply:**
- Start the watchdog with every pod. Do not rely on remembering to check.
- Put `timeout N` on remote calls that could block; ssh's ConnectTimeout does NOT bound
  command execution.
- Launch long-running remote processes from a standalone script (as `pod/launch_av.sh` does),
  then poll from a SEPARATE connection. Do not inline the launcher.
- Every loop and launcher needs one line answering "how would I know this worked?" — the
  `launched` echo never printed, and that evidence was in the log, read past twice.
See [[check-before-you-claim]].
