# dimuon/ — what does Tinker's Dimuon optimizer do to a LoRA?

Empirical study. The coordinator (main session) designs experiments; sub-agents implement and run
them; the coordinator reviews. Everything here is measured through ordinary documented Tinker SDK
calls on the owner's account.

## Ground rules for every agent

- **Report what you measured, not what you expected.** Every claim in your report needs the number
  and the file it came from. If a check fails or is skipped, say so. A negative or messy result is
  a fine result. Do not round a mismatch into "matches".
- **Noise floor first.** Before interpreting any difference, measure how much the same call differs
  from itself when repeated.
- **Small and cheap.** Base model `Qwen/Qwen3.5-4B` unless told otherwise. Prompts under 200 tokens.
  Checkpoints saved with `ttl_seconds=3600`. No more than ~150 training clients per agent task
  without asking the coordinator. No tight polling loops; run calls sequentially or with at most
  4 concurrent clients.
- **Stay in your lane.** Write only under `dimuon/`. Do not commit or push; the coordinator does.
  Do not read or use any third-party reimplementation of Dimuon from the internet; the point is an
  independent measurement.
- Append a dated entry to `dimuon/experiments.md` for each experiment: question, exact setup,
  result with numbers, caveats, files.

## Environment

- Python: `/tmp/claude-0/-home-user-AGsResearch-Claudeoppolous/31484cee-6902-5b45-bc20-3fc4fb38db73/scratchpad/venv/bin/python`
  (tinker 0.39.0, numpy, safetensors; `pip install` more into it if needed, e.g. scipy/torch-cpu).
- Auth: run with `TINKER_API_KEY=tml-proxy-injected`. The network proxy injects the real key; never
  print, look for, or handle a real key.
- SDK source is readable in the venv's `site-packages/tinker/` — read it rather than guess.
- `dimuon/pipeline/pilot.py` is a working end-to-end example; `dimuon/pipeline/billing.py` reads usage.

## What is known (measured, PILOT-01, 2026-11-10)

- `create_lora_training_client(..., optimizer=types.DimuonOptimizerConfig())`, then
  `optim_step(types.DimuonParams(learning_rate=..., beta1=0.9, grad_clip_norm=0.0))`. Those three
  are the only per-step knobs. AdamW is `types.AdamParams(learning_rate, beta1, beta2, eps,
  weight_decay, grad_clip_norm)`.
- `save_weights_external(name)` + `RestClient.get_external_weights_urls(path)` exports a PEFT-format
  `adapter_model.safetensors` (float32). rank 4, attention-only on Qwen3.5-4B gives 304 tensors:
  `lora_A` is (r, d_in), `lora_B` is (d_out, r), `lora_alpha` = 32 in the config.
- At init `lora_B` = 0 and `lora_A` is random with Frobenius norm ≈ 1.16.
- One Dimuon step at lr = 1e-3, rank 4: every `lora_A` is exactly unchanged; every `lora_B` update
  has Frobenius norm ≈ 0.00139 regardless of d_out (1024…8192), and its 4 singular values are
  close to each other but not equal (e.g. 0.000737, 0.000731, 0.000698, 0.000606).
- The SDK also has `RestClient.external_weights_upload_start/finish` (experimental: upload a
  safetensors and `load_state` it — a way to choose A and B) and `forward_backward_custom` (custom
  loss — a way to shape the gradient). Neither has been tried yet.
