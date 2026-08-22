# NLA inference — walkthrough cheat sheet

Companion to `nla_inference_pipeline.md` (the dense reference). This one is the
**teaching order**, with the demos that made each step click. Model:
`kitft/nla-gemma3-12b-L32`.

---

## The rule that unblocks the shapes

```
hidden_states[33]        →  [1, T, 3840]     LAYER      (layer 32 → index 33)
hidden_states[33][0]     →  [T, 3840]        BATCH      (drop batch dim, size 1)
hidden_states[33][0][p]  →  [3840]           POSITION   (one activation)
```

**Layer → batch → position.** The `[0]` is the batch index, NOT token 0.
`hidden_states[0]` is the *embedding output*, before any block — which is why
layer L lives at index L+1.

`ids` is `[1, T]` integers = WHICH token at each position.
`h` is `[T, 3840]` floats = WHAT THE MODEL COMPUTED at each position.
Parallel arrays over the same positions, holding different kinds of thing.

**T is the token COUNT, not a token id.**

---

## STEP 1 — text → tokens

```
"During the Middle Ages, the wild boar..."   (205 chars)
  → ids  [1, 42]
  → [2, 14521, 506, 12266, ...]
    ['<bos>', 'During', ' the', ' Middle', ...]
```
- **Position 0 is `<bos>` (id 2)**, added automatically. Nothing you wrote is there.
- Tokens carry their leading space: `' the'` ≠ `'the'`.
- Vocab 262,144; 7 special ids. The marker `㈜` (246566) is an *ordinary* token in it.
- ≈4.9 chars/token → median Ultra-FineWeb doc (2186 chars) ≈ 450 tokens.

## STEP 2 — forward the base model, take one layer

```python
hs = model(**ids, output_hidden_states=True).hidden_states[LAYER+1][0]   # [T, 3840]
```
- 48 blocks → **49** hidden states. Index 0 = embeddings. Layer 32 → **index 33**.
- The residual stream is an **accumulator**: each block *adds* twice (attention, MLP),
  never replaces. `h[p]` = embedding + contributions of blocks 0..31 at position p,
  having attended over everything up to p.
- Same token, different position ⇒ **different vector** (demo: cos 0.45 for the same
  token id at two positions).
- Off-by-one (`[32]` instead of `[33]`) is **silent** — same shapes, no error, cos
  collapses to ~0.2. Suspect the index before the method.
- `output_hidden_states=True` materialises all 49: 49 × T × 3840 × 2 bytes
  (≈169 MB at T=450).

### One Gemma-3 block
```
self_attn · mlp · input_layernorm · post_attention_layernorm
                · pre_feedforward_layernorm · post_feedforward_layernorm
```
- **Sandwich norms** (4, not 2): normalise before AND after each sublayer.
- Attention: q/k/v/o, no bias. **GQA** 16 query heads / 8 KV heads.
  **head_dim = 256, decoupled from hidden_size** (16×256 = 4096 ≠ 3840).
- MLP: **GeGLU** — `down(act(gate(x)) * up(x))`, `gelu_pytorch_tanh`, 3840→15360→3840.
- Params: attention 47.19M + MLP 176.95M ≈ **224.15M per block** (~79% MLP)
  × 48 + embedding 1.01B = **11.77B** ✓ "Gemma-3-12B".
- Attention type alternates 5 sliding : 1 full. Full layers = [5,11,17,23,29,35,41,47].
  **Layer 32 is sliding, window 1024** — irrelevant for ~450-token passages
  (window wider than sequence ⇒ sliding == full).

## STEP 3 — choose the position

```python
candidates = [i for i,tid in enumerate(ids) if i >= 50 and tid not in special_ids]
if not candidates: return []            # DOC SKIPPED
rng = random.Random(sha256(f"{seed}|{doc_id}"))
return rng.sample(candidates, k=min(10, len(candidates)))
```
- **`_MIN_POSITION = 50`** — "need enough left-context for the activation to be
  meaningful". Real, in the official repo, NOT in the paper.
- Positions are **random and scattered**, not the last N.
- Deterministic per `(seed, doc_id)` → parallel slices merge cleanly.
- The 42-token boar passage yields **[] — skipped entirely.** Need ≥51 tokens (~250 chars).
- Output `h = hs[33][0][p]` → `[3840]` float32, **RAW** (data-gen never normalizes).

**DECISION I OWE:** my experiment picks its own position policy. Recurrence (one of my
three signals) only means "stability of the readout" if positions are **adjacent** —
the paper uses the last 10 tokens. Scattered positions would measure something else.

## STEP 4 — build the AV prompt

```
template (from sidecar) .format(injection_char='㈜')
  → apply_chat_template(..., add_generation_prompt=True)   ONE STEP
  → 108 tokens
```
- `<bos><start_of_turn>user\n ... <end_of_turn>\n<start_of_turn>model\n`
- Marker lands at **position 93**, flanked by `'>'` (236813) and `'</'` (954) — the
  closing bracket of `<concept>` and the opening of `</concept>`. **That's why the
  neighbour check works**: a stray ㈜ in real text would never have that exact pair.
- Verified live: all three ids match the sidecar, exactly 1 occurrence. **Re-run this
  check before every session** — it catches tokenizer drift.
- Two-step tokenizing double-BOSes Gemma (silent on Qwen, `bos_token=None`).

**THE KEY IDEA:** the prompt is **fixed**. All 108 tokens are identical for every
activation ever decoded. The only thing that changes is one embedding row at index 93.

## STEP 5 — embed, and the Gemma ×62 fix

```python
embeds = (embed(ids) * embed_scale).float()      # [1, 108, 3840]
```
- `load_embedding_only` reads ONE tensor from safetensors (~2s / 2 GB) instead of
  loading the model (~30s / 24 GB) — and returns a **plain `nn.Embedding`**.
- Gemma's real class `Gemma3TextScaledWordEmbedding` multiplies by √d in `forward()`.
  The plain one doesn't ⇒ **all token embeddings 62× too small** ⇒ the injected vector
  dominates ⇒ garbage. `embed_scale` restores it by hand.
- Deliberate trade-off, not a bug: bypass the model's own code for a 15× speedup,
  pay by re-implementing what it did. Explicit registry, not a `startswith` match.

### The three scales — different KINDS of thing
| | what it is | set by | applies to | Gemma-3-12B | Qwen2.5-7B |
|---|---|---|---|---|---|
| `embed_scale` | **bug fix** (restores ×√d) | architecture — derivable | all 108 tokens | 61.97 | 1.0 |
| `injection_scale` | **distribution match** | training sweep — NOT derivable | 1 row | 80000.0 | 150.0 |
| `mse_scale` | makes MSE = 2(1−cos) | = √d_model | scoring only | 61.97 | 59.87 |

Gemma's residual stream is ~500× larger than Qwen's because Gemma inflates embeddings
by √d on the way in. **Never port a scale across models.** (Llama-3.3-70B: 30.0.)

**Why scale matters mechanically:** attention compares by dot product, which scales with
norm. Demo — one position at 62× norm took **80%** of the attention instead of its fair
17%. Compounded over 16 heads × 48 blocks, the prompt becomes inaudible.

## STEP 6 — apply `injection_scale`

```python
assert torch.isfinite(v_raw).all()                        # NaN/Inf guard
v_scaled = normalize_activation(v_raw.float().view(1,-1), 80000.0)

def normalize_activation(v, target):
    n = v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v / (n / target).to(v.dtype)
```
- `.view(1,-1)` → `[1, 3840]`, because injection expects `[N, d]` (N = number of sites).
- **THIS IS WHERE MAGNITUDE DIES.** Demo: inputs at ‖v‖ = 100 / 5000 / 13974 (a 140× span)
  all come out at **exactly 80000**, with `cos(in,out) = 1.000000`. Direction perfect,
  magnitude gone. A different direction stays different (cos 0.0071 ≈ chance for d=3840).
- Outputs from different input magnitudes agree to ~5e-7 relative — *equal to
  floating-point precision, NOT bit-identical* (different divisors round differently).
- `clamp_min(1e-12)` → an all-zero activation stays exactly zero instead of becoming NaN.
- `.float()` before `.norm()` is defensive: torch already upcasts reductions, and the
  activation arrives as float32 from the parquet anyway. bf16 *storage* costs ~3.6e-5
  relative, and `.float()` cannot recover that — it only guards the reduction.

## STEP 7 — inject (the surgical swap)

```python
out = embeddings.clone()                       # never mutate the input
for b, p in (input_ids == inj_id).nonzero():
    if p == 0 or p == seq_len-1: continue      # can't check neighbours at the edges
    if ids[b,p-1] != left_id or ids[b,p+1] != right_id: continue
    out[b, p] = vectors[vec_idx]; vec_idx += 1
assert vec_idx == vectors.shape[0]
return injected[0].contiguous().numpy()        # [108, 3840] fp32
```
- Verified live: **exactly one row changes (93).** All other 107 rows bit-identical,
  max change 0.0. Row 93 norm → 79999.98.
- **REPLACEMENT, not addition.** The marker's own embedding is discarded.
  (EasyNLA's Karvonen path *adds* — different scheme.)
- `.clone()` first: the caller's embeddings are never mutated.
- Failure mode A — marker present, neighbours wrong →
  `AssertionError: found 0 injection sites with correct neighbors, expected 1`.
  This is what fires on tokenizer drift or template change.
- Failure mode B — **marker appears twice** → the OFFICIAL code raises a bare
  `IndexError`, because `vectors[vec_idx]` is indexed with no bounds guard before the
  count assert. **EasyNLA guards this explicitly** ("fail with the diagnostic message,
  not a bare IndexError"). → a real 5-line PR for the official repo.


## STEP 8 — generate (SGLang server, over HTTP)

```python
sp = {"temperature": 1.0, "max_new_tokens": 200, "skip_special_tokens": False}
body = orjson.dumps({"input_embeds": embeds_np, "sampling_params": sp},
                    option=orjson.OPT_SERIALIZE_NUMPY)
POST {sglang_url}/generate
```
- The AV runs as a **separate SGLang server process** (localhost:30000) — hence
  `launch_av.sh` and "keep that terminal open". SGLang is used because it accepts
  **`input_embeds`**, which most servers don't. Without that the method is impossible.
- **NEVER also send `input_ids`.** "SGLang may use input_ids for logprob bookkeeping
  while forwarding on input_embeds, causing misalignment." Silent, not an error.
- **Payload: 1.66 MB per decode; the activation is 15 KB of it (0.93%).** The other
  99.07% is the same 107 prompt rows re-sent every call — unavoidable, the API takes
  embeddings not ids. 10k activations = 16.6 GB over loopback to deliver 154 MB.
- `orjson.OPT_SERIALIZE_NUMPY` avoids building 414,720 Python floats (~10 MB) per call.
- **`temperature = 1.0` → THE AV SAMPLES.** Same activation, decoded twice, gives two
  DIFFERENT explanations. **This is the "replay the match" mechanism from the cricket
  story** — K decodes of one activation is what H2's 1/sqrt(K) variance reduction rides on.
  Step 8 both creates the noise and supplies the only tool against it.
  → **DECISION I OWE:** `round_trip.py` overrides to `temperature=0.7`. Fine for a smoke
  test; for the experiment T changes how much variance exists to average. Pick on purpose.
- `skip_special_tokens=False` keeps tags intact so truncation is detectable.


## STEP 9 — extract the explanation

```python
EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)
m = EXPLANATION_RE.search(text)
if m is None:
    print("[NLAClient] WARNING: no <explanation> tags. ...")
    return text                      # returns the PARTIAL generation, does not raise
return m.group(1).strip()
```
- Output format: **bolded heading, colon, elaboration** — style inherited from the
  Opus-4.5 warm-start summaries, persists through RL. Opus NLA = 4-5 snippets;
  our Gemma AV's prompt asks for **2-3**.
- **No tags ⇒ truncation or model drift.** It warns but does NOT raise — the partial
  string flows on. WATCH FOR THAT WARNING. First knob: `max_new_tokens` (client
  default 200; `round_trip.py` also passes 200; a 5-snippet Opus explanation is ~280
  tokens, so 200 can truncate).
- **THE FREE DIAGNOSTIC:** `extract_explanation=False` returns the raw generation.
  "If ALL outputs are CJK, or describe a CJK char in English, injection likely failed."
  Why: a failed injection leaves row 93 as the real embedding of `㈜`, so the model
  dutifully describes a parenthesised Korean glyph. **The failure is legible — read the
  output, don't compute a metric.** First check on any run.

  **CJK = Chinese/Japanese/Korean.** The markers are literally Korean letters:
  `㈜` U+321C PARENTHESIZED HANGUL CIEUC U (Gemma), `㈎` U+320E PARENTHESIZED HANGUL
  KIYEOK A (Qwen) — Unicode block "Enclosed CJK Letters and Months" (U+3200–U+32FF).
  A failed injection leaves the model describing a Korean letter in brackets, so CJK
  output is not a heuristic — it is the correct response to the input it got.
  Marker chosen because it is: one token · absent from ordinary English text · not special.
  NOTE: the neighbour check guards against markers in *pasted/multi-turn text* — our AV
  prompt is FIXED, so for us it really guards tokenizer drift + template change.
  The code comment shows `㊗` while sidecars say `㈜` → **trust the id, not the glyph.**


## STEPS 10-11 — the AR forward

```python
prompt = "Summary of the following text: <text>{explanation}</text> <summary>"
ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]        # BOS matters
h   = backbone.model(ids, use_cache=False).last_hidden_state[0, -1]  # [3840]
```
- **No chat template.** Paper: "We did not use chat formatting for the AR prompt."
  Plain string. (AV gets BOS from `apply_chat_template`; AR from `add_special_tokens=True`.
  Two routes, same requirement. Qwen `bos_token=None` → silently doesn't care.)
  Omitting BOS on Gemma "shifts position-0 meaning → degraded reconstruction everywhere".
- **The AR is Gemma-3-12B with 15 of 48 blocks deleted.** Verified: its config says
  `num_hidden_layers: 33` (= layer 32 + 1); 33 x 224.15M + 1.01B embed = 8.40B params
  = 16.8 GB bf16, vs the actual 16.9 GB repo. ✓
- `[0, -1]` = batch 0, **LAST token** (the `>` of `<summary>`). The prompt ends
  mid-thought on purpose; the AR's answer is its hidden state at that final position —
  the same trick as the AV's extraction point, mirrored.

## STEPS 12-13 — value head, then score

```python
pred   = value_head(h)                       # Linear(3840,3840), ~14.7M params -> RAW
pred_n = pred / pred.norm() * mse_scale      # both to L2 = sqrt(3840) = 61.9677
gold_n = gold / gold.norm() * mse_scale
mse = ((pred_n - gold_n)**2).mean()          # == 2(1 - cos)
cos = pred_n @ gold_n / (|pred_n| |gold_n|)
```
- **Second place magnitude is discarded.** (First was injection.)
- cos 0.9 -> mse 0.2 (good) · cos 0 -> mse 2.0 (chance, in 3840-D) · cos 1 -> mse 0.
- **One round trip gives mse + cos ONLY. Not FVE** — FVE needs a corpus baseline.

---

# THE ROUND TRIP IN ONE BLOCK

```
passage -> tokens -> base model -> h[3840] RAW
  -> normalise to 80000 -> swap into row 93 of a FIXED 108-token prompt
  -> SGLang @ T=1 -> text between <explanation> tags
  -> "Summary of the following text: <text>...</text> <summary>"
  -> 33 blocks -> last token -> Linear(3840,3840) -> h_hat[3840]
  -> normalise both to sqrt(3840) -> mse, cos
```
**13 steps · 3 models · 2 normalisations · 1 swapped row.**

## THE 7 THINGS THAT ACTUALLY MATTER (everything else: look it up)
1. `hidden_states[L+1][batch][position]` — layer, batch, position
2. Positions must be >= 50
3. The prompt is fixed; exactly one row gets swapped
4. Three scales, three jobs, all read from the sidecar — never ported across models
5. Magnitude is discarded at injection (and again at scoring)
6. `temperature=1.0` -> resampling is the variance tool (H2 rides on this)
7. CJK output = injection broke


---

# ⚠ REVISIT LATER — do not let these slide

1. **The α / 75th-percentile statistics.** Understood the *voice-volume* analogy (too
   quiet = ignored, too loud = drowns the prompt). Have NOT internalised: why a
   percentile rather than a mean (answer: rare high-norm spikes — e.g. Qwen L20
   newlines at ~14k vs typical 100–170 — drag the mean above the 90th percentile), nor
   the sweep procedure (train AV on correct vs scrambled pairs; the **loss gap** measures
   how much information got through; pick the α maximising it; p75 is a post-hoc
   summary of where sweeps landed, NOT a derivation).
   → Practically: I never compute it. I read `injection_scale` from the sidecar.

2. **Architecture as second nature.** Layer-by-layer visualisation of the network is a
   separate, bigger study. Park it; it is not needed for this project.

3. **IDEA I HAD (parked, worth writing into Future Work):** the paper says the fixed
   α "should be replaced with a learned scale or an affine projection" and they
   "weakly recommend" it. Why not do it myself? → Because it needs TRAINING: change the
   map and the AV is out of distribution, so the AV must adapt too; and the AV samples
   text (non-differentiable) so it needs RL — the $1,500–5,000 loop. Out of scope for
   20 hours. **But it costs one sentence in the write-up and shows I read carefully.**

4. **Measure on the pod (closes a gap Claude admitted):**
   ```python
   raw = m.model.embed_tokens.weight.norm(dim=-1).median().item()
   print(raw, raw * 61.97, "vs injection_scale 80000")
   ```
   Tells me whether the injected vector is meant to be comparable to an ordinary token
   or deliberately much louder. Claude assumed the former and has not verified it.

5. **Magnitude is discarded TWICE** — normalised at injection (every injected vector
   ends at the same norm α) and again at scoring (both to √d). So the NLA structurally
   **cannot report how strongly something is represented, only which direction the
   state points.** Internal proportions survive; overall scale does not.
   → Belongs in the write-up's limitations.
