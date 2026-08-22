# NLA inference — every step, with shapes

Target: **`kitft/nla-gemma3-12b-L32`**. Read from the official
`nla_inference.py` (`NLAClient`, `NLACritic`) and `nla/datagen/stage0_extract.py`,
not from the paper. Where the paper and the code disagree, **the code wins**.

## The constants for this model (from `nla_meta.yaml`)

| name | value | what it is |
|---|---|---|
| `d_model` | **3840** | vector width |
| `extraction_layer_index` | **32** (of 48) | which residual stream |
| `injection_scale` | **80000.0** | L2 norm the **AV** expects the vector at |
| `mse_scale` | **61.9677** = √3840 | L2 norm both vectors get before MSE |
| `embed_scale` | **61.9677** = √3840 | Gemma-only fix, see step 5 |
| `injection_char` / id | `㈜` / **246566** | marker token |
| left / right neighbour id | **236813** / **954** | marker validity check |

**TRAP:** `embed_scale` and `mse_scale` are both √3840 for Gemma and are
**completely unrelated quantities**. Same number, different jobs. `injection_scale`
(80000) is a third thing again. Confusing them is silent, not loud.

---

## Stage A — get the activation  (`stage0_extract.py`)

1. **text → tokens.** `ids [1, T]`
2. **forward the BASE model** (`google/gemma-3-12b-it`), `output_hidden_states=True`
   → `hidden_states[LAYER+1][0]` → `[T, 3840]`
   **`LAYER+1` because `hidden_states[0]` is the embedding layer**, so layer 32 lives
   at index 33. Off-by-one here fails silently.
3. **choose position p.** Candidates: `p >= _MIN_POSITION (=50)` and not a special
   token. Training data draws `positions_per_doc` (=10) of them with
   `rng.sample()`, seeded on `(seed, doc_id)`. *Not* the last-N tokens.
   → `h = hidden_states[33][p]` → **`[3840]` float32, RAW (unnormalized)**
   Docstring warning: first ~10 positions decode toward the training prior;
   rare high-norm spikes decode poorly (Qwen L20 newlines ~14k vs typical 100–170).

One row of the parquet = one (doc, position):
`n_raw_tokens` · `detokenized_text_truncated` (**the prefix — what claims get graded
against**) · `activation_vector [3840] raw` · `activation_layer` · `doc_id`

## Stage B — AV: activation → text  (`NLAClient.generate`)

4. **build the prompt.** `cfg.actor_prompt_template.format(injection_char='㈜')`,
   then `tokenizer.apply_chat_template([{role:user, content}], tokenize=True,
   add_generation_prompt=True)` — **one step**. Two-step tokenizing double-BOSes
   Gemma and shifts every position by one; silent on Qwen (`bos_token=None`).
5. **embed and arch-scale.** `embeds = embed(ids) * embed_scale` → `[1, T, 3840]`, cast fp32.
   For Gemma, `Gemma3TextScaledWordEmbedding.forward()` normally multiplies by √d, but
   `load_embedding_only` returns a plain `nn.Embedding`, so it doesn't happen — token
   embeddings come out **62× too small** and the injected vector dominates → garbage.
   `embed_scale` restores it. Qwen/Llama: 1.0.
6. **scale the activation.** `normalize_activation(h, injection_scale)` — L2-normalize
   (norm computed in fp32) and rescale to **80000**. → `v_scaled [1, 3840]`
7. **inject.** Find `p` where `ids[p]==246566` AND `ids[p-1]==236813` AND
   `ids[p+1]==954`. Overwrite `embeds[0, p] = v_scaled`. Count must match exactly or
   it asserts — *crash loud rather than silently serve the marker as text*.
   → `[T, 3840]` fp32, contiguous
8. **generate.** POST `input_embeds` to the SGLang server. **Never send `input_ids`
   too** — SGLang may use them for logprob bookkeeping while forwarding on embeds.
   Defaults: `temperature=1.0`, `max_new_tokens=200`, `skip_special_tokens=False`.
9. **extract** the text between `<explanation>` tags. No tags ⇒ truncated generation
   or model drift; it warns and returns the raw string.

## Stage C — AR: text → activation  (`NLACritic.reconstruct` / `.score`)

10. **build the AR prompt.** `Summary of the following text: <text>{explanation}</text> <summary>`
    Tokenize with **`add_special_tokens=True`** — the Gemma critic was trained with a
    BOS prefix (this template is a raw string, not chat-template-processed). Omitting
    BOS shifts position-0 meaning and degrades reconstruction everywhere.
11. **forward the truncated backbone** (first ℓ+1 = 33 blocks),
    `.last_hidden_state[0, -1]` → **final token** → `[3840]`
12. **value head** — `Linear(d, d)` → `ĥ [3840]`, raw and unnormalized.
13. **score.** L2-normalize **both** `ĥ` and `h` to `mse_scale` (61.9677 = √d), then
    `mse = mean((pred_n - gold_n)**2)`, `cos = cosine(pred_n, gold_n)`.

## Why MSE = 2(1 − cos)

Both vectors at norm `s`: `||p−g||² = 2s²(1−cos)`. Per-element mean = `2s²(1−cos)/d`.
Choosing `s = √d` makes `s²/d = 1`, so the mean **is** `2(1−cos)`. The multiply is
load-bearing — without it you'd get `≈0.0005`. The returned MSE is final; do not rescale.

| cos | MSE | reading |
|---|---|---|
| 1.0 | 0.0 | perfect |
| **0.9** | **0.2** | good decode — expect this on clean positions |
| 0.5 | 1.0 | mediocre |
| 0.0 | 2.0 | orthogonal |

### What `cos` actually is (the notation is sloppy)

`cos` is a **variable holding cosine similarity**, not a function call. The angle is
**θ = the angle between the reconstructed vector ĥ and the original h**, in 3840-D space.
You never compute θ — you get cos θ straight from the dot product:

    cos θ = (p · g) / (||p|| ||g||)

Derivation of the identity: ||p−g||² = ||p||² − 2(p·g) + ||g||². With both rescaled to
norm s: = 2s²(1−cos θ). MSE is the mean over d components → 2s²(1−cos)/d. Choose s = √d
and s²/d = 1 → **MSE = 2(1−cos θ)**. Verified numerically to 2e-16.

### The chance baseline — read every cos against this

Two RANDOM vectors in d=3840: **mean cos = 0.00004, sd = 0.0162** (theory 1/√d = 0.01614).
High-dimensional space is overwhelmingly orthogonal, so **cos = 0 IS chance**, with a very
tight spread. Therefore **cos = 0.90 is ~56 sd above chance** — an enormous signal, not a
modest one. Do not eyeball a cosine without this reference point.

2-D check by hand: p=(3,4), g=(4,3) → dot 24, |p||g| = 25 → cos = 0.96, θ = 16.26°;
normalise both to √2 → MSE = 0.0800 = 2(1−0.96). ✓

## What you canNOT get from one round trip

**FVE.** It is variance-explained against a corpus baseline (needs the mean activation
over many samples). One passage gives **MSE and cosine only**. `round_trip.py` prints
`mse_nrm` and `cos` for exactly this reason; EasyNLA ships
`scripts/compute_fve_baseline.py` for the population statistic.

## Fastest failure diagnoses

| symptom | cause |
|---|---|
| output is CJK, or describes a CJK character | injection failed — vector never landed; check marker ids and `injection_scale` |
| assertion "found 0 injection sites" | template drift or tokenizer version mismatch |
| cos ≈ 0 on every position | wrong layer index (forgot `LAYER+1`), or `embed_scale` missing on Gemma |
| poor decode at one position only | position < 50, or a rare high-norm activation |
| MSE ≈ 0.0005 | forgot the `mse_scale` multiply |
