# Dashboard status — complete, 2026-10-31T00:44:47+05:30

The static dashboard is complete with the explicitly disclosed sampling below. It contains all 338 capture records (201 free, 137 forced), representing 201 unique prompts. It starts on `tay|sin x|2|hold`, forced. No model was run, no answer correctness was classified, and S3 was never written. Root `latent_cube/AGENTS.md` was deleted at the user's request; the session record is kept here.

## Deliverable and use

- `site/index.html` plus its relative data files: 438 files, **54,568,952 bytes** total; largest file **741,008 bytes**. Limits: 60,000,000 bytes total and 15,000,000 bytes per file.
- Upload/serve the entire `site/` directory together. Relative fetch needs HTTP hosting; opening the HTML through `file://` is not a supported preview route.
- For a local preview, run `python3 -m http.server 18765 --bind 127.0.0.1 --directory dashboard/site`, then open `http://127.0.0.1:18765/`. The test server from this session is stopped during cleanup.
- Eight views: overview/counts/sizes; prompt filters/full replies/every token; residual/mixer/MLP norm maps; logit lens/top-five/ranks; MLP neuron lists and cross-prompt values; attention; pairwise cosine; all-capture PCA.
- Native MathML renders explanatory formulae and ground truth. The full replies use KaTeX 0.16.11, pinned to cdnjs with an integrity hash, with MathML-only output and no external stylesheet. The only allowed external page request is that script.
- Per-capture files are gzip-compressed JSON. Cross-prompt files use gzip-compressed `LCB2`: little-endian fixed-point int32 values at scale 100000, byte-shuffled for lossless compression, plus a JSON header. This changes encoding only, preserving all retained five-decimal values. Both Python and the page decode the format.

## Opening figure and read accounting

Exactly recorded and supplied to every build invocation:

> AWS month-to-date net USD 0.00 for 2026-10 at 2026-10-30 23:20 IST, Cost Explorer, credits offset charges, may lag 24h; budget 15 GB S3 reads, about USD 2 maximum

- Accounted successful S3 payload: **13,455,579,081 bytes = 13.455579081 GB**.
- Conservative reserved S3 payload: **13,463,130,313 bytes = 13.463130313 GB**, below the 15,000,000,000-byte watchdog cap. Reservations count interrupted/failed/retried requests too; use this upper accounting figure when assessing the cap. HTTP headers and S3 listing payloads are not in the tensor/JSON GET byte ledger.
- S3 GET attempts accounted: 106,526. Public Hugging Face payload across initial loading and resumes: 7,693,459,646 bytes, separate from S3 accounting.
- Only the supplied temporary environment credentials were used. AWS config/credential files and metadata credential fallback were disabled; no `~/.aws` access.
- Source inventory: 1,322 objects; prefix size 153,501,787,048 bytes; the 338 included JSON/safetensors pairs total 106,002,764,155 bytes. Superseded records and dry runs were excluded from analysis, while their sizes are shown in the prefix overview.
- The watchdog logged every 30 seconds, reserved bytes before requests, blocked budget overruns, and stopped reads after prolonged lack of progress. Numerical thread limits were four; network workers were sixteen. Explicit transient connection errors have at most two retries, each separately reserved.

## Numerical and structural checks

- Mandatory final-layer logit gate: **PASS**, six recorded runs with identical results. All **52 tokens × 248,320 vocabulary entries** of the initial forced capture were compared.
- Maximum absolute logit difference: **0.06486892700195312**; RMSE: **0.005661397064948961**; maximum absolute saved logit: **31.625**; relative maximum: **0.20511913676506915%**.
- Original tolerance remained unchanged throughout: maximum ≤0.5, RMSE ≤0.05, and relative maximum ≤1%. Projection was float32 from stored F16 activations and original BF16 weights. It was not bit-exact, and the difference is explicitly reported.
- `validate.py`: exit 0, all **338 captures** and **96 layer/readout summaries** checked for shapes, coverage, ordering, probability/rank bounds, encoding, and publication constraints.
- Maximum attention row-sum error: **0.002946000000000115**. Largest allowed rounding bound: **0.00436006862745098**. Maximum future-token attention weight: **0**.
- The attention bound accounts for bf16 unit roundoff (1/256) and six-decimal display rounding. The official Transformers v5.17.0 `eager_attention_forward` casts float32 softmax back to the query dtype. An initial arbitrary diagnostic threshold of 0.002 was corrected using that source; this did not change the logit gate or the data.
- Maximum cosine diagonal error: **0**; maximum cosine symmetry error: **0**.
- **96** float64 direct-dot cosine references from actual temporary R1 readouts: maximum exported difference **5.494328122801484e-06**, below the predeclared **0.000006** tolerance. These are implementation checks, not independent scientific validation by the researcher.
- Maximum per-capture versus cross-prompt neuron-value difference: **0**, across all 2,076,672 retained cross-prompt neuron values.
- Maximum absolute PCA coordinate mean (diagnostic): **7.988165681248335e-07**.
- All 201 local prompt IDs mapped to unique capture filenames; initial filename verified. Final Python and JavaScript syntax checks passed.

## Retention and sampling — also stated on the page

1. **No prompt, reply, or token-text subsampling:** all 338 records, 201 unique prompts, and **33,551 token positions**. Every free/captured reply is retained in full. Both variants remain in comparison and PCA.
2. **Norm maps (R1/R2/R5):** all 32 layers, first 128 token positions plus `instr_end`, `prompt_end`, and `reply_end`. **24,811 / 33,551 positions** retained; **65 captures** have omitted norm-map positions. A 129-token capture can still be complete because the final readout adds its final token. Omitted cells are explicitly unavailable, never zero-filled.
3. **Logit lens and MLP top-20 lists:** all 52 positions and all 32 layers for the initial forced sin/second-term/hold capture. For the other 337 captures, only the three named readouts. **1,063 retained token positions**, or **34,016 layer/token cells**. MLP lists select the 20 largest absolute values from all 9,216 neurons, retaining signs. Lens top-five probabilities and the seven specified ranks use the full 248,320-token vocabulary.
4. **Attention:** all query tokens for the initial capture; the three named readout query rows for every other capture. All key tokens, all 8 full-attention layers, and all 16 heads retained. Total retained query positions: **1,063**. Noninitial views are explicitly sampled query rows, not full T×T matrices.
5. **Across-prompt neurons:** 64 evenly spaced neuron indices, `0, 144, 288, …, 9072`, for every capture, all 32 layers and all three readouts. This is an index sample, not an importance ranking; other neuron indices are unavailable in this view.
6. **Rounding:** norms 3 decimal places; attention 6; neuron values 5; probabilities 7; cosine and PCA coordinates 5. Displayed cell percentages may be shorter, with detailed values on focus/hover/click. Fixed-point packing introduces no additional rounding.
7. **No PCA or prompt-comparison subsampling:** centered raw 2,560-dimensional R1 vectors; separate fit for every layer/readout across all 338 captures, including both variants. S (DeltaNet memory), R0, raw residual arrays, raw MLP arrays and full vocabulary logits are not published. R0 contributes source-size accounting only.

## Browser and text checks

- Isolated Chromium 145 / Playwright 1.58.2 tested at **1440, 390 and 320 px**, in **light and dark** mode: all six cases passed, including explicit theme overrides and UTF-8 checks.
- **0** page-level horizontal overflow cases, **0** JavaScript exceptions, **0** unapproved external requests. Six allowed cdnjs script loads across the six contexts; no external stylesheets.
- Verified the default 52-token capture, map/list shapes, all five filters, rank/top-five display, norm-cell details, attention layer/head selection, neuron/readout changes, PCA controls and point details, same-capture cosine identity, and the default paired comparison.
- The longest capture, `tay|sin(2x)|4|imagine` free, retains **877 tokens**, both full reply fields, exact unavailable-cell counts, and no page overflow at all three widths.
- All **676 reply fields** were reconstructed exactly from the rendered page fragments: **0 preservation failures**. **3,077 mathematical expressions** rendered with **0 parse errors**; no unrendered LaTeX commands found by the formatting check. No answer grading was performed.
- Desktop and phone screenshots were inspected. Visual review caught and fixed missing UTF-8 metadata, then confirmed correct symbols, responsive cards and phone-width equations.
- One attention-control test initially selected the visible numeric label “7” rather than option value 7; the test was corrected to select by value, and layer 31/head 15 passed. This was a test selector issue.

## Fixes and continuation record

The prior session wrote an unpopulated viewer and was blocked on the opening figure. This session recorded the supplied figure and ran the unchanged check-only gate first. Full-data reads would have needed 16.49 GB for R1/R2/R5 alone, so the disclosed first-128-plus-readouts sampling was necessary to respect the 15 GB cap. I/O and lens batches were improved without changing calculations. One connection-closed error stopped an attempt after 77 completed captures; bounded retries and resume completed the data.

The first complete export failed the 60 MB gate at 66,010,020 bytes. The first packing revision still measured 63,479,253 bytes. Byte-shuffled fixed-point encoding and gzip level 9 brought the complete site below the limit without reducing retained data or precision. Local `--compact` and validation completed successfully; no additional S3 reads were needed for packaging, UI fixes or validation. The full-build path now calls the same finalization automatically.

Detailed evidence is in `check-only.log`, the `export*.log` files, `read_ledger.json`, `validation.log`, `validation_report.json`, `browser-check.log`, `browser_report.json`, `math_report.txt`, `cosine_reference.json`, and `browser-*.png` screenshots. Preserve these debug records.

## Rerun

`build_data.py` requires numpy, requests, boto3 and threadpoolctl. Use the same opening figure with `--opening-balance`; the persistent ledger must not be reset to bypass the budget. `--check-only` runs the unchanged gate. Normal mode resumes completed per-capture files, reconstructs the temporary readout cache, finishes cross-prompt files, compacts and checks size. `--compact` is local-only and refreshes the compact files/manifest without network access. Run `python3 dashboard/validate.py` afterwards.

`check_browser.cjs` requires Playwright 1.58.2 and a local static server on port 18765. Set `PLAYWRIGHT_MODULE` if Playwright is installed outside the project, and `PLAYWRIGHT_BROWSERS_PATH` for its browser cache; pass the absolute site path. The temporary installation used during this session is removed after verification. Browser reports and screenshots remain under dashboard, outside site.

## Limits of verification and cleanup

- **Not published to or tested inside claude.ai.** That host's asset routing, CSP and MathML handling remain unverified.
- Browser testing covered isolated Chromium, not Safari or Firefox, and did not exhaust every possible interactive selection.
- The full-vocabulary numerical logit gate covered the initial forced capture only, not every capture.
- Capture acquisition, all scientific conclusions, and model-answer correctness were not independently validated. This dashboard makes no claim that a hidden mathematical term is present.
- Actual AWS billing was not re-read; the opening Cost Explorer figure was read and supplied by Claude (not by AG) and may lag.
- No raw embedding or activation cache files remain in `/tmp`; no GPU/pod/model run was started. No root AGENTS.md remains. Temporary test dependencies/scripts and the local test server are cleaned up after the final checks.

Official implementation references consulted: [pinned Qwen model](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a), [Transformers 5.17.0 Qwen3.5 code](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py), [KaTeX options](https://katex.org/docs/options.html), [Playwright configuration](https://playwright.dev/docs/test-use-options).

## Claude's review, 2026-10-31
- Independent check `independent_check.py` (separate code, raw S3 tensors): 6 random captures, 132 comparisons of norms, MLP top-20, attention rows, layer-31 lens against saved logits, mid-layer lens probability ratios from embedding rows, replies and tokens: **0 mismatches**.
- Published to claude.ai from `site_pub/` (made by `make_publish.py`: .bin is not a served type, so each file is base64 .txt; bytes asserted identical; loader patched). https://claude.ai/artifact/Vhkj2EQGXtyufVUcUbCKYe
- On claude.ai: first view loads with real data. Scrolling and section links could NOT be tested: browser automation does not reach inside the artifact frame (same result on a control page). Interactions rest on Codex's local Chromium test only.
