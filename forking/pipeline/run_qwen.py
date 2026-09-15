"""QWEN-01: run the Forking Fast sampler on a Qwen3.5 dense model for one question.

Wraps reference_repos/forking-fast/forking_paths WITHOUT editing it. Patches:
  1. prompt: tokenizer.apply_chat_template(..., enable_thinking=False) — Qwen3.5 thinks by
     default; we turn it off so the run matches Llama-3-8B-Instruct's non-reasoning setup.
     (Thinking-on is a separate experiment, and the DeepSeek 'Other' problem shows why.)
  2. eos: add <|im_end|> (Qwen end-of-turn) to the stop set.
  3. device: --device cuda|mps|cpu (their class hard-codes cuda).
  4. model class: try AutoModelForCausalLM, fall back to AutoModelForImageTextToText and use
     its language model — Qwen3.5 checkpoints ship with a vision tower.
  5. report_progress: referenced in their run.py but never defined; stubbed.
  6. continuation TEXT is saved alongside outcomes (their release dropped it).
Everything else — branch enumeration at p>=0.05, S draws at T=1, regex+logit answer
extraction, o_t / o_tw — is their code, untouched.
"""
from __future__ import annotations
import os, sys, json, argparse, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# the sampler package: laptop layout (reference_repos/forking-fast) or pod layout (/root/forking-fast)
for _cand in (os.environ.get("FORKING_FAST_ROOT", ""),
              os.path.join(ROOT, "reference_repos", "forking-fast"), "/root/forking-fast"):
    if _cand and os.path.isdir(os.path.join(_cand, "forking_paths", "forking_paths")):
        sys.path.insert(0, os.path.join(_cand, "forking_paths")); break
else:
    raise SystemExit("forking-fast repo not found; set FORKING_FAST_ROOT")

import torch
import forking_paths.model as fpm
import forking_paths.prompts as fpp
import forking_paths.run as fpr
from forking_paths.config import ForkingConfig
from forking_paths.resample import enumerate_branches

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
ap.add_argument("--questions", default=os.path.join(ROOT, "forking/questions/row080.json"))
ap.add_argument("--out-dir", required=True)
ap.add_argument("--device", default="cuda")
ap.add_argument("--dtype", default="bfloat16")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--n-samples", type=int, default=200)
ap.add_argument("--n0-samples", type=int, default=200)
ap.add_argument("--base-max-tokens", type=int, default=400)
ap.add_argument("--cont-max-tokens", type=int, default=400)
ap.add_argument("--tok-depth", type=int, default=400)
ap.add_argument("--gen-batch", type=int, default=256)
ap.add_argument("--positions", default=None, help="comma list; restrict branching to these positions")
ap.add_argument("--only-forks", action="store_true", help="drop positions where only the greedy token cleared p_thresh (keeps t=0)")
args = ap.parse_args()

# ---- patch 1: Qwen chat prompt, thinking off ---------------------------------
def build_prompt_ids_qwen(tokenizer, question, choices, is_instruct=True):
    user = fpp.format_mmlu_instruct_user(question, choices)
    ids = tokenizer.apply_chat_template([{"role": "user", "content": user}],
                                        add_generation_prompt=True, tokenize=True,
                                        enable_thinking=False)
    if hasattr(ids, "input_ids"): ids = ids["input_ids"]   # transformers 5 returns BatchEncoding
    ids = list(ids)
    if ids and isinstance(ids[0], list): ids = ids[0]
    return ids
fpr.build_prompt_ids = build_prompt_ids_qwen

# ---- patches 2-4: model wrapper -----------------------------------------------
class QwenForkingModel(fpm.ForkingModel):
    def __init__(self, model_path, device, dtype, seed, gen_batch):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.device = device; self.seed = seed; self.gen_batch = gen_batch
        self.enable_prefix_caching = False
        td = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        try:
            m = AutoModelForCausalLM.from_pretrained(model_path, dtype=td)
            print("[qwen] loaded via AutoModelForCausalLM", flush=True)
        except Exception as e:
            print(f"[qwen] AutoModelForCausalLM failed ({type(e).__name__}: {str(e)[:120]}); trying ImageTextToText", flush=True)
            from transformers import AutoModelForImageTextToText
            m = AutoModelForImageTextToText.from_pretrained(model_path, dtype=td)
        self.model = m.to(device).eval()
        self.eos_ids = self._eos_ids()
        for t in ("<|im_end|>", "<|endoftext|>"):
            tid = self.tokenizer.convert_tokens_to_ids(t)
            if tid is not None and tid >= 0 and tid not in self.eos_ids: self.eos_ids.append(tid)
        self.eos_ids = sorted(set(self.eos_ids))
        print(f"[qwen] eos_ids={self.eos_ids} -> {[self.tokenizer.decode([i]) for i in self.eos_ids]}", flush=True)

# ---- patch 5 -------------------------------------------------------------------
fpr.report_progress = lambda **k: None

# ---- config --------------------------------------------------------------------
cfg = ForkingConfig()
cfg.n_samples, cfg.n0_samples = args.n_samples, args.n0_samples
cfg.base_max_tokens, cfg.cont_max_tokens, cfg.tok_depth = args.base_max_tokens, args.cont_max_tokens, args.tok_depth
if args.smoke:
    cfg.n_samples, cfg.n0_samples, cfg.base_max_tokens, cfg.cont_max_tokens, cfg.tok_depth = 3, 3, 60, 40, 60

os.makedirs(args.out_dir, exist_ok=True)
questions = json.load(open(args.questions))
model = QwenForkingModel(args.model, args.device, args.dtype, cfg.seed, args.gen_batch)
therefore_ids = model.tokenizer(fpp.THEREFORE_ABCD, add_special_tokens=False)["input_ids"]

# ---- patch 6: keep continuation text -------------------------------------------
_orig_resample = model.resample
_conts_log = []
def resample_logged(prefixes, n, max_tokens, temperature, seed=0):
    """Their resample() takes ALL prefixes in one call and prints nothing. Call it one
    prefix at a time so there is a timestamped progress line per branch (QWEN-01 lesson:
    an hour of silence on a billing GPU is not acceptable)."""
    outs, t_all = [], time.time()
    for i, pfx in enumerate(prefixes):
        t1 = time.time()
        o, _ = _orig_resample([pfx], n, max_tokens, temperature, seed + i)
        outs.append(o[0])
        lens = [len(c) for c in o[0]]
        print(f"[qwen] {time.strftime('%H:%M:%S')} branch {i+1}/{len(prefixes)} prefix_len={len(pfx)} "
              f"n={n} mean_cont_len={sum(lens)/len(lens):.0f} {time.time()-t1:.1f}s", flush=True)
    _conts_log.append({"prefix_lens": [len(p) for p in prefixes], "conts": [[model.decode(c) for c in o] for o in outs]})
    return outs, time.time() - t_all
model.resample = resample_logged

_orig_enum = fpr.enumerate_branches
def _enum(base, cfg):
    bs = _orig_enum(base, cfg)
    if args.positions:
        keep = set(int(x) for x in args.positions.split(","))
        bs = [b for b in bs if b.idx in keep or b.idx == 0]
    if args.only_forks:
        from collections import Counter
        n_at = Counter(b.idx for b in bs)
        bs = [b for b in bs if n_at[b.idx] >= 2 or b.idx == 0]
    print(f"[qwen] branches after filters: {len(bs)} at {len(set(b.idx for b in bs))} positions", flush=True)
    return bs
fpr.enumerate_branches = _enum

print(f"[qwen] model={args.model} device={args.device} S={cfg.n_samples} N0={cfg.n0_samples} "
      f"base_max={cfg.base_max_tokens} cont_max={cfg.cont_max_tokens} depth={cfg.tok_depth} smoke={args.smoke}", flush=True)
t0 = time.time()
for q in questions:
    rec = fpr.run_question(model, q, cfg, True, therefore_ids, diag_sample=50)
    rec["meta"]["model"] = args.model; rec["meta"]["thinking"] = False
    rec["continuations_text"] = _conts_log; _conts_log = []
    fn = os.path.join(args.out_dir, f"outcomes_row{q['row_id']:03d}.json")
    json.dump(rec, open(fn, "w"), ensure_ascii=False)
    print(f"[qwen] row={q['row_id']} L={rec['base']['n_response_tokens']} branches={rec['n_branches']} "
          f"gen={rec['timing']['gen_secs']:.1f}s regex_cov={rec['diagnostics']['regex_coverage']} -> {fn}", flush=True)
    print("[qwen] base text:\n" + rec["base"]["base_text"][:1500], flush=True)
print(f"[qwen] DONE {time.time()-t0:.1f}s", flush=True)
