"""SCREEN-01: per-question o_0 and greedy accuracy for Qwen3.5-9B (thinking off) on tinyMMLU.
For each question: 1 greedy answer + S0 sampled answers from the prompt alone (t=0, T=1).
Extraction = same rule as fpa_vllm.py (explicit 'answer is' regex, else logit read).
Checkpoint per question to <out>/screen.jsonl; --resume skips done rows; progress line per question."""
from __future__ import annotations
import os, sys, json, time, argparse, collections
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _c in (os.environ.get("FORKING_FAST_ROOT", ""), os.path.join(ROOT, "reference_repos", "forking-fast"), "/root/forking-fast"):
    if _c and os.path.isdir(os.path.join(_c, "forking_paths", "forking_paths")):
        sys.path.insert(0, os.path.join(_c, "forking_paths")); break
from forking_paths.answers import parse_mmlu_answer, VALID, _COMPILED
from forking_paths.prompts import format_mmlu_instruct_user, THEREFORE_ABCD

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B"); ap.add_argument("--questions", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--S0", type=int, default=10); ap.add_argument("--max-tokens", type=int, default=1500); ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.90); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-questions", type=int, default=None); ap.add_argument("--chunk", type=int, default=10)
    args = ap.parse_args(); os.makedirs(args.out_dir, exist_ok=True)
    LOG = lambda *a: print(f"[screen {time.strftime('%H:%M:%S')}]", *a, flush=True)
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    llm = LLM(model=args.model, dtype="bfloat16", seed=args.seed, gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    eos = sorted({i for i in (tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")) if i is not None and i >= 0})
    therefore = tok(THEREFORE_ABCD, add_special_tokens=False)["input_ids"]
    letter_ids = {L: tok(L, add_special_tokens=False)["input_ids"][0] for L in VALID}
    def prompt_ids(q):
        ids = tok.apply_chat_template([{"role": "user", "content": format_mmlu_instruct_user(q["question"], q["choices"])}], add_generation_prompt=True, tokenize=True, enable_thinking=False)
        if hasattr(ids, "input_ids"): ids = ids["input_ids"]
        ids = list(ids); return ids[0] if ids and isinstance(ids[0], list) else ids
    def extract(prefix, outs):
        conts = [dict(token_ids=list(o.token_ids), text=o.text, finish=o.finish_reason) for o in outs]
        ans, miss = [], []
        for i, c in enumerate(conts):
            a = parse_mmlu_answer(c["text"]); explicit = any(cre.search(c["text"]) for cre in _COMPILED[:4])
            c["regex_kind"] = "explicit" if (a and explicit) else ("option_mention" if a else "none")
            if a is not None and not explicit: a = None
            ans.append(a)
            if a is None: miss.append(i)
        if miss:
            sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20, seed=args.seed)
            res = llm.generate([TokensPrompt(prompt_token_ids=prefix + conts[i]["token_ids"] + therefore) for i in miss], sp, use_tqdm=False)
            for i, o in zip(miss, res):
                lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
                best, bl = None, -1e9
                for L, tid in letter_ids.items():
                    if tid in lp and lp[tid].logprob > bl: best, bl = L, lp[tid].logprob
                ans[i] = best; conts[i]["fallback"] = True
        return [a if a in VALID else "Other" for a in ans], conts
    qs = json.load(open(args.questions))
    if args.max_questions: qs = qs[:args.max_questions]
    ck = os.path.join(args.out_dir, "screen.jsonl"); done = set()
    if args.resume and os.path.exists(ck):
        done = {json.loads(l)["row_id"] for l in open(ck)}; LOG(f"resume: {len(done)} rows done")
    todo = [q for q in qs if q["row_id"] not in done]
    LOG(f"{len(todo)} questions to screen; S0={args.S0}; upper bound {(len(todo)*(args.S0+1)*args.max_tokens)/1e6:.1f}M tokens")
    t_all = time.time(); CH = args.chunk
    with open(ck, "a") as f:
        for ci in range(0, len(todo), CH):
            chunk = todo[ci:ci+CH]; t1 = time.time()
            ps = [prompt_ids(q) for q in chunk]
            # one generate call per phase per chunk: CH greedy + CH*S0 sampled sequences in flight
            g_out = llm.generate([TokensPrompt(prompt_token_ids=p) for p in ps],
                                 [SamplingParams(temperature=0.0, max_tokens=args.max_tokens, stop_token_ids=eos, seed=args.seed) for _ in chunk], use_tqdm=False)
            s_out = llm.generate([TokensPrompt(prompt_token_ids=p) for p in ps],
                                 [SamplingParams(n=args.S0, temperature=1.0, top_p=1.0, top_k=-1, max_tokens=args.max_tokens, stop_token_ids=eos, seed=args.seed + q["row_id"]) for q in chunk], use_tqdm=False)
            for q, p, go, so in zip(chunk, ps, g_out, s_out):
                ga, gc = extract(p, go.outputs); sa, sc = extract(p, so.outputs)
                rec = dict(row_id=q["row_id"], answer_letter=q["answer_letter"], greedy=ga[0], greedy_correct=(ga[0] == q["answer_letter"]),
                           greedy_len=len(gc[0]["token_ids"]), greedy_finish=gc[0]["finish"], greedy_text=gc[0]["text"],
                           sampled=sa, counts=dict(collections.Counter(sa)), n_fallback=sum(1 for c in sc if c.get("fallback")) + (1 if gc[0].get("fallback") else 0),
                           sampled_finish=[c["finish"] for c in sc], sampled_lens=[len(c["token_ids"]) for c in sc], sampled_texts=[c["text"] for c in sc])
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                LOG(f"row {q['row_id']:3d} greedy={ga[0]} ({'ok' if rec['greedy_correct'] else 'WRONG'}, {rec['greedy_len']} tok) o_0={rec['counts']} fallback={rec['n_fallback']}/{args.S0+1}")
            LOG(f"chunk done: rows {chunk[0]['row_id']}-{chunk[-1]['row_id']} in {time.time()-t1:.0f}s  [{min(ci+CH,len(todo))}/{len(todo)} {(time.time()-t_all)/60:.1f} min]")
    LOG("ALL DONE")
if __name__ == "__main__": main()
