"""Forking Paths sampling on vLLM. OUR code; mirrors the paper's semantics (Bigelow 2024 /
Forking Fast 2026, forking_paths/{resample,answers}.py) but is not their code:

  base path : greedy (temperature 0), top-10 logprobs recorded at every generated position
  branches  : at each position t, every token with p >= p_thresh (0.05) incl. the greedy one;
              `--only-forks` keeps only positions with >= 2 such tokens (plus t=0)
  resample  : prefix = prompt + base[:t] + [w]; n=S continuations, T=1.0, top_p=1, top_k=all,
              max cont_max_tokens, stop at <|im_end|>/<|endoftext|>
  outcome   : their regex `parse_mmlu_answer` on the continuation text (imported from the
              forking-fast repo, unchanged); regex misses -> 1-token logit read after the
              "Therefore, among A through D, the answer is (" suffix, argmax over A-D among
              the top-20 logprobs; nothing there -> "Other"

Engineering that their code lacks, per the pre-launch rule:
  * one timestamped progress line per chunk of branches, with tokens/s
  * per-chunk checkpoint: outcomes AND continuation text appended to <out>/branches.jsonl
    as they complete; a rerun with --resume skips finished branches
  * cost line printed before generation starts (tokens to generate at the configured S)
  * everything saved raw: base path ids/text/top-10, per-branch token ids + text + answers
"""
from __future__ import annotations
import os, sys, json, time, argparse, collections
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _c in (os.environ.get("FORKING_FAST_ROOT", ""), os.path.join(ROOT, "reference_repos", "forking-fast"), "/root/forking-fast"):
    if _c and os.path.isdir(os.path.join(_c, "forking_paths", "forking_paths")):
        sys.path.insert(0, os.path.join(_c, "forking_paths")); break
from forking_paths.answers import parse_mmlu_answer, VALID, _COMPILED
from forking_paths.prompts import format_mmlu_instruct_user, THEREFORE_ABCD

def main():
    global args, llm, tok, eos_ids, therefore_ids, letter_ids, LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--S", type=int, default=200)
    ap.add_argument("--S0", type=int, default=200, help="draws at t=0")
    ap.add_argument("--p-thresh", type=float, default=0.05)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--base-max-tokens", type=int, default=400)
    ap.add_argument("--cont-max-tokens", type=int, default=400)
    ap.add_argument("--stride", type=int, default=1, help="consider positions t %% stride == 0 only")
    ap.add_argument("--only-forks", action="store_true")
    ap.add_argument("--max-branches", type=int, default=None, help="smoke: cap number of branches")
    ap.add_argument("--chunk", type=int, default=8, help="branches per generate() call / checkpoint")
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thinking", action="store_true", help="leave Qwen thinking ON (default off)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    LOG = lambda *a: print(f"[fpa {time.strftime('%H:%M:%S')}]", *a, flush=True)

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    llm = LLM(model=args.model, dtype="bfloat16", seed=args.seed, gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_model_len, enable_prefix_caching=True, trust_remote_code=False)
    tok = llm.get_tokenizer()
    eos_ids = sorted({i for i in (tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")) if i is not None and i >= 0})
    LOG("model", args.model, "eos_ids", eos_ids, [tok.decode([i]) for i in eos_ids])
    therefore_ids = tok(THEREFORE_ABCD, add_special_tokens=False)["input_ids"]
    letter_ids = {L: tok(L, add_special_tokens=False)["input_ids"] for L in VALID}
    letter_ids = {L: ids[0] for L, ids in letter_ids.items() if len(ids) == 1}
    LOG("letter token ids", letter_ids)

    def build_prompt_ids(q):
        user = format_mmlu_instruct_user(q["question"], q["choices"])
        ids = tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True,
                                      tokenize=True, enable_thinking=bool(args.thinking))
        if hasattr(ids, "input_ids"): ids = ids["input_ids"]
        ids = list(ids); return ids[0] if ids and isinstance(ids[0], list) else ids

    def base_path(prompt_ids):
        sp = SamplingParams(temperature=0.0, max_tokens=args.base_max_tokens, logprobs=args.top_k,
                            stop_token_ids=eos_ids, seed=args.seed)
        out = llm.generate([TokensPrompt(prompt_token_ids=prompt_ids)], sp, use_tqdm=False)[0].outputs[0]
        gen = list(out.token_ids)
        if gen and gen[-1] in eos_ids: gen = gen[:-1]
        topk = []
        for pos, d in enumerate(out.logprobs[:len(gen)]):
            items = sorted(((tid, lp.logprob) for tid, lp in d.items()), key=lambda x: -x[1])[:args.top_k]
            topk.append([(int(t), float(l)) for t, l in items])
        return gen, topk, out.finish_reason

    def enumerate_branches(prompt_ids, gen, topk):
        import math
        branches = []
        for t in range(len(gen)):
            if args.stride > 1 and t % args.stride != 0 and t != 0: continue
            cands = [(tid, math.exp(lp)) for tid, lp in topk[t] if math.exp(lp) >= args.p_thresh or tid == gen[t]]
            if gen[t] not in [c[0] for c in cands]: cands.append((gen[t], math.exp(dict(topk[t]).get(gen[t], -1e9))))
            if args.only_forks and len(cands) < 2 and t != 0: continue
            for tid, p in cands:
                branches.append(dict(t=t, tok_id=int(tid), tok_p=float(p), is_base=bool(tid == gen[t]),
                                     prefix=prompt_ids + gen[:t] + [int(tid)]))
        return branches

    def extract(prefix, conts):
        """Returns list of category strings for the continuations of one branch."""
        answers, misses = [], []
        for i, c in enumerate(conts):
            a = parse_mmlu_answer(c["text"])
            # Their regex's "option X" pattern fires on Qwen's running commentary ("Option C uses
            # Rxzy...") and takes the LAST mention as the verdict (smoke2: 97/97 regex hits were
            # this). Accept a regex answer only if an explicit "answer is/answer:" pattern matched;
            # otherwise treat as a miss -> logit read. Recorded per continuation as `regex_kind`.
            explicit = any(cre.search(c["text"]) for cre in _COMPILED[:4])
            c["regex_kind"] = ("explicit" if (a and explicit) else ("option_mention" if a else "none"))
            if a is not None and not explicit: a = None
            answers.append(a)
            if a is None: misses.append(i)
        if misses:
            sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20, seed=args.seed)
            prompts = [TokensPrompt(prompt_token_ids=prefix + conts[i]["token_ids"] + therefore_ids) for i in misses]
            outs = llm.generate(prompts, sp, use_tqdm=False)
            for i, o in zip(misses, outs):
                lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
                best, bestlp = None, -1e9
                for L, tid in letter_ids.items():
                    if tid in lp and lp[tid].logprob > bestlp: best, bestlp = L, lp[tid].logprob
                answers[i] = best
                conts[i]["fallback"] = True
        return [a if a in VALID else "Other" for a in answers]

    questions = json.load(open(args.questions))
    for q in questions:
        rid = int(q["row_id"]); out_base = os.path.join(args.out_dir, f"row{rid:03d}")
        ckpt = out_base + "_branches.jsonl"; done = set()
        if args.resume and os.path.exists(ckpt):
            for line in open(ckpt): r = json.loads(line); done.add((r["t"], r["tok_id"]))
            LOG(f"resume: {len(done)} branches already done")
        prompt_ids = build_prompt_ids(q)
        t0 = time.time(); gen, topk, fin = base_path(prompt_ids)
        base_text = tok.decode(gen)
        LOG(f"row {rid}: base path {len(gen)} tokens, finish={fin}, {time.time()-t0:.1f}s; think tag present: {'<think>' in base_text}")
        branches = enumerate_branches(prompt_ids, gen, topk)
        if args.max_branches: branches = branches[:args.max_branches]
        n_pos = len({b['t'] for b in branches})
        est_tokens = sum((args.S0 if b["t"] == 0 else args.S) for b in branches) * args.cont_max_tokens
        LOG(f"row {rid}: {len(branches)} branches at {n_pos} positions; S={args.S} S0={args.S0}; "
            f"UPPER BOUND {est_tokens/1e6:.1f}M generated tokens (cont_max {args.cont_max_tokens})")
        json.dump(dict(meta=dict(row_id=rid, question=q["question"], choices=q["choices"], answer_letter=q.get("answer_letter"),
                                 model=args.model, thinking=bool(args.thinking), args=vars(args)),
                       prompt_ids=prompt_ids, base=dict(gen_ids=gen, text=base_text, finish=fin, token_texts=[tok.decode([i]) for i in gen], topk=topk),
                       branches=[{k: v for k, v in b.items() if k != "prefix"} for b in branches]),
                  open(out_base + "_base.json", "w"), ensure_ascii=False)
        todo = [b for b in branches if (b["t"], b["tok_id"]) not in done]
        t_gen = time.time(); tokens_done = 0
        with open(ckpt, "a") as f:
            for ci in range(0, len(todo), args.chunk):
                chunk = todo[ci:ci + args.chunk]; t1 = time.time()
                sps = [SamplingParams(n=(args.S0 if b["t"] == 0 else args.S), temperature=1.0, top_p=1.0, top_k=-1,
                                      max_tokens=args.cont_max_tokens, stop_token_ids=eos_ids, seed=args.seed + b["t"] * 1000 + b["tok_id"] % 1000)
                       for b in chunk]
                outs = llm.generate([TokensPrompt(prompt_token_ids=b["prefix"]) for b in chunk], sps, use_tqdm=False)
                for b, o in zip(chunk, outs):
                    conts = [dict(token_ids=list(c.token_ids), text=c.text, finish=c.finish_reason) for c in o.outputs]
                    answers = extract(b["prefix"], conts)
                    rec = dict(t=b["t"], tok_id=b["tok_id"], tok_p=b["tok_p"], is_base=b["is_base"], tok_text=tok.decode([b["tok_id"]]),
                               answers=answers, counts=dict(collections.Counter(answers)), n_fallback=sum(1 for c in conts if c.get("fallback")),
                               cont_lens=[len(c["token_ids"]) for c in conts], conts=conts)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                    tokens_done += sum(rec["cont_lens"])
                dt = time.time() - t1
                LOG(f"row {rid}: branches {min(ci+args.chunk, len(todo))}/{len(todo)}  t={[b['t'] for b in chunk]}  "
                    f"{dt:.1f}s  {sum(len(c['token_ids']) for o in outs for c in [dict(token_ids=x.token_ids) for x in o.outputs])/dt:.0f} tok/s  "
                    f"elapsed {(time.time()-t_gen)/60:.1f} min")
        LOG(f"row {rid}: DONE. {len(todo)} branches, {tokens_done/1e6:.2f}M tokens, {(time.time()-t_gen)/60:.1f} min generation")
    LOG("ALL DONE")


if __name__ == "__main__":
    main()
