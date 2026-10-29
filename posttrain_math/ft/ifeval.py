#!/usr/bin/env python3
"""IFEval on one model, thinking off and on. Runs ON THE EVAL POD.

Why it is here: the fine-tune trains instruction following. If IFEval does not move, the
fine-tune did not take, and any change (or no change) in maths means nothing. It is the check
that the treatment was actually delivered.

Data: google/IFEval (official, Apache-2.0), 541 prompts. Scoring: Google's own code,
google-research/instruction_following_eval (evaluation_lib.test_instruction_following_strict /
_loose), imported unmodified. Generation: vLLM, greedy, the model's own chat template, same
8,192-token budget as the maths runs. With thinking on, the scored response is the text after
</think>; a run that never closes </think> scores as not following (counted separately).
"""
import argparse, json, os, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from instruction_following_eval import evaluation_lib

URL = "https://huggingface.co/datasets/google/IFEval/resolve/main/ifeval_input_data.jsonl"
def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=8192)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    name = os.path.basename(a.model.rstrip("/"))
    path_in = os.path.join(a.out, "ifeval_input_data.jsonl")
    if not os.path.exists(path_in): urllib.request.urlretrieve(URL, path_in)
    inputs = evaluation_lib.read_prompt_list(path_in)
    log(f"IFEval prompts: {len(inputs)}")
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, dtype="bfloat16", seed=0, gpu_memory_utilization=0.90, max_model_len=24576)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, seed=0, skip_special_tokens=True)
    summary = {}
    for think in (False, True):
        cond = "on" if think else "off"
        prompts = [tok.apply_chat_template([{"role": "user", "content": x.prompt}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=think) for x in inputs]
        t0 = time.time()
        outs = llm.generate(prompts, sp)
        resp, unclosed, ntok = {}, 0, 0
        raw = open(os.path.join(a.out, f"ifeval_raw_{name}_{cond}.jsonl"), "w")
        for x, o in zip(inputs, outs):
            text = o.outputs[0].text; ntok += len(o.outputs[0].token_ids)
            if "</think>" in text: r = text.split("</think>", 1)[1].strip()
            elif think: r = ""; unclosed += 1
            else: r = text.strip()
            resp[x.prompt] = r
            raw.write(json.dumps({"key": x.key, "prompt": x.prompt, "raw": text, "response": r,
                                  "finish": o.outputs[0].finish_reason}) + "\n")
        raw.close()
        res = {}
        for kind, fn in (("strict", evaluation_lib.test_instruction_following_strict),
                         ("loose", evaluation_lib.test_instruction_following_loose)):
            outs_e = [fn(x, resp) for x in inputs]
            res[f"prompt_{kind}"] = sum(o.follow_all_instructions for o in outs_e) / len(outs_e)
            flat = [b for o in outs_e for b in o.follow_instruction_list]
            res[f"instruction_{kind}"] = sum(flat) / len(flat)
            with open(os.path.join(a.out, f"ifeval_scored_{name}_{cond}_{kind}.jsonl"), "w") as f:
                for x, o in zip(inputs, outs_e):
                    f.write(json.dumps({"key": x.key, "follow_all": o.follow_all_instructions,
                                        "follow_list": o.follow_instruction_list}) + "\n")
        res.update(unclosed_think=unclosed, mean_tokens=ntok / len(inputs), minutes=(time.time() - t0) / 60)
        summary[cond] = res
        log(f"IFEval {name} thinking {cond}: {json.dumps(res)}")
    json.dump(summary, open(os.path.join(a.out, f"ifeval_summary_{name}.json"), "w"), indent=1)
    log("IFEVAL DONE")

if __name__ == "__main__":
    main()
