#!/usr/bin/env python3
"""Collect raw answers for one model under one condition. Scoring is a separate step (score.py).

Four conditions, same questions, same user prompt in every one:
  notool_off   model alone, thinking disabled      (one pass, no scratchpad)
  notool_on    model alone, thinking enabled       (scratchpad)
  tool_off     compute sandbox available, thinking disabled
  tool_on      compute sandbox available, thinking enabled

THE HARNESS IS THE MODEL'S OWN CHAT TEMPLATE. Prompts are rendered with the tokenizer's official
template (`apply_chat_template(..., tools=..., enable_thinking=...)`), never hand-built, so the
tool instructions the model reads are the ones Qwen ships. The only thing this file adds is the
loop: generate -> if the model emitted a tool call, run the code and hand back the output ->
generate again, until it answers or a limit is reached.

THE SANDBOX DOES NO MATHS. It runs whatever code the model wrote in a fresh interpreter and
returns what was printed. Every choice is the model's: whether to call it, what to compute,
degrees or radians, how to round.

Rule-7 properties, by construction: one progress line per chunk with rate and ETA; one JSON line
per finished question, flushed per chunk; a restart skips ids already on disk; nothing is held
back to the end.
"""
import argparse, json, os, re, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor

USER_PROMPT = ("Compute {expr}. Angles written with a degree sign (°) are in degrees; any other "
               "argument of a trigonometric function is in radians.{unit} Respond with only the "
               "numeric value, to 6 significant figures. If the value is undefined over the real "
               "numbers, respond with exactly: undefined")

TOOLS = [{"type": "function", "function": {
    "name": "run_python",
    "description": ("Execute Python 3 code in a fresh interpreter and return what it prints to "
                    "standard output. Each call starts with no memory of earlier calls. The "
                    "standard library, numpy, sympy and mpmath are available. Use print() to see "
                    "a value."),
    "parameters": {"type": "object",
                   "properties": {"code": {"type": "string", "description": "Python source code to run."}},
                   "required": ["code"]}}}]

CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM_RE = re.compile(r"<parameter=code>\n?(.*?)\n?</parameter>", re.S)

def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)

# ------------------------------------------------------------------ sandbox
def _demote():
    # the pod runs as root; model-written code must not be able to touch the results
    try:
        os.setgid(65534); os.setuid(65534)
    except Exception:
        pass

def run_code(code, timeout=10, cap=4000):
    d = tempfile.mkdtemp(prefix="sbx_")
    os.chmod(d, 0o777)
    try:
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           timeout=timeout, cwd=d, env={"PATH": os.environ.get("PATH", ""), "HOME": d},
                           preexec_fn=_demote if os.geteuid() == 0 else None)
        out, err, rc, to = p.stdout, p.stderr, p.returncode, False
    except subprocess.TimeoutExpired:
        out, err, rc, to = "", f"timed out after {timeout} s", -1, True
    shown = out[:cap] if out.strip() else ""
    if rc != 0:
        shown += ("\n" if shown else "") + "ERROR:\n" + err[-1500:]
    elif not shown:
        shown = "(the code ran and printed nothing)"
    return {"stdout": out[:cap], "stderr": err[-1500:], "rc": rc, "timed_out": to, "shown": shown}

# ------------------------------------------------------------------ backends
class VLLMBackend:
    def __init__(self, a):
        from vllm import LLM
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(a.model)
        self.llm = LLM(model=a.model, dtype="bfloat16", seed=0, gpu_memory_utilization=a.gpu_mem,
                       max_model_len=a.max_model_len, enable_prefix_caching=True)
        self.eos = [self.tok.convert_tokens_to_ids("<|im_end|>")]
    def generate(self, prompts, max_tokens):
        from vllm import SamplingParams
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, stop_token_ids=self.eos,
                            skip_special_tokens=False, seed=0)
        outs = self.llm.generate(prompts, sp, use_tqdm=False)
        return [(o.outputs[0].text, len(o.outputs[0].token_ids), o.outputs[0].finish_reason) for o in outs]

class HFBackend:
    """Laptop dry-run only: same prompts, same parsing, one question at a time on CPU."""
    def __init__(self, a):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(a.model)
        self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
        self.m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval().to(self.dev)
        self.eos = self.tok.convert_tokens_to_ids("<|im_end|>")
        log(f"hf backend on {self.dev}")
    def generate(self, prompts, max_tokens):
        res = []
        for p in prompts:
            enc = self.tok(p, return_tensors="pt").to(self.dev)
            with self.torch.no_grad():
                out = self.m.generate(**enc, max_new_tokens=max_tokens, do_sample=False,
                                      eos_token_id=self.eos, pad_token_id=self.eos)
            new = out[0, enc["input_ids"].shape[1]:]
            hit = len(new) > 0 and int(new[-1]) == self.eos
            text = self.tok.decode(new[:-1] if hit else new, skip_special_tokens=False)
            res.append((text, len(new), "stop" if hit else "length"))
        return res

# ------------------------------------------------------------------ the loop
def split_think(text, thinking):
    if "</think>" in text:
        t, rest = text.split("</think>", 1)
        return t.replace("<think>", "").strip(), rest.strip()
    return (text.strip(), "") if thinking else ("", text.strip())   # never closed its thinking

def run_condition(a, be, cond, qs_all, name):
    use_tool, thinking = cond.startswith("tool"), cond.endswith("_on")
    path = os.path.join(a.out, f"raw_{name}_{cond}{('_' + a.tag) if a.tag else ''}.jsonl")
    done = set()
    if os.path.exists(path):
        for l in open(path):
            try: done.add(json.loads(l)["id"])
            except Exception: pass
        log(f"resuming {cond}: {len(done)} already on disk")
    qs = [q for q in qs_all if q["id"] not in done]
    log(f"=== {cond}: {len(qs)} questions | model {name} | tool={use_tool} thinking={thinking} "
        f"| max_tokens/turn {a.max_tokens} | max_turns {a.max_turns}")
    json.dump({"user_prompt": USER_PROMPT, "tools": TOOLS if use_tool else None, "cond": cond,
               "model": a.model, "max_tokens": a.max_tokens, "max_turns": a.max_turns,
               "temperature": 0.0, "backend": a.backend},
              open(path.replace(".jsonl", ".meta.json"), "w"), indent=1, ensure_ascii=False)
    if not qs:
        log("nothing to do"); return
    tok = be.tok
    def render(msgs):
        return tok.apply_chat_template(msgs, tools=TOOLS if use_tool else None,
                                       add_generation_prompt=True, tokenize=False,
                                       enable_thinking=thinking)
    probe = render([{"role": "user", "content": "x"}])
    log(f"prompt template check: tools block {'present' if '<tools>' in probe else 'ABSENT'} | "
        f"ends with {probe[-24:]!r}")
    assert ("<tools>" in probe) == use_tool, "tool block does not match the condition"

    fh = open(path, "a")
    t0 = time.time(); n_done = 0; tok_total = 0
    pool = ThreadPoolExecutor(16)
    for c0 in range(0, len(qs), a.chunk):
        chunk = qs[c0:c0 + a.chunk]
        st = [{"q": q, "msgs": [{"role": "user", "content": USER_PROMPT.format(
                    expr=q["expr"], unit=(" Give the angle in degrees." if q.get("answer_unit") == "degrees" else ""))}],
               "turns": [], "calls": [], "end": None} for q in chunk]
        for turn in range(a.max_turns):
            live = [s for s in st if s["end"] is None]
            if not live: break
            outs = be.generate([render(s["msgs"]) for s in live], a.max_tokens)
            todo = []
            for s, (text, ntok, fin) in zip(live, outs):
                tok_total += ntok
                think, body = split_think(text, thinking)
                s["turns"].append({"raw": text, "n_tokens": ntok, "finish": fin})
                if fin == "length":
                    s["end"] = "truncated"; continue
                calls = CALL_RE.findall(body) if use_tool else []
                if use_tool and "<tool_call>" in body and not calls:
                    s["end"] = "malformed_call"; continue
                if not calls:
                    s["end"] = "answered"; s["final"] = body; continue
                codes = []
                for fname, inner in calls[:3]:
                    m = PARAM_RE.search(inner)
                    codes.append((fname.strip(), m.group(1) if m else None))
                pre = CALL_RE.sub("", body).strip()
                s["msgs"].append({"role": "assistant", "content": pre, "reasoning_content": think,
                                  "tool_calls": [{"type": "function", "function": {"name": fn, "arguments": {"code": cd or ""}}}
                                                 for fn, cd in codes]})
                todo.append((s, codes))
            jobs = [(s, fn, cd) for s, codes in todo for fn, cd in codes]
            res = list(pool.map(lambda j: run_code(j[2]) if (j[1] == "run_python" and j[2] is not None)
                                else {"stdout": "", "stderr": "", "rc": -2, "timed_out": False,
                                      "shown": f"ERROR: unknown tool or missing code parameter ({j[1]})"}, jobs))
            for (s, fn, cd), r in zip(jobs, res):
                s["calls"].append({"turn": turn, "tool": fn, "code": cd, "stdout": r["stdout"],
                                   "stderr": r["stderr"], "rc": r["rc"], "timed_out": r["timed_out"]})
                s["msgs"].append({"role": "tool", "content": r["shown"]})
        for s in st:
            if s["end"] is None: s["end"] = "turn_limit"
            q = s["q"]
            fh.write(json.dumps({"id": q["id"], "expr": q["expr"], "cond": cond, "model": name,
                                 "end": s["end"], "final": s.get("final", ""), "n_turns": len(s["turns"]),
                                 "n_calls": len(s["calls"]), "gen_tokens": sum(t["n_tokens"] for t in s["turns"]),
                                 "turns": s["turns"], "calls": s["calls"]}, ensure_ascii=False) + "\n")
        fh.flush(); n_done += len(chunk)
        el = time.time() - t0; rate = n_done / max(el, 1e-9)
        ends = {}
        for s in st: ends[s["end"]] = ends.get(s["end"], 0) + 1
        log(f"  {cond} {n_done}/{len(qs)} done · {rate:.2f} q/s · {tok_total/max(el,1e-9):.0f} tok/s · "
            f"eta {(len(qs)-n_done)/max(rate,1e-9)/60:.1f} min · this chunk {ends}")
    fh.close()
    log(f"DONE {name} {cond}: {n_done} questions, {tok_total} generated tokens, {(time.time()-t0)/60:.1f} min")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cond", required=True, help="comma-separated: notool_off,tool_off,notool_on,tool_on")
    ap.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--max-tokens", type=int, default=8192, help="per model turn")
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1, help="take every Nth question (an even spread for smoke tests)")
    ap.add_argument("--core", action="store_true")
    ap.add_argument("--core-first", action="store_true", help="run the core subset before the rest")
    ap.add_argument("--ids", default="")
    ap.add_argument("--max-model-len", type=int, default=24576)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    conds = [c.strip() for c in a.cond.split(",") if c.strip()]
    for c in conds: assert c in ("notool_off", "notool_on", "tool_off", "tool_on"), c

    qs = [json.loads(l) for l in open(a.questions)]
    if a.core: qs = [q for q in qs if q["core"]]
    if a.ids:  qs = [q for q in qs if q["id"] in set(a.ids.split(","))]
    if a.stride > 1: qs = qs[::a.stride]
    if a.limit: qs = qs[:a.limit]
    if a.core_first: qs = [q for q in qs if q["core"]] + [q for q in qs if not q["core"]]
    os.makedirs(a.out, exist_ok=True)
    name = os.path.basename(a.model.rstrip("/"))
    log(f"{len(qs)} questions x {len(conds)} conditions on {name}")
    be = VLLMBackend(a) if a.backend == "vllm" else HFBackend(a)
    for cond in conds:
        run_condition(a, be, cond, qs, name)
    log("ALL CONDITIONS DONE")

if __name__ == "__main__":
    main()
