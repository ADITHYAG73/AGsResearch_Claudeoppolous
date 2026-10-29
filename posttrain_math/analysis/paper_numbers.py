#!/usr/bin/env python3
"""Every number and figure series in the BASE-01 paper draft, from the scored/raw files on disk.
Run from posttrain_math/:  python3 analysis/paper_numbers.py  -> analysis/paper_numbers.json
Reads only; never calls a model. Uses score.py's own matching rule."""
import json, math, re, collections, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from score import matches, NUM, parse_final

D = 'runs/2026-10-06_base/'
Q = {q['id']: q for q in map(json.loads, open('data/questions_v0.jsonl'))}
M, C = ('4B', '9B'), ('notool_off', 'notool_on', 'tool_off', 'tool_on')
NA = ('truncated', 'turn_limit', 'malformed')
load = lambda kind, m, c, tag='': {r['id']: r for r in map(json.loads, open(f'{D}{kind}_Qwen3.5-{m}_{c}{tag}.jsonl'))}
S = {(m, c): load('scored', m, c) for m in M for c in C}
R = {(m, c): load('raw', m, c) for m in M for c in C}
N = len(Q); out = {}

def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z*z/n; ctr = (p + z*z/(2*n)) / den
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return round(100*(ctr-h), 1), round(100*(ctr+h), 1)
ok = lambda r: r['status'] == 'correct'
truth = lambda q: 0.0 if q.get('is_zero') else float(q['truth'])

# headline: outcome counts + CI + within 1e-3
out['headline'] = []
for m in M:
    for c in C:
        v = S[(m, c)].values(); k = sum(map(ok, v)); na = sum(r['status'] in NA for r in v)
        out['headline'].append(dict(model=m, cond=c, correct=k, wrong=N-k-na, noanswer=na,
                                    pct=round(100*k/N, 1), ci=wilson(k, N), close=round(100*sum(r['close_1e-3'] for r in v)/N, 1)))
# re-ask merged
out['reask'] = {}
for m in M:
    re_ = load('scored', m, 'tool_on', '_turns12'); merged = dict(S[(m, 'tool_on')]); merged.update(re_)
    out['reask'][m] = dict(n=len(re_), correct=sum(map(ok, re_.values())), merged_pct=round(100*sum(map(ok, merged.values()))/N, 1))
# paired transitions (b better, a better) and McNemar exact-ish z
def pair(a, b):
    x = sum(ok(a[i]) and not ok(b[i]) for i in Q); y = sum(not ok(a[i]) and ok(b[i]) for i in Q)
    z = (abs(x-y)-1)/math.sqrt(x+y) if x+y else 0
    return dict(a_only=x, b_only=y, z=round(z, 1))
out['pairs'] = {f'{m} {a}->{b}': pair(S[(m, a)], S[(m, b)]) for m in M for a, b in
                [('notool_off', 'tool_off'), ('notool_off', 'notool_on'), ('tool_off', 'tool_on'), ('notool_on', 'tool_on')]}
out['pairs'].update({f'4B vs 9B {c}': pair(S[('4B', c)], S[('9B', c)]) for c in C})
# what happens when thinking breaks a question the model had right
out['thinking_breaks'] = {m: dict(collections.Counter(S[(m, 'notool_on')][i]['status'] for i in Q
                                  if ok(S[(m, 'notool_off')][i]) and not ok(S[(m, 'notool_on')][i]))) for m in M}
# thinking alone, questions both models finished
a, b = S[('4B', 'notool_on')], S[('9B', 'notool_on')]
both = [i for i in Q if a[i]['status'] not in NA and b[i]['status'] not in NA]
out['both_finished'] = dict(n=len(both), **{m: round(100*sum(ok(S[(m, 'notool_on')][i]) for i in both)/len(both), 1) for m in M})
# family table
out['family'] = {}
for f in ['trig_int', 'trig_var', 'transcendental', 'composite', 'nested', 'mixed', 'undefined']:
    ids = [i for i in Q if Q[i]['family'] == f]
    out['family'][f] = dict(n=len(ids), **{f'{m} {c}': round(100*sum(ok(S[(m, c)][i]) for i in ids)/len(ids), 1) for m in M for c in C})
# trig dissociation: integer / negative / half / two-decimal / over-360, strict and within 1e-3
out['trig_sub'] = []
for m in M:
    for c in ('notool_off', 'notool_on', 'tool_off'):
        g = collections.defaultdict(list)
        for i, r in S[(m, c)].items():
            q = Q[i]
            if q['family'] == 'trig_var': g[q['sub'].split('_')[0]].append(r)
            elif q['family'] == 'trig_int': g['int'].append(r)
        # matched baseline: the variants are built from d in 1..89 (neg = -d, over360 = d+360, half/dec2 = d+frac),
        # so the fair integer comparison is the same first-quadrant angles, not all of 0..359
        g['int_1_89'] = [r for i, r in S[(m, c)].items() if Q[i]['family'] == 'trig_int'
                         and 1 <= int(re.search(r'\((\d+)°', Q[i]['expr']).group(1)) <= 89]
        for k, v in g.items():
            out['trig_sub'].append(dict(model=m, cond=c, variant=k, n=len(v), strict=round(100*sum(map(ok, v))/len(v), 1),
                                        close=round(100*sum(r['close_1e-3'] for r in v)/len(v), 1)))
# tabulation gradient on integer degrees
def angle(e): return int(re.search(r'\((-?\d+)°\)', e).group(1))
out['tabulation'] = []
for m in M:
    for c in ('notool_off', 'notool_on'):
        g = collections.defaultdict(list)
        for i, r in S[(m, c)].items():
            if Q[i]['family'] != 'trig_int': continue
            x = angle(Q[i]['expr']); k = 'multiple of 15' if x % 15 == 0 else 'multiple of 5' if x % 5 == 0 else 'other even' if x % 2 == 0 else 'odd'
            g[k].append(r)
        for k in ('multiple of 15', 'multiple of 5', 'other even', 'odd'):
            v = g[k]; out['tabulation'].append(dict(model=m, cond=c, group=k, n=len(v), pct=round(100*sum(map(ok, v))/len(v), 1),
                                                    noanswer=sum(r['status'] in NA for r in v)))
# truncated thinking: had the right value already been written?  (any number in the trace within half a unit)
def sig_digits(x):
    m = x.lstrip('+-').split('e')[0].split('E')[0].replace('.', '').lstrip('0')
    return len(m)
def has_value(text, t):
    # a match counts only if the written number carries >= 5 significant digits, so "1." of a list or a
    # bare "0.5" cannot count as the model having computed the value (found by reading q00270, sin(90°))
    for x in NUM.findall(text.replace(',', '')):
        try:
            if t != 0 and sig_digits(x) >= 5 and matches(float(x), t): return True
        except ValueError: pass
    return False
out['truncated_thinking'] = {}
for m in M:
    n = k = loop = 0
    for i, r in R[(m, 'notool_on')].items():
        if r['end'] != 'truncated' or Q[i]['undefined'] or Q[i].get('is_zero'): continue
        if sig_digits(repr(float(Q[i]['truth']))) < 5: continue   # simple values (1, 0.5, 2) cannot be tested this way
        n += 1; txt = ''.join(t['raw'] for t in r['turns'])
        k += has_value(txt, truth(Q[i]))
        tail = txt[-3000:]; ch = [tail[j:j+60] for j in range(0, len(tail)-60, 60)]
        loop += bool(ch) and collections.Counter(ch).most_common(1)[0][1] >= 3
    out['truncated_thinking'][m] = dict(n=n, right_value_written=k, pct=round(100*k/n, 1), repeating_tail=loop, repeating_pct=round(100*loop/n, 1))
# sandbox labels, and how far off the "right value, wrong answer" and "tool gave wrong value" answers are
out['labels'] = {f'{m} {c}': dict(collections.Counter(r['tool_label'] for r in S[(m, c)].values())) for m in M for c in ('tool_off', 'tool_on')}
out['calls'] = {f'{m} {c}': round(sum(r['n_calls'] for r in S[(m, c)].values())/N, 2) for m in M for c in ('tool_off', 'tool_on')}
out['copy_error_distance'] = {}
for m in M:
    for c in ('tool_off', 'tool_on'):
        cnt = collections.Counter()
        for i, s in S[(m, c)].items():
            if s['tool_label'] not in ('right_value_wrong_answer', 'tool_gave_wrong_value'): continue
            if Q[i]['undefined']: cnt[(s['tool_label'], 'undefined question')] += 1; continue
            kind, v = parse_final(R[(m, c)][i]['final']); t = truth(Q[i])
            if v is None: cnt[(s['tool_label'], 'no number')] += 1; continue
            rel = abs(v-t)/abs(t) if t else abs(v)
            cnt[(s['tool_label'], 'within 1e-5' if rel < 1e-5 else 'within 1e-3' if rel < 1e-3 else 'further')] += 1
        out['copy_error_distance'][f'{m} {c}'] = {f'{a} | {b}': v for (a, b), v in cnt.items()}
# code printed with a fixed number of decimals (":.6f" style) among tool_gave_wrong_value
out['fixed_decimals_in_code'] = {}
for m in M:
    for c in ('tool_off', 'tool_on'):
        ids = [i for i, s in S[(m, c)].items() if s['tool_label'] == 'tool_gave_wrong_value']
        fx = sum(bool(re.search(r':\.\d+f|round\(|%\.\d+f', '\n'.join(x['code'] for x in R[(m, c)][i]['calls']))) for i in ids)
        out['fixed_decimals_in_code'][f'{m} {c}'] = dict(n=len(ids), fixed_decimals=fx)
# budget curve: share correct if the budget had been B tokens (exact for one-turn greedy runs)
grid = [256, 512, 1024, 1536, 2048, 3072, 4096, 5120, 6144, 7168, 8192]
out['budget'] = []
for m in M:
    for c in ('notool_on', 'tool_on'):
        rows = list(S[(m, c)].values())
        for B in grid:
            out['budget'].append(dict(model=m, cond=c, budget=B, pct=round(100*sum(ok(r) and r['gen_tokens'] <= B for r in rows)/N, 1)))
# tokens
tok = {f'{m} {c}': sum(r['gen_tokens'] for r in R[(m, c)].values()) for m in M for c in C}
out['tokens'] = dict(total=sum(tok.values()), alone_thinking_share=round(sum(v for k, v in tok.items() if 'notool_on' in k)/sum(tok.values()), 3),
                     median_tokens={k: sorted(r['gen_tokens'] for r in R[tuple(k.split())].values())[N//2] for k in tok})
json.dump(out, open('analysis/paper_numbers.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
