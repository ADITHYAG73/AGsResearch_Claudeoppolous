#!/usr/bin/env python3
"""Validate publication constraints and, when present, every exported capture."""
import gzip
import json
from pathlib import Path
import re
from build_data import unpacked
import numpy as np

SITE = Path(__file__).resolve().parent/'site'


def main():
    html = (SITE/'index.html').read_text()
    assert html.startswith('<title>Latent Cube Explorer</title>')
    assert '<meta charset="utf-8">' in html[:1024]
    assert not re.search(r'<!doctype|<(?:html|head|body)(?:>|\s)', html, re.I)
    scripts = re.findall(r'<script[^>]+src="([^"]+)"', html, re.I)
    assert scripts == ['https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.11/katex.min.js']
    assert not re.search(r'<link', html, re.I)
    assert not re.search(r'\b(?:alert|confirm|prompt)\s*\(', html)
    files = [p for p in SITE.rglob('*') if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    largest = max(p.stat().st_size for p in files)
    assert total <= 60_000_000 and largest <= 15_000_000
    print(f'PASS static fragment/resource/size checks: site={total} bytes; largest={largest} bytes')
    path = SITE/'manifest.json'
    if not path.exists():
        print('NOT VERIFIED: real-data export absent; logit gate, numerical values, UI data interactions and complete export cannot be checked')
        return 2
    manifest = json.loads(path.read_text())
    assert manifest['site_bytes'] == total
    check = manifest['check']
    assert check['passed']
    assert check['max_abs_difference'] <= .5 and check['rmse'] <= .05
    assert check['max_abs_difference']/max(check['max_abs_saved_logit'],1) <= .01
    assert len(manifest['rows']) == 338 and manifest['unique_prompts'] == 201
    assert manifest['s3_reserved_bytes'] <= 15_000_000_000
    report = {'captures':len(manifest['rows']), 'unique_prompts':manifest['unique_prompts'],
              'site_bytes':total, 'largest_file_bytes':largest, 'files':len(files),
              'logit_check':manifest['check'], 's3_received_bytes':manifest['s3_received_bytes'],
              'full_token_positions':0, 'norm_token_positions':0, 'lens_token_positions':0,
              'attention_query_positions':0, 'norm_subsampled_captures':0,
              'max_attention_row_sum_error':0., 'max_attention_allowed_rounding_bound':0., 'max_future_attention_weight':0.,
              'max_cosine_diagonal_error':0., 'max_cosine_symmetry_error':0.,
              'max_pca_coordinate_mean_abs':0.}
    expected_neurons = np.empty((338,3,32,64), dtype=np.float64)
    report['max_neuron_cross_export_difference'] = 0.
    for row in manifest['rows']:
        d = unpacked(SITE/row['path'])
        m = d['meta']; length = len(m['tokens'])
        expected_neurons[row['index']] = d['neuron_readouts']
        assert len(m['token_ids']) == length
        assert m['id'] == row['id'] and m['variant'] == row['variant']
        assert all(0 <= p < length for p in m['positions'].values())
        assert all(k in m for k in ['free_reply','captured_reply','answer','prompt','chat_text'])
        required = sorted(set(range(min(length,128))) | set(m['positions'].values()))
        assert d['norm_positions'] == required
        report['full_token_positions'] += length
        report['norm_token_positions'] += len(required)
        report['lens_token_positions'] += len(d['lens_positions'])
        report['attention_query_positions'] += len(d['attention_queries'])
        report['norm_subsampled_captures'] += int(len(required)<length)
        for values in d['norms'].values():
            assert len(values) == 32 and all(len(r) == len(d['norm_positions']) for r in values)
        for layer in d['lens']:
            assert [c['position'] for c in layer] == d['lens_positions']
            for cell in layer:
                assert len(cell['top']) == 5 and len(cell['ranks']) == 7
                assert all(0 <= t[2] <= 1 for t in cell['top'])
                assert all(cell['top'][i][2] >= cell['top'][i+1][2] for i in range(4))
                assert all(1 <= rank <= 248320 for rank in cell['ranks'])
        for layer in d['mlp']:
            assert [c['position'] for c in layer] == d['mlp_positions']
            assert all(len(c['top']) == 20 for c in layer)
            assert all(all(abs(c['top'][i][1]) >= abs(c['top'][i+1][1]) for i in range(19)) for c in layer)
        assert len(d['attention']) == 8
        for layer in d['attention']:
            assert len(layer) == 16
            for head in layer:
                assert len(head) == len(d['attention_queries'])
                assert all(len(r) == length for r in head)
                for query, weights in zip(d['attention_queries'], head):
                    row_error = abs(sum(weights)-1)
                    # Official eager_attention_forward casts fp32 softmax to query.dtype (bf16).
                    # Unit roundoff is 2^-8; add half a decimal unit per displayed weight.
                    bound = (2**-8)/(1-2**-8) + length*0.5e-6
                    report['max_attention_allowed_rounding_bound'] = max(report['max_attention_allowed_rounding_bound'], bound)
                    assert row_error <= bound, (m['id'], query, row_error, bound)
                    report['max_attention_row_sum_error'] = max(report['max_attention_row_sum_error'], row_error)
                    report['max_future_attention_weight'] = max(report['max_future_attention_weight'], max(weights[query+1:], default=0))
        assert report['max_future_attention_weight'] == 0
    for readout_index,position in enumerate(['instr_end','prompt_end','reply_end']):
        for layer in range(32):
            d = unpacked(SITE/'across'/f'{position}-{layer}.bin')
            assert len(d['pca']) == 338 and all(len(x) == 2 for x in d['pca'])
            assert len(d['cosine']) == 338 and all(len(x) == 338 for x in d['cosine'])
            assert all(abs(d['cosine'][i][i]-1) <= .00002 for i in range(338))
            assert len(d['neurons']) == 338 and all(len(x) == 64 for x in d['neurons'])
            neuron_error = float(np.max(np.abs(np.asarray(d['neurons'])-expected_neurons[:,readout_index,layer])))
            report['max_neuron_cross_export_difference'] = max(report['max_neuron_cross_export_difference'], neuron_error)
            assert neuron_error < 1e-12
            cosine = np.asarray(d['cosine'])
            report['max_cosine_diagonal_error'] = max(report['max_cosine_diagonal_error'], float(np.max(np.abs(np.diag(cosine)-1))))
            report['max_cosine_symmetry_error'] = max(report['max_cosine_symmetry_error'], float(np.max(np.abs(cosine-cosine.T))))
            report['max_pca_coordinate_mean_abs'] = max(report['max_pca_coordinate_mean_abs'], float(np.max(np.abs(np.mean(d['pca'],axis=0)))))
            assert np.max(np.abs(cosine)) <= 1.00002
            assert report['max_cosine_symmetry_error'] <= .00002
            assert all(0 <= x <= 1 for x in d['explained_variance_ratio'])
    reference_path = SITE.parent/'cosine_reference.json'
    if reference_path.exists():
        reference = json.loads(reference_path.read_text())
        error = 0.
        for p,position in enumerate(reference['positions']):
            for layer in range(32):
                data = unpacked(SITE/'across'/f'{position}-{layer}.bin')
                error = max(error, abs(data['cosine'][0][1]-reference['values'][p][layer]))
        report['cosine_float64_reference_comparisons'] = 96
        report['max_cosine_float64_reference_difference'] = error
        assert error <= 0.000006
    report['status'] = 'passed'
    (SITE.parent/'validation_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print('PASS complete exported-data shape, coverage, probability/rank bounds, cosine diagonal checks')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
