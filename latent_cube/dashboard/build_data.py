#!/usr/bin/env python3
"""Read-only, resumable LATENT-CUBE exporter. No model execution or S3 writes.

Temporary raw arrays live only in a TemporaryDirectory under /tmp. Published JSON
contains derived summaries, never credentials. S3 bytes are reserved before each
GET in a persistent ledger (including failed requests, conservatively).
"""
import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = '4'
# Do not consult ~/.aws, profiles, credential processes, or instance metadata.
os.environ['AWS_CONFIG_FILE'] = '/dev/null'
os.environ['AWS_SHARED_CREDENTIALS_FILE'] = '/dev/null'
os.environ['BOTO_CONFIG'] = '/dev/null'
os.environ['AWS_EC2_METADATA_DISABLED'] = 'true'
os.environ.pop('AWS_PROFILE', None)
os.environ.pop('AWS_DEFAULT_PROFILE', None)
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import struct
import tempfile
import threading
import time
import numpy as np
import requests
from threadpoolctl import threadpool_limits
import boto3
from botocore.config import Config
from botocore.exceptions import ConnectionClosedError, ReadTimeoutError, ConnectTimeoutError, EndpointConnectionError

ROOT = Path(__file__).resolve().parent
SITE = ROOT / 'site'
REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
HF = f'https://huggingface.co/Qwen/Qwen3.5-4B/resolve/{REVISION}/'
BUCKET = 'posttrain-math-459653581741-ap-south-1'
PREFIX = 'latent_cube/capture01/'
INTEREST = [41776, 18, 21, 61, 22997, 87, 17]
POSITIONS = ['instr_end', 'prompt_end', 'reply_end']


def log(message):
    print(datetime.now(timezone.utc).isoformat(timespec='seconds'), message, flush=True)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False))
    temporary.replace(path)


def filename(pid):
    return pid.replace('|', '__').replace('/', '_').replace('·', 'dot').replace('(', '').replace(')', '').replace('^', 'pow').replace('²', 'sq')


class Reader:
    def __init__(self, opening_balance, budget=15_000_000_000):
        self.ledger_path = ROOT / 'read_ledger.json'
        self.ledger = json.loads(self.ledger_path.read_text()) if self.ledger_path.exists() else {
            's3_reserved_bytes': 0, 's3_received_bytes': 0, 's3_requests': 0, 'hf_received_bytes': 0,
            'opening_balance': opening_balance, 'started_utc': datetime.now(timezone.utc).isoformat()}
        self.budget = budget
        self.stop = threading.Event()
        self.last_progress = time.monotonic()
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=16)
        self.client = None
        self.http = requests.Session()
        self.watchdog = threading.Thread(target=self.watch, daemon=True)
        self.watchdog.start()
        self.persist()
        log(f'WATCHDOG running; S3 budget={budget} bytes; opening balance={opening_balance}')

    def watch(self):
        while not self.stop.wait(30):
            log(f'WATCHDOG S3 reserved={self.ledger["s3_reserved_bytes"] / 1e9:.4f} GB; last progress {time.monotonic()-self.last_progress:.0f}s ago')
            if time.monotonic() - self.last_progress > 600:
                self.stop.set()
                log('WATCHDOG STOP: no progress for 600 seconds')

    def persist(self):
        with self.lock:
            save(self.ledger_path, self.ledger)

    def s3(self):
        if self.client is None:
            names = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN']
            if not all(os.environ.get(k) for k in names):
                raise RuntimeError('Missing supplied temporary environment credentials; no fallback permitted')
            self.client = boto3.client('s3', region_name='ap-south-1',
                aws_access_key_id=os.environ[names[0]], aws_secret_access_key=os.environ[names[1]],
                aws_session_token=os.environ[names[2]], config=Config(
                    retries={'total_max_attempts': 1, 'mode': 'standard'}, connect_timeout=15, read_timeout=60, max_pool_connections=16))
        return self.client

    def get(self, source, start=None, end=None):
        for attempt in range(3):
            try:
                return self._get_once(source, start, end)
            except (ConnectionClosedError, ReadTimeoutError, ConnectTimeoutError, EndpointConnectionError,
                    requests.ConnectionError, requests.Timeout) as exc:
                if attempt == 2 or self.stop.is_set():
                    raise
                log(f'Transient {type(exc).__name__}; retry {attempt+1}/2; each S3 retry reserves its bytes again')
                time.sleep(attempt+1)

    def _get_once(self, source, start=None, end=None):
        if self.stop.is_set():
            raise RuntimeError('Watchdog stopped reads')
        if source.startswith('https://'):
            headers = {} if start is None else {'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'}
            # Range-specific query avoids caches reusing a response for another range.
            url = source if start is None else source + f'?range={start}-{end}'
            with self.http.get(url, headers=headers, timeout=(20, 90), stream=True) as response:
                response.raise_for_status()
                if start is not None and response.status_code != 206:
                    raise RuntimeError('Server ignored Range; refusing whole-file download')
                data = response.content
                if start is not None and len(data) != end-start+1:
                    raise RuntimeError('Incorrect HTTP range length')
            self.ledger['hf_received_bytes'] += len(data)
        else:
            if start is None:
                raise RuntimeError('S3 reads require explicit byte bounds')
            size = end-start+1
            with self.lock:
                if self.ledger['s3_reserved_bytes'] + size > self.budget:
                    raise RuntimeError('S3 byte watchdog: next read exceeds 15 GB budget')
                self.ledger['s3_reserved_bytes'] += size
                self.ledger['s3_requests'] += 1
                self.persist()
            response = self.s3().get_object(Bucket=BUCKET, Key=source, Range=f'bytes={start}-{end}')
            with response['Body'] as body:
                if response['ResponseMetadata']['HTTPStatusCode'] != 206:
                    raise RuntimeError('S3 ignored range')
                data = body.read()
            if len(data) != size:
                raise RuntimeError('Incorrect S3 range length')
            with self.lock:
                self.ledger['s3_received_bytes'] += len(data)
        self.last_progress = time.monotonic()
        self.persist()
        return data

    def inventory(self):
        cache = ROOT / 'inventory.json'
        if cache.exists():
            return json.loads(cache.read_text())
        objects = []
        for page in self.s3().get_paginator('list_objects_v2').paginate(Bucket=BUCKET, Prefix=PREFIX):
            objects.extend({'key': item['Key'], 'size': item['Size']} for item in page.get('Contents', []))
            self.last_progress = time.monotonic()
        save(cache, objects)
        return objects

    def close(self):
        self.stop.set()
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.persist()
        self.http.close()
        if self.client:
            self.client.close()


class SafeTensor:
    """Safetensors contiguous byte ranges; never fetch a whole capture file."""
    def __init__(self, reader, source):
        self.reader, self.source = reader, source
        length = struct.unpack('<Q', reader.get(source, 0, 7))[0]
        if length > 20_000_000:
            raise RuntimeError('Unreasonable safetensors header length')
        self.header = json.loads(reader.get(source, 8, 7+length))
        self.base = 8+length

    def array(self, name, first=0, count=None):
        desc = self.header[name]
        shape = desc['shape']
        total = int(np.prod(shape))
        count = total-first if count is None else count
        if first < 0 or count <= 0 or first+count > total:
            raise ValueError('Tensor slice outside bounds')
        size = {'F16': 2, 'BF16': 2, 'F32': 4}[desc['dtype']]
        begin = self.base+desc['data_offsets'][0]+first*size
        raw = self.reader.get(self.source, begin, begin+count*size-1)
        if desc['dtype'] == 'BF16':
            result = (np.frombuffer(raw, '<u2').astype(np.uint32) << 16).view(np.float32)
        else:
            result = np.frombuffer(raw, '<f2' if size == 2 else '<f4').astype(np.float32)
        return result.reshape(shape) if first == 0 and count == total else result

    def rows(self, name, positions):
        """Read exact selected rows, merging consecutive positions; parallel I/O only."""
        shape = self.header[name]['shape']
        length, width = shape[-2:]
        leading = int(np.prod(shape[:-2]))
        positions = list(positions)
        runs = []
        for column, position in enumerate(positions):
            if runs and position == runs[-1][1]+runs[-1][2]:
                runs[-1][2] += 1
            else:
                runs.append([column, position, 1])
        out = np.empty((leading, len(positions), width), dtype=np.float32)
        requests_to_make = [(layer, column, position, count) for layer in range(leading) for column, position, count in runs]
        def read(task):
            layer, column, position, count = task
            values = self.array(name, (layer*length+position)*width, count*width).reshape(count, width)
            out[layer, column:column+count] = values
        # Client must exist before worker threads start; boto3 clients can be shared.
        self.reader.s3()
        list(self.reader.pool.map(read, requests_to_make))
        return out.reshape(*shape[:-2], len(positions), width)


class Lens:
    def __init__(self, reader, scratch):
        index = json.loads(reader.get(HF+'model.safetensors.index.json'))['weight_map']
        norm_name = 'model.language_model.norm.weight'
        embed_name = 'model.language_model.embed_tokens.weight'
        norm = SafeTensor(reader, HF+index[norm_name])
        self.weight = norm.array(norm_name)
        embed = SafeTensor(reader, HF+index[embed_name])
        shape = embed.header[embed_name]['shape']
        assert shape == [248320, 2560], shape
        self.embedding = np.memmap(Path(scratch)/'embedding.f32', mode='w+', dtype='float32', shape=tuple(shape))
        for start in range(0, shape[0], 4096):
            end = min(start+4096, shape[0])
            self.embedding[start:end] = embed.array(embed_name, start*2560, (end-start)*2560).reshape(-1, 2560)
            if start % 32768 == 0:
                log(f'HF embedding rows {end}/{shape[0]}')
        self.embedding.flush()

    def project(self, x):
        x = np.asarray(x, dtype=np.float32)
        h = x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True)+1e-6) * (1+self.weight)
        return h @ self.embedding.T

    def check(self, capture, meta):
        length = len(meta['tokens'])
        x = capture.array('R1', 31*length*2560, length*2560).reshape(length, 2560)
        maximum, squares, elements, magnitude = 0., 0., 0, 0.
        for start in range(0, length, 8):
            end = min(start+8, length)
            predicted = self.project(x[start:end])
            actual = capture.array('logits', start*248320, (end-start)*248320).reshape(end-start, 248320)
            diff = predicted-actual
            maximum = max(maximum, float(np.max(np.abs(diff))))
            magnitude = max(magnitude, float(np.max(np.abs(actual))))
            squares += float(np.sum(diff.astype(np.float64)**2))
            elements += diff.size
        result = {'prompt': meta['id'], 'variant': meta['variant'], 'tokens_checked': length,
                  'vocabulary_checked': 248320, 'max_abs_difference': maximum,
                  'rmse': (squares/elements)**.5, 'max_abs_saved_logit': magnitude,
                  'precision': 'float32 projection from stored F16 residuals and original BF16 weights'}
        # Conservative diagnostic only: never silently label a nonzero difference as exact.
        result['passed'] = maximum <= .5 and result['rmse'] <= .05 and maximum/max(magnitude, 1) <= .01
        result['tolerance'] = 'max <= 0.5 AND RMSE <= 0.05 AND max / maximum absolute saved logit <= 1%; bf16 capture differs from float32 projection'
        save(SITE/'logit_check.json', result)
        log('LOGIT CHECK '+json.dumps(result))
        if not result['passed']:
            raise RuntimeError('Mandatory logit check failed: STOP; see site/logit_check.json')
        return result


def packed(path, value):
    import gzip
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.name == 'across':
        metadata = {k:v for k,v in value.items() if k not in ['cosine','neurons']}
        buffers = []
        metadata['fixed_point_scale'] = 100000
        for key in ['cosine','neurons']:
            array = np.asarray(value[key], dtype=np.float64)
            integers = np.rint(array*100000)
            if np.max(np.abs(integers)) > np.iinfo(np.int32).max:
                raise RuntimeError('Fixed-point value exceeds int32 capacity')
            if not np.allclose(integers/100000, array, atol=1e-12, rtol=0):
                raise RuntimeError('Fixed-point encoding would lose retained decimal precision')
            metadata[key+'_shape'] = list(array.shape)
            buffers.append(integers.astype('<i4').view(np.uint8).reshape(-1,4).T.tobytes())
        header = json.dumps(metadata, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
        raw = b'LCB2' + struct.pack('<I', len(header)) + header + b''.join(buffers)
    else:
        raw = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
    data = gzip.compress(raw, compresslevel=9, mtime=0)
    if len(data) > 15_000_000:
        raise RuntimeError(f'Publication file too large: {path.name}')
    temp = path.with_suffix('.tmp')
    temp.write_bytes(data)
    temp.replace(path)


def unpacked(path):
    import gzip
    raw = gzip.decompress(Path(path).read_bytes())
    if raw[:4] not in [b'LCB1', b'LCB2']:
        return json.loads(raw)
    length = struct.unpack('<I', raw[4:8])[0]
    metadata = json.loads(raw[8:8+length])
    offset = 8+length
    for key in ['cosine','neurons']:
        shape = metadata.pop(key+'_shape')
        count = int(np.prod(shape))
        if raw[:4] == b'LCB2':
            integers = np.frombuffer(raw, dtype=np.uint8, count=count*4, offset=offset).reshape(4,-1).T.copy().view('<i4').reshape(-1)
        else:
            integers = np.frombuffer(raw, dtype='<i4', count=count, offset=offset)
        metadata[key] = (integers.astype(np.float64)/metadata['fixed_point_scale']).reshape(shape).tolist()
        offset += count*4
    assert offset == len(raw), 'Trailing or missing fixed-point bytes'
    metadata.pop('fixed_point_scale')
    return metadata


def rounded(x, places=4):
    return np.round(np.asarray(x, dtype=np.float64), places).tolist()


class Exporter:
    def __init__(self, reader, lens, scratch, inventory, check):
        self.reader, self.lens, self.scratch = reader, lens, Path(scratch)
        self.inventory, self.check = inventory, check
        self.neurons = np.arange(0, 9216, 144)  # deterministic uniform index subset; not an importance ranking
        tokenizer = json.loads(reader.get(HF+'tokenizer.json'))
        self.vocab = {int(v): k for k, v in tokenizer['model']['vocab'].items()}
        self.vocab.update({int(t['id']): t['content'] for t in tokenizer['added_tokens']})

    def lens_cells(self, r1, positions, source_positions):
        source_index = {p:i for i,p in enumerate(source_positions)}
        jobs = [(layer, position) for layer in range(32) for position in positions]
        result = [[] for _ in range(32)]
        # Batch across layers as well as tokens; avoid 32 tiny memory-bound projections.
        for start in range(0, len(jobs), 16):
            batch = jobs[start:start+16]
            vectors = np.stack([r1[layer, source_index[position]] for layer,position in batch])
            logits = self.lens.project(vectors)
            top = np.argpartition(logits, -5, axis=1)[:, -5:]
            top = np.take_along_axis(top, np.argsort(-np.take_along_axis(logits, top, axis=1), axis=1), axis=1)
            maximum = np.max(logits, axis=1, keepdims=True)
            denominator = np.exp(logits-maximum).sum(axis=1)
            for row, (layer, position) in enumerate(batch):
                ids = top[row].tolist()
                probability = np.exp(logits[row, ids]-maximum[row, 0])/denominator[row]
                ranks = [int(np.count_nonzero(logits[row] > logits[row, ident]))+1 for ident in INTEREST]
                result[layer].append({'position': position, 'top': [[ident, self.vocab.get(ident, f'ID {ident}'), round(float(p), 7)] for ident,p in zip(ids,probability)], 'ranks': ranks})
            self.reader.last_progress = time.monotonic()
        return result

    def run(self):
        json_objects = sorted((o for o in self.inventory if o['key'].startswith(PREFIX+'out/') and
                            o['key'].endswith(('__free.json', '__forced.json'))), key=lambda o:o['key'])
        if len(json_objects) != 338:
            raise RuntimeError(f'Expected 338 capture metadata files, found {len(json_objects)}')
        summaries = []
        n = len(json_objects)
        vectors = np.memmap(self.scratch/'readouts.f32', mode='w+', dtype='float32', shape=(n, 3, 32, 2560))
        neuron_values = np.zeros((n, 3, 32, 64), dtype=np.float32)
        for number, obj in enumerate(json_objects):
            label = obj['key'].split('/')[-1].removesuffix('.json')
            dest = SITE/'prompts'/f'{number}.bin'
            if dest.exists():
                data = unpacked(dest)
                meta = data['meta']
                if data['source_key'] != obj['key']:
                    raise RuntimeError('Resume inventory changed')
                capture = SafeTensor(self.reader, obj['key'].replace('.json', '.safetensors'))
                length = len(meta['tokens'])
                readouts = [meta['positions'][p] for p in POSITIONS]
                vectors[number] = capture.rows('R1', readouts).transpose(1, 0, 2)
                desired = sorted(set(range(min(length, 128))) | set(readouts))
                previous = data.get('norm_positions', list(range(length)))
                slots = [previous.index(p) for p in desired]
                data['norms'] = {k: [[row[i] for i in slots] for row in values] for k,values in data['norms'].items()}
                data['norm_positions'] = desired
                packed(dest, data)
                neuron_values[number] = data['neuron_readouts']
                log(f'[{number+1}/{n}] RESUME {label}')
            else:
                log(f'[{number+1}/{n}] START {label}')
                meta = json.loads(self.reader.get(obj['key'], 0, obj['size']-1))
                capture = SafeTensor(self.reader, obj['key'].replace('.json', '.safetensors'))
                length = len(meta['tokens'])
                initial = meta['id'] == 'tay|sin x|2|hold' and meta['variant'] == 'forced'
                readouts = [meta['positions'][p] for p in POSITIONS]
                retained = list(range(length)) if initial else sorted(set(readouts))
                norm_positions = sorted(set(range(min(length, 128))) | set(readouts))
                r1 = capture.rows('R1', norm_positions)
                readout_slots = [norm_positions.index(p) for p in readouts]
                vectors[number] = r1[:, readout_slots].transpose(1, 0, 2)
                data = {'meta': meta, 'source_key': obj['key'], 'norm_positions': norm_positions, 'norms': {'R1': rounded(np.linalg.norm(r1, axis=-1), 3)},
                        'lens_positions': retained, 'lens': self.lens_cells(r1, retained, norm_positions)}
                del r1
                log(f'[{number+1}/{n}] lens complete {label}')
                for name in ['R2', 'R5']:
                    values = capture.rows(name, norm_positions)
                    data['norms'][name] = rounded(np.linalg.norm(values, axis=-1), 3)
                    del values
                log(f'[{number+1}/{n}] norm maps complete {label}')
                mlp = []
                neuron_rows = capture.rows('R4', retained)
                for layer in range(32):
                    cells = []
                    for position in retained:
                        values = neuron_rows[layer, retained.index(position)]
                        top = np.argpartition(np.abs(values), -20)[-20:]
                        top = top[np.argsort(-np.abs(values[top]))]
                        cells.append({'position': position, 'top': [[int(i), round(float(values[i]), 5)] for i in top]})
                        for p, readout in enumerate(readouts):
                            if readout == position:
                                neuron_values[number, p, layer] = values[self.neurons]
                    mlp.append(cells)
                del neuron_rows
                data['mlp_positions'], data['mlp'] = retained, mlp
                data['neuron_readouts'] = rounded(neuron_values[number], 5)
                attention = []
                # Short recordings are cheaper to fetch in one tensor range than hundreds of row requests.
                full_attention = capture.array('attn') if length <= 128 else capture.rows('attn', retained)
                # All heads and keys, all query tokens for initial prompt; readout query rows elsewhere.
                for layer in range(8):
                    heads = []
                    for head in range(16):
                        rows = []
                        for query in retained:
                            values = full_attention[layer, head, query if length <= 128 else retained.index(query)]
                            rows.append(rounded(values, 6))
                        heads.append(rows)
                    attention.append(heads)
                del full_attention
                data['attention_queries'], data['attention'] = retained, attention
                packed(dest, data)
                log(f'[{number+1}/{n}] DONE {label}; tokens={length}; file={dest.stat().st_size}; S3={self.reader.ledger["s3_received_bytes"]/1e9:.4f}GB')
            summaries.append({**{key: meta.get(key) for key in ['id', 'func', 'kth', 'template', 'variant', 'power', 'prompt', 'answer', 'coef']},
                              'index': number, 'tokens': len(meta['tokens']), 'path': f'prompts/{number}.bin'})
            if sum(p.stat().st_size for p in SITE.rglob('*') if p.is_file()) > 60_000_000:
                raise RuntimeError('Site exceeds 60 MB; do not publish')
        vectors.flush()
        for p, position in enumerate(POSITIONS):
            for layer in range(32):
                x = np.asarray(vectors[:, p, layer], dtype=np.float32)
                norms = np.linalg.norm(x, axis=1)
                normalized = x / np.maximum(norms[:, None], 1e-30)
                cosine = normalized @ normalized.T
                centered = x-x.mean(axis=0)
                gram = centered @ centered.T
                eigenvalues, eigenvectors = np.linalg.eigh(gram)
                eig = np.maximum(eigenvalues[-2:][::-1], 0)
                points = eigenvectors[:, -2:][:, ::-1]*np.sqrt(eig)
                explained = eig/max(float(np.maximum(eigenvalues, 0).sum()), 1e-30)
                packed(SITE/'across'/f'{position}-{layer}.bin', {'cosine': rounded(cosine, 5),
                    'pca': rounded(points, 5), 'explained_variance_ratio': rounded(explained, 6),
                    'neurons': rounded(neuron_values[:, p, layer], 5)})
            log(f'ACROSS PROMPTS complete: {position}')
        counts = {key:dict(Counter(str(row.get(key) if row.get(key) is not None else 'not applicable') for row in summaries)) for key in ['template', 'variant', 'func']}
        sizes = {name:sum(o['size'] for o in self.inventory if o['key'].startswith(PREFIX+name)) for name in ['out/', 'dryruns/', 'meta/']}
        manifest = {'rows':summaries, 'counts':counts, 'source_sizes':sizes,
            'source_total_bytes':sum(o['size'] for o in self.inventory),
            'capture_bytes':sum(o['size'] for o in self.inventory if o['key'].endswith(('__free.json','__forced.json','__free.safetensors','__forced.safetensors')) and o['key'].startswith(PREFIX+'out/')),
            'unique_prompts':len({r['id'] for r in summaries}), 'check':self.check,
            's3_received_bytes':self.reader.ledger['s3_received_bytes'], 's3_reserved_bytes':self.reader.ledger['s3_reserved_bytes'],
            'neuron_ids':self.neurons.tolist(), 'interest_ids':INTEREST, 'revision':REVISION,
            'sampling':[
                'All 338 captures: complete metadata, full replies, every token; all prompts retained for comparisons and PCA.',
                'R1/R2/R5 norms: all 32 layers, first 128 token positions plus instr_end, prompt_end and reply_end. Captures of 128 tokens or fewer have full maps; other cells are unavailable. No reply or token text is shortened.',
                'Initial forced sin x, term 2, hold: logit lens and top-20 neurons at every layer and token; all attention query tokens, keys and heads.',
                'Other captures: logit lens, top-20 neurons and attention query rows at instr_end, prompt_end and reply_end only. All attention keys and all 16 heads in each of 8 full-attention layers retained. Uncomputed positions are unavailable, never zero-filled.',
                'Across-prompt neuron values: 64 evenly spaced neuron indices (0, 144, 288 through 9072), all 32 layers and all three readouts. Other neuron indices are unavailable in this view.',
                'Rounding: norms 3 decimal places; attention 6; neuron values 5; probabilities 7; cosine/PCA coordinates 5. PCA is centered raw R1, without scaling; separate fit for each layer/readout across all 338 captures, including both variants.',
                'S (DeltaNet memory), raw residual vectors, raw MLP arrays and full vocabulary logits are not published. Top-five probabilities use the full 248,320-token vocabulary. Ranks use 1 + count of strictly larger logits; ties share rank.']}
        save(SITE/'manifest.json', manifest)
        compact_exports()
        sizes = [p.stat().st_size for p in SITE.rglob('*') if p.is_file()]
        if sum(sizes) > 60_000_000 or max(sizes) > 15_000_000:
            raise RuntimeError('Static publication size gate failed')
        log(f'BUILD COMPLETE site={sum(sizes)} bytes largest={max(sizes)} bytes S3={self.reader.ledger["s3_received_bytes"]/1e9:.6f}GB')


def compact_exports():
    """Remove float32 JSON expansion without changing the documented decimal precision."""
    for path in sorted((SITE/'prompts').glob('*.bin')):
        data = unpacked(path)
        data['norms'] = {k:rounded(v, 3) for k,v in data['norms'].items()}
        data['attention'] = rounded(data['attention'], 6)
        data['neuron_readouts'] = rounded(data['neuron_readouts'], 5)
        packed(path, data)
    for path in sorted((SITE/'across').glob('*.bin')):
        data = unpacked(path)
        for k in ['cosine', 'pca', 'neurons']:
            data[k] = rounded(data[k], 5)
        data['explained_variance_ratio'] = rounded(data['explained_variance_ratio'], 6)
        packed(path, data)
    manifest_path = SITE/'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        totals = Counter()
        coverage = Counter()
        for row in manifest['rows']:
            data = unpacked(SITE/row['path'])
            for name,shape in data['meta']['saved'].items():
                totals[name] += int(np.prod(shape))*(4 if name == 'S' else 2)
            coverage['tokens'] += len(data['meta']['tokens'])
            coverage['norm_tokens'] += len(data['norm_positions'])
            coverage['lens_tokens'] += len(data['lens_positions'])
            coverage['attention_queries'] += len(data['attention_queries'])
            coverage['norm_subsampled_captures'] += int(len(data['norm_positions'])<len(data['meta']['tokens']))
        manifest['tensor_bytes'] = dict(totals)
        manifest['coverage'] = dict(coverage)
        ledger = json.loads((ROOT/'read_ledger.json').read_text())
        manifest['s3_received_bytes'] = ledger['s3_received_bytes']
        manifest['s3_reserved_bytes'] = ledger['s3_reserved_bytes']
        for _ in range(3):
            manifest['site_bytes'] = sum(p.stat().st_size for p in SITE.rglob('*') if p.is_file())
            save(manifest_path, manifest)
    log('Compacted existing derived JSON at its documented decimal precision')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--compact', action='store_true', help='Compact completed derived files locally, with no network access')
    ap.add_argument('--check-only', action='store_true', help='Stop after the mandatory final-layer check')
    ap.add_argument('--opening-balance', required=True, help='Recorded AWS opening balance/budget or explicit user waiver')
    args = ap.parse_args()
    SITE.mkdir(exist_ok=True)
    if args.compact:
        compact_exports()
        return
    reader = Reader(args.opening_balance)
    try:
        with tempfile.TemporaryDirectory(prefix='latent-cube-', dir='/tmp') as scratch, threadpool_limits(limits=4):
            inventory = reader.inventory()
            wanted = PREFIX+'out/'+filename('tay|sin x|2|hold')+'__forced.json'
            size = next(o['size'] for o in inventory if o['key'] == wanted)
            meta = json.loads(reader.get(wanted, 0, size-1))
            save(SITE/'initial_prompt.json', meta)
            capture = SafeTensor(reader, wanted.replace('.json', '.safetensors'))
            lens = Lens(reader, scratch)
            check = lens.check(capture, meta)
            if not args.check_only:
                Exporter(reader, lens, scratch, inventory, check).run()
    finally:
        reader.close()


if __name__ == '__main__':
    main()
