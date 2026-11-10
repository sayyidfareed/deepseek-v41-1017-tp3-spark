# Retained/adapted TP3 integration source; original copyrights retained. See CREDITS.md and licenses/.
"""Plan the exact deployed EAQ pack against the native three-rank loader keys.

Only headers are consumed here. This is an export compatibility check, not
payload verification, arithmetic qualification or an inference benchmark.
"""
import argparse
import hashlib
import json
from math import prod
from pathlib import Path, PurePosixPath

from materialize_checkpoint import SIZES
from rank_cache import layout
from tp3_layout import extra_name, patch_config, transformation

REPO = 'Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw'
REVISION = '45ee5d0fd106c94a3d77f721e09cbb2f50e5a16b'
PREFIX = 'models--' + REPO.replace('/', '--') + '/snapshots/' + REVISION + '/'


def read_json(path):
    if path.stat().st_size > 100 * 1024**2:
        raise ValueError('Metadata exceeds the owner whole-read ceiling')
    return json.loads(path.read_text())


def tensor_plan(headers, index):
    tensors = {}
    header_pins = {}
    for shard, record in headers.items():
        rel = PurePosixPath(shard)
        if rel.is_absolute() or len(rel.parts) != 1 or not shard.endswith('.safetensors'):
            raise ValueError('Unexpected source shard path')
        header_pins[shard] = record['header_sha256']
        ranges = []
        for name, spec in record['tensors'].items():
            if name == '__metadata__':
                continue
            if name in tensors or index['weight_map'].get(name) != shard:
                raise ValueError('Duplicate or incorrectly indexed source tensor')
            lo, hi = spec['data_offsets']
            size = prod(spec['shape']) * SIZES[spec['dtype']]
            if not (0 <= lo <= hi <= record['file_bytes'] - 8 - record['header_bytes']):
                raise ValueError('Source tensor outside shard')
            if hi - lo != size:
                raise ValueError('Source tensor shape disagrees with byte range')
            tensors[name] = {**spec, 'bytes': size, 'source': 'mia',
                             'shard': PREFIX + shard}
            ranges.append((lo, hi))
        ranges.sort()
        if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
            raise ValueError('Overlapping source tensor payloads')
    if set(tensors) != set(index['weight_map']):
        raise ValueError('Source index/header tensor coverage differs')
    for name in tensors:
        alias = extra_name(name)
        if alias is not None and alias in tensors:
            raise ValueError('Physical padding would shadow an original tensor')
    return tensors, header_pins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', required=True, type=Path)
    parser.add_argument('--native-keys', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    config = read_json(args.metadata / 'config.json')
    padded = patch_config(config)
    quant = config['quantization_config']
    if quant['quant_method'] != 'exl3' or quant['codebook'] != 'mul1':
        raise ValueError('Expected the current EXL3 mul1 pack')
    tensors, header_pins = tensor_plan(
        read_json(args.metadata / 'headers.json'),
        read_json(args.metadata / 'model.safetensors.index.json'))
    report = {'status': 'HEADERS_CHECKED_NOT_GPU_QUALIFIED', 'repository': REPO,
              'revision': REVISION, 'source_tensors': len(tensors),
              'transformed_source_tensors': sum(transformation(n) is not None for n in tensors),
              'rank_plans': [], 'payload_bytes_read': 0, 'gpu_calls': 0}
    failures = 0
    examples = []
    for rank in range(3):
        key_file = args.native_keys / f'rank{rank}.keys'
        if key_file.stat().st_size > 100 * 1024**2:
            raise ValueError('Native key file exceeds whole-read ceiling')
        raw = key_file.read_bytes()
        keys = raw.decode().splitlines()
        if len(keys) != len(set(keys)):
            raise ValueError('Native loader plan contains duplicate keys')
        payload = 0
        rank_failures = 0
        for key in keys:
            try:
                entry = layout(key, tensors)
            except (ValueError, KeyError) as error:
                rank_failures += 1
                if len(examples) < 12:
                    examples.append({'rank': rank, 'key': key, 'error': str(error)})
            else:
                payload += entry['bytes']
        failures += rank_failures
        report['rank_plans'].append({'rank': rank, 'entries': len(keys),
                                     'keys_sha256': hashlib.sha256(raw).hexdigest(),
                                     'payload_bytes': payload, 'errors': rank_failures})
    report.update(errors=failures, examples=examples)
    if not failures:
        plan = {'arm': 'MIA1017_EAQ_TP3', 'sources': {
            'mia': {'repo': REPO, 'revision': REVISION}},
            'source_header_sha256': header_pins, 'tensors': tensors}
        with (args.out / 'checkpoint-plan.json').open('x') as file:
            json.dump(plan, file, separators=(',', ':'))
        (args.out / 'config.json').write_text(json.dumps(padded, indent=2) + '\n')
        report['status'] = 'NATIVE_PLANS_HEADER_COMPATIBLE_NOT_PAYLOAD_VERIFIED'
    (args.out / 'compatibility.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
