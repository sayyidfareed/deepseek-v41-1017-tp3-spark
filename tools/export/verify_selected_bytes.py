# Retained/adapted TP3 integration source; original copyrights retained. See CREDITS.md and licenses/.
"""Verify native three-rank slices against selected real EXL3 payload bytes.

No tensor decoding, GPU calls, or source writes. This is a bounded sample,
not verification of all checkpoint bytes or numerical model qualification.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import struct

from io_budget import Budget, Reader
from rank_cache import layout, write_tensor
from source_verify import fingerprint, safe_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', required=True, type=Path)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.cases.stat().st_size > 1 << 20 or args.out.exists():
        raise ValueError('Unexpected metadata size or existing receipt')
    cases = json.loads(args.cases.read_text())
    budget = Budget(4)  # Four MiB/s, no global page-cache changes.
    results = []
    for case in cases:
        name, spec = case['name'], case['spec']
        if spec['dtype'] != 'I16' or not 0 < spec['bytes'] <= 8 << 20:
            raise ValueError('Only the bounded quantized-byte cases are supported')
        path = safe_path(args.root, spec['shard'])
        before = fingerprint(path)
        with Reader(path.open('rb'), budget) as file:
            length = struct.unpack('<Q', file.read(8))[0]
            if length > 100 << 20:
                raise ValueError('Header exceeds whole-read ceiling')
            header_bytes = file.read(length)
            if hashlib.sha256(header_bytes).hexdigest() != case['header_sha256']:
                raise ValueError('Pinned source header differs')
            actual = json.loads(header_bytes)[name]
            if actual != {k: spec[k] for k in ('shape', 'dtype', 'data_offsets')}:
                raise ValueError('Tensor metadata differs')
            base = 8 + length
            pieces = []
            for key in case['keys']:
                entry = layout(key, {name: spec})
                output = io.BytesIO()
                write_tensor(file, base, entry, output)
                if output.tell() != entry['bytes']:
                    raise ValueError('Slice byte accounting differs')
                pieces.append(output.getvalue())
            if len(pieces) != 3:
                raise ValueError('Expected three native rank slices')
            # Restore original row-major ordering, independently of the
            # exporter's selected-interval and virtual-padding functions.
            if case['axis'] == 0:
                restored = b''.join(pieces)
            elif case['axis'] == 1:
                rows, _, _ = spec['shape']
                row_bytes = len(pieces[0]) // rows
                restored = b''.join(part[r*row_bytes:(r+1)*row_bytes]
                                    for r in range(rows) for part in pieces)
            else:
                raise ValueError('Unsupported reassembly axis')
            file.seek(base + spec['data_offsets'][0])
            original = file.read(spec['bytes'])
            if len(original) != spec['bytes'] or restored != original:
                raise ValueError('Quantized source bytes were not preserved')
        if fingerprint(path) != before:
            raise ValueError('Source file changed during verification')
        results.append({'tensor': name, 'bytes': len(original),
                        'source_sha256': hashlib.sha256(original).hexdigest(),
                        'rank_sha256': [hashlib.sha256(x).hexdigest() for x in pieces],
                        'reconstructed_exact': True})
    result = {'status': 'SELECTED_BYTES_PRESERVED_NOT_FULL_SOURCE_VERIFIED',
              'tensors': results, 'gpu_calls': 0, 'source_writes': 0,
              'read_limit_mib_s': 4}
    with args.out.open('x') as file:
        json.dump(result, file, indent=2)
        file.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
