# Modified for the public TP3 recipe; source attribution in CREDITS.md.
"""Apply only the matching attention overlay to a NEW TP3 rank cache.

Reuses the accepted TP3 exporter; MTP and all other tensor bytes stay unchanged.
Dry-run by default. Does not load a model, change a service, or touch live files.
"""
import argparse
import hashlib
import importlib
import json
from math import prod
import os
from pathlib import Path
import shutil
import struct
import sys

PARTS = ('trellis', 'suh', 'svh', 'mul1')
NAMES = {f'layers.{i}.attn.wo_b.{p}' for i in range(10, 36) for p in PARTS}
SIZES = {'F16': 2, 'F32': 4, 'I16': 2, 'I32': 4}
NATIVE_SIZES = {'float16': 2, 'bfloat16': 2, 'float32': 4, 'int16': 2,
                'int32': 4, 'int64': 8, 'uint8': 1, 'float8_e4m3fn': 1}
PINS = {
    'rank_cache.py': '85f1fcd65f958776f126bafa8c48db8ee0875ba262f9c30d8573657aaf4b172b',
    'tp3_layout.py': '13bb07dacbef1ae16fd8455cb8e9baa7b04960a0a28129db54c3150fca9de0e2',
    'materialize_checkpoint.py': 'b458bae51836075ec33edfa35d34debb2ea32cc68563133ae53b225784002abc',
    'source_verify.py': 'e32dc46010819d07cc67b0409f493bc079084179e105bde6f6b86dc0e525fb47',
    'io_budget.py': '874e1b0952c755982c49cc5d3aa3df8cd02194472d110175559e396a74a5aa43',
}


def digest(path, budget=None):
    h = hashlib.sha256()
    with path.open('rb') as f:
        if budget and hasattr(os, 'POSIX_FADV_NOREUSE'):
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_NOREUSE)
        def read():
            if budget: budget.before_read(1 << 20)
            return f.read(1 << 20)
        for chunk in iter(read, b''):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(path):
    s = path.stat()
    if not path.is_file() or path.is_symlink():
        raise ValueError('regular source required')
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns


def exporter():
    folder = Path(__file__).parent
    for name, expected in PINS.items():
        if digest(folder / name) != expected:
            raise ValueError('exporter identity mismatch')
    sys.path.insert(0, str(folder))
    return importlib.import_module('rank_cache')


def native_index(path):
    size = path.stat().st_size
    with path.open('rb') as f:
        f.seek(-16, 2)
        end, magic = struct.unpack('<Q8s', f.read(16))
        if magic != b'TFDSRK01' or not 0 < size-end-16 <= 64 << 20:
            raise ValueError('native footer invalid')
        f.seek(end)
        rows = json.loads(f.read(size-end-16))
    at, seen = 0, set()
    for key, dtype, shape, offset, count in rows:
        if key in seen or dtype not in NATIVE_SIZES or offset != at:
            raise ValueError('native index invalid')
        if any(type(n) is not int or n < 0 for n in shape):
            raise ValueError('native shape invalid')
        if count != prod(shape) * NATIVE_SIZES[dtype]:
            raise ValueError('native byte count invalid')
        seen.add(key)
        at += count
    if at != end:
        raise ValueError('native payload coverage invalid')
    confidence = [r for r in rows if r[0] == 'mtp.2.confidence_head.proj.weight|None']
    if len(confidence) != 1 or confidence[0][1:3] != ['float16', [1, 5376]]:
        raise ValueError('Mia draft confidence must remain F16')
    return rows


def overlay_header(path):
    with path.open('rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        if not 0 < n <= 1 << 20:
            raise ValueError('overlay header limit')
        h = json.loads(f.read(n))
        h.pop('__metadata__', None)
        if set(h) != NAMES:
            raise ValueError('expected exactly 104 attention tensors; MTP edits refused')
        at = 0
        for name, spec in sorted(h.items(), key=lambda pair: pair[1]['data_offsets'][0]):
            dtype, shape, offsets = spec['dtype'], spec['shape'], spec['data_offsets']
            if dtype not in SIZES or any(type(x) is not int or x < 0 for x in shape):
                raise ValueError('overlay dtype or shape')
            length = prod(shape) * SIZES[dtype]
            if offsets != [at, at+length]:
                raise ValueError('overlay payload coverage')
            spec['bytes'] = length
            if name.endswith('.mul1'):
                if shape != [] or dtype != 'I32':
                    raise ValueError('scalar I32 codebook multiplier required')
                f.seek(8+n+at)
                # EXL3 mul1 is a codebook seed, not an output scale. Matches
                # exl3_decode.cuh decode2<CB_MUL1>: state * 0x83DCD12D.
                if struct.unpack('<I', f.read(length))[0] != 0x83DCD12D:
                    raise ValueError('native codebook multiplier mismatch')
            at += length
        if 8+n+at != path.stat().st_size:
            raise ValueError('overlay size mismatch')
    return h, 8+n


def plan(source, overlay, tools):
    rows = native_index(source)
    specs, base = overlay_header(overlay)
    patches = []
    for row in rows:
        name = row[0].split('|')[0]
        if name not in {f'layers.{i}.attn.wo_b' for i in range(10, 36)}:
            continue
        layout = tools.layout(row[0], specs)
        if (layout['dtype'], layout['outshape'], layout['bytes']) != (row[1], row[2], row[4]):
            raise ValueError('overlay and accepted native tensor layout differ')
        patches.append((row, layout))
    if len(patches) != 78 or {x[1]['source_name'] for x in patches} != NAMES - {n for n in NAMES if n.endswith('.mul1')}:
        raise ValueError('expected all 78 native attention slices')
    return patches, base


def apply(source, overlay, output, source_sha, overlay_sha, *, write=False, read_mib_s=256):
    before = fingerprint(source), fingerprint(overlay)
    tools = exporter()
    budget = tools.Budget(read_mib_s)
    patches, base = plan(source, overlay, tools)
    report = {'status': 'prepared-not-written', 'patched_native_tensors': len(patches),
              'mtp_policy': 'all bytes preserved; confidence F16 required',
              'artifact_bytes': source.stat().st_size}
    if not write:
        return report
    partial = output.with_suffix(output.suffix + '.partial')
    if output.exists() or output.is_symlink() or partial.exists() or partial.is_symlink():
        raise FileExistsError('destination already exists')
    if shutil.disk_usage(output.parent).free < source.stat().st_size + 50 * 1024**3:
        raise RuntimeError('50 GiB reserve would be violated')
    if digest(overlay, budget) != overlay_sha:
        raise ValueError('overlay identity mismatch')
    created = False
    try:
        with source.open('rb') as src, partial.open('xb') as dst:
            created = True
            h = hashlib.sha256()
            reader = tools.Reader(src, budget)
            for block in iter(lambda: reader.read(1 << 20), b''):
                h.update(block)
                dst.write(block)
            if h.hexdigest() != source_sha:
                raise ValueError('baseline identity mismatch')
            with overlay.open('rb') as sidecar:
                for row, layout in patches:
                    dst.seek(row[3])
                    tools.write_tensor(sidecar, base, layout, dst)
                    if dst.tell() != row[3] + row[4]:
                        raise ValueError('patch byte count mismatch')
            dst.flush()
            os.fsync(dst.fileno())
        if before != (fingerprint(source), fingerprint(overlay)):
            raise ValueError('source changed during preparation')
        report.update(status='written-not-GPU-qualified', artifact_sha256=digest(partial, budget))
        os.link(partial, output)
        return report
    finally:
        if created and partial.exists():
            partial.unlink()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'overlay', 'output', 'source-sha256', 'overlay-sha256'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--write', action='store_true')
    p.add_argument('--read-mib-s', type=float, default=256)
    a = p.parse_args()
    print(json.dumps(apply(Path(a.source), Path(a.overlay), Path(a.output),
                           a.source_sha256, a.overlay_sha256, write=a.write, read_mib_s=a.read_mib_s), indent=2))


if __name__ == '__main__':
    main()
