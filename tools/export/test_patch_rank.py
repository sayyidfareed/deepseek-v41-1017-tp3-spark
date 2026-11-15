# Modified for the public TP3 recipe; source attribution in CREDITS.md.
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import patch_rank as repair


def fixture(folder, *, confidence='float16', extra=False, multiplier=-2082680531, multiplier_dtype="I32"):
    overlay = folder / 'overlay.safetensors'
    h, data = {}, bytearray()
    for layer in range(10, 36):
        for part, dtype, shape, value in (
                ('trellis', 'I16', [512, 2, 1], struct.pack('<h', 7)),
                ('suh', 'F16', [8192], struct.pack('<e', 2)),
                ('svh', 'F16', [4], struct.pack('<e', 3)),
                ('mul1', multiplier_dtype, [], struct.pack('<i' if multiplier_dtype=='I32' else '<e', multiplier))):
            payload = value * repair.prod(shape)
            h[f'layers.{layer}.attn.wo_b.{part}'] = {
                'dtype': dtype, 'shape': shape, 'data_offsets': [len(data), len(data)+len(payload)]}
            data.extend(payload)
    if extra:
        h['mtp.2.bad.weight'] = {'dtype': 'F16', 'shape': [0], 'data_offsets': [len(data), len(data)]}
    header = json.dumps(h).encode()
    overlay.write_bytes(struct.pack('<Q', len(header)) + header + data)
    tools = repair.exporter()
    specs = {k: dict(v, bytes=v['data_offsets'][1]-v['data_offsets'][0]) for k, v in h.items()}
    rows, payload = [], bytearray()
    for layer in range(10, 36):
        for part in ('tr', 'suh', 'svh'):
            key = f'layers.{layer}.attn.wo_b|None|(6144, 9216)|{part}'
            l = tools.layout(key, specs)
            rows.append([key, l['dtype'], l['outshape'], len(payload), l['bytes']])
            payload.extend(bytes([165])*l['bytes'])
    count = 5376*repair.NATIVE_SIZES[confidence]
    rows.append(['mtp.2.confidence_head.proj.weight|None', confidence, [1, 5376], len(payload), count])
    payload.extend(bytes([90])*count)
    source = folder / 'baseline.bin'
    source.write_bytes(payload + json.dumps(rows).encode() + struct.pack('<Q8s', len(payload), b'TFDSRK01'))
    return source, overlay, folder / 'candidate.bin', rows


class OverlayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)

    def run_apply(self, s, o, d, **kwargs):
        with patch.object(repair.shutil, 'disk_usage', return_value=SimpleNamespace(free=1 << 50)):
            return repair.apply(s, o, d, repair.digest(s), repair.digest(o), **kwargs)

    def test_real_exporter_padding_and_untouched_mtp(self):
        s, o, d, rows = fixture(self.folder)
        before = s.read_bytes()
        self.assertEqual(self.run_apply(s, o, d, write=True)['patched_native_tensors'], 78)
        after = d.read_bytes()
        self.assertEqual(s.read_bytes(), before)
        self.assertEqual(after[rows[-1][3]:], before[rows[-1][3]:])
        for row in rows[:-1]:
            raw = after[row[3]:row[3]+row[4]]
            if row[0].endswith('|suh'):
                self.assertEqual(raw, struct.pack('<e', 2)*2048 + bytes(1024*2))
            elif row[0].endswith('|tr'):
                self.assertEqual(raw, struct.pack('<h', 7)*(row[4]//2))
            else:
                self.assertEqual(raw, struct.pack('<e', 3)*4)

    def test_real_overlay_integer_multiplier(self):
        s, o, d, _ = fixture(self.folder, multiplier_dtype='I32')
        self.assertEqual(self.run_apply(s, o, d, write=True)['patched_native_tensors'], 78)

    def test_integer_nonunit_multiplier_refused(self):
        s, o, d, _ = fixture(self.folder, multiplier_dtype='I32', multiplier=2)
        with self.assertRaisesRegex(ValueError, 'multiplier'):
            self.run_apply(s, o, d)

    def test_float_unit_is_not_codebook_seed(self):
        s, o, d, _ = fixture(self.folder, multiplier_dtype='F16', multiplier=1)
        with self.assertRaisesRegex(ValueError, 'codebook multiplier'):
            self.run_apply(s, o, d)

    def test_dry_run_writes_nothing(self):
        s, o, d, _ = fixture(self.folder)
        self.assertEqual(self.run_apply(s, o, d)['status'], 'prepared-not-written')
        self.assertFalse(d.exists())

    def test_f32_confidence_refused(self):
        s, o, d, _ = fixture(self.folder, confidence='float32')
        with self.assertRaisesRegex(ValueError, 'confidence'):
            self.run_apply(s, o, d)

    def test_mtp_in_overlay_refused(self):
        s, o, d, _ = fixture(self.folder, extra=True)
        with self.assertRaisesRegex(ValueError, '104'):
            self.run_apply(s, o, d)

    def test_wrong_codebook_multiplier_refused(self):
        s, o, d, _ = fixture(self.folder, multiplier=2)
        with self.assertRaisesRegex(ValueError, 'multiplier'):
            self.run_apply(s, o, d)

    def test_existing_destination_preserved(self):
        s, o, d, _ = fixture(self.folder)
        d.write_bytes(b'keep')
        with self.assertRaises(FileExistsError):
            self.run_apply(s, o, d, write=True)
        self.assertEqual(d.read_bytes(), b'keep')

    def test_bad_source_identity_leaves_no_candidate(self):
        s, o, d, _ = fixture(self.folder)
        with patch.object(repair.shutil, 'disk_usage', return_value=SimpleNamespace(free=1 << 50)):
            with self.assertRaisesRegex(ValueError, 'baseline identity'):
                repair.apply(s, o, d, '0'*64, repair.digest(o), write=True)
        self.assertFalse(d.exists())
        self.assertFalse(d.with_suffix('.bin.partial').exists())


if __name__ == '__main__':
    unittest.main()
