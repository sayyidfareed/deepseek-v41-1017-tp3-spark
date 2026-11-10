# Retained/adapted TP3 integration source; original copyrights retained. See CREDITS.md and licenses/.
"""CPU checks of the actual export path's quantized bytes and padding."""
import io
import struct
import unittest

from rank_cache import layout, write_tensor, VirtualTensor
from tp3_layout import Append, extra_name
from prepare_checkpoint import tensor_plan


def spec(shape, dtype='I16'):
    size = {'I16': 2, 'F16': 2}[dtype]
    for n in shape:
        size *= n
    return {'shape': shape, 'dtype': dtype, 'bytes': size,
            'data_offsets': [0, size], 'source': 'mia', 'shard': 'fixture'}


def exported(raw, key, tensors):
    entry = layout(key, tensors)
    out = io.BytesIO()
    write_tensor(io.BytesIO(raw), 0, entry, out)
    result = out.getvalue()
    if len(result) != entry['bytes']:
        raise AssertionError('Export byte accounting differs')
    return result


class QuantizedExport(unittest.TestCase):
    def test_three_expert_column_parts_reconstruct_every_original_word(self):
        name = 'layers.18.ffn.experts.0.w1'
        tensor = spec([16, 48, 32])  # K=256, N=768; three N=256 parts
        raw = b''.join(struct.pack('<H', i % 65536)
                       for i in range(tensor['bytes'] // 2))
        parts = [exported(raw, f'{name}|({r*256}, {(r+1)*256})|None|tr',
                          {name + '.trellis': tensor}) for r in range(3)]
        row_bytes = 48 * 32 * 2
        reconstructed = b''.join(part[row*row_bytes//3:(row+1)*row_bytes//3]
                                 for row in range(16) for part in parts)
        self.assertEqual(reconstructed, raw)

    def test_three_expert_down_parts_reconstruct_every_original_word(self):
        name = 'layers.18.ffn.experts.0.w2'
        tensor = spec([48, 8, 32])  # K=768, N=128; three K=256 parts
        raw = bytes(range(256)) * (tensor['bytes'] // 256)
        pieces = [exported(raw, f'{name}|None|({r*256}, {(r+1)*256})|tr',
                           {name + '.trellis': tensor}) for r in range(3)]
        self.assertEqual(b''.join(pieces), raw)

    def test_refuses_partial_hadamard_blocks(self):
        with self.assertRaisesRegex(ValueError, 'Hadamard block'):
            layout('x|(0, 192)|None|tr', {'x.trellis': spec([16, 48, 32])})

    def test_added_attention_group_is_an_exact_alias_of_group_zero(self):
        name = 'layers.7.attn.wo_a.slice.0.trellis'
        alias = extra_name(name)
        tensor = spec([8, 8, 32])
        raw = bytes(range(256)) * (tensor['bytes'] // 256)
        key = alias.removesuffix('.trellis') + '|None|None|tr'
        self.assertEqual(exported(raw, key, {name: tensor}), raw)

    def test_wo_b_original_scales_survive_and_only_added_group_is_zero(self):
        name = 'layers.7.attn.wo_b.suh'
        tensor = spec([8192], 'F16')
        raw = b''.join(struct.pack('<H', i) for i in range(8192))
        result = exported(raw, name + '|None', {name: tensor})
        self.assertEqual(result[:len(raw)], raw)
        self.assertEqual(result[len(raw):], bytes(1024 * 2))

    def test_virtual_padding_handles_reads_across_multiple_outer_rows(self):
        tensor = spec([3, 8, 2])
        raw = bytes(range(tensor['bytes']))
        view = VirtualTensor(io.BytesIO(raw), 0, tensor, Append(1, 4))
        expected = b''.join(raw[i*32:(i+1)*32] + raw[i*32:i*32+16]
                            for i in range(3))
        self.assertEqual(b''.join(view.chunks(29, 90)), expected[29:119])
        with self.assertRaisesRegex(ValueError, 'outside tensor'):
            list(view.chunks(view.size - 1, 2))

    def test_truncated_source_never_yields_a_completed_tensor(self):
        name = 'layers.7.attn.wo_b.suh'
        tensor = spec([8192], 'F16')
        with self.assertRaisesRegex(ValueError, 'Truncated'):
            exported(bytes(2), name + '|None', {name: tensor})

    def test_exact_fp16_widening_preserves_values_and_signed_zero(self):
        raw = b''.join(struct.pack('<H', x)
                       for x in [0, 0x8000, 1, 0x3c00, 0x7bff, 0x7c00])
        result = exported(raw, 'norm|torch.float32', {'norm': spec([6], 'F16')})
        expected = b''.join(struct.pack('<f', x[0])
                            for x in struct.iter_unpack('<e', raw))
        self.assertEqual(result, expected)

    def test_header_overlap_refused_before_any_payload_is_read(self):
        headers = {'fixture.safetensors': {
            'header_sha256': 'unused', 'header_bytes': 128, 'file_bytes': 144,
            'tensors': {'a': {'shape': [2], 'dtype': 'I16', 'data_offsets': [0, 4]},
                        'b': {'shape': [2], 'dtype': 'I16', 'data_offsets': [2, 6]}}}}
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            tensor_plan(headers, {'weight_map': {'a': 'fixture.safetensors',
                                               'b': 'fixture.safetensors'}})


if __name__ == '__main__':
    unittest.main()
