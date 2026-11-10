import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('launch', ROOT/'scripts/render_launch.py')
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class Packaging(unittest.TestCase):
    def test_overlay_hashes_and_paths(self):
        manifest=json.loads((ROOT/'provenance.json').read_text())
        listed=set()
        for e in manifest['files']:
            p=Path(e['path'])
            self.assertFalse(p.is_absolute())
            self.assertNotIn('..',p.parts)
            self.assertNotIn(str(p),listed)
            listed.add(str(p))
            self.assertEqual(hashlib.sha256((ROOT/'overlay'/p).read_bytes()).hexdigest(),e['export_sha256'])
        self.assertEqual(listed,{str(p.relative_to(ROOT/'overlay')) for p in (ROOT/'overlay').rglob('*') if p.is_file()})

    def config(self):
        c=json.loads((ROOT/'config/cluster.example.json').read_text())
        c['image']='sha256:'+'a'*64
        for r in c['ranks']:
            r['rank_cache_sha256']='b'*64
            r['network_env']={'NCCL_NET':'IB'}
        return c

    def test_head_loopback_and_three_rank_contract(self):
        c=self.config();head=launch.command(c,0)
        self.assertIn('TF_TP_WORLD=3',head)
        self.assertEqual(head[head.index('--host')+1],'127.0.0.1')
        self.assertEqual(head[head.index('--parallel')+1],'12')
        self.assertIn('TF_DS_POOL=6400000',head)
        self.assertIn('TF_DS_KV_SHARD=0',head)
        self.assertIn('--log-driver=none',head)
        for rank in (1,2):
            worker=launch.command(c,rank)
            pos=worker.index('/rank-cache',worker.index(c['image']))
            self.assertEqual(worker[pos+1:pos+3],[str(rank),'3'])

    def test_configuration_cannot_override_profile(self):
        c=self.config();c['ranks'][0]['network_env']['TF_DS_POOL']='1'
        with self.assertRaises(ValueError):launch.command(c,0)

    def test_example_cannot_be_launched_as_is(self):
        with self.assertRaises(ValueError):launch.command(json.loads((ROOT/'config/cluster.example.json').read_text()),0)

    def test_invalid_rank_and_unpinned_image_refused(self):
        c=self.config()
        with self.assertRaises(ValueError):launch.command(c,3)
        c['image']='latest'
        with self.assertRaises(ValueError):launch.command(c,0)

    def test_paths_cannot_inject_bind_options(self):
        c=self.config();c['ranks'][0]['runtime']='/srv/model,readonly=false'
        with self.assertRaises(ValueError):launch.command(c,0)

    def test_required_notices_preserved(self):
        for name in ('LICENSE','NOTICE','CREDITS.md','licenses/mia/NOTICE','licenses/laky/NOTICE','licenses/tensorfold/NOTICE','licenses/kernels/LICENSE.ExLlamaV3'):
            self.assertGreater((ROOT/name).stat().st_size,100)

if __name__=='__main__':unittest.main()
