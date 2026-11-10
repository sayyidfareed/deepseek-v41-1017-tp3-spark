#!/usr/bin/env python3
"""Print one rank's Docker command. Does not execute commands or contact hosts."""
import argparse
import json
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parents[1]


def command(config, rank):
    if rank not in (0, 1, 2):
        raise ValueError('This recipe requires exactly three ranks')
    profile = json.loads((ROOT / 'config/profile.json').read_text())
    local = config['ranks'][rank]
    for key in ('runtime', 'kit', 'rank_cache', 'engram', 'serving', 'spill'):
        if not Path(local[key]).is_absolute() or ',' in local[key] or '\n' in local[key]:
            raise ValueError('Use absolute, comma-free local paths: ' + key)
    image = config['image']
    if '@sha256:' not in image and not image.startswith('sha256:'):
        raise ValueError('Pin the locally prepared image by digest')
    env = dict(profile['env'])
    env.update(local['network_env'])
    if any(not k.startswith('NCCL_') for k in local['network_env']):
        raise ValueError('Only NCCL settings belong in network_env')
    env.update(TF_DS_RANK_CACHE='/rank-cache', TF_DS_RANK_CACHE_SHA=local['rank_cache_sha256'],
               TF_DS_KIT='/runtime/kit', TF_DS_ENGRAM='/engram',
               TF_DS_TOKEN_MAP='/runtime/dsv41_token_map.json')
    if len(local['rank_cache_sha256']) != 64 or any(c not in '0123456789abcdef' for c in local['rank_cache_sha256']):
        raise ValueError('A verified rank-cache SHA256 is required')
    master, port = config['master'], str(config['port'])
    if rank == 0:
        env.update(TF_TP_WORLD='3', TF_TP_MASTER=master, TF_TP_PORT=port)
        program = 'tensorfold-native'
        args = ['serve', '/runtime/model', '--backend', 'cuda', '--host', '127.0.0.1',
                '--port', '8000', '--name', profile['model'], '--context', str(profile['context_tokens']),
                '--parallel', '12', '--max-tokens', '65536', '--temperature', '1.0', '--top-p', '0.95',
                '--top-k', '20', '--thinking', '--spill-dir', '/spill', '--spill-gib', '64',
                '--spill-min-free-gib', '50', '--spill-min-tokens', '8192', '--no-update-check']
    else:
        program = 'tf-dsv41-lanes'
        args = ['/runtime/model', '/rank-cache', str(rank), '3', master, port,
                '/runtime/kit', '--engram', '/engram', '--token-map', '/runtime/dsv41_token_map.json',
                '--pool', '6400000', '--streams', '12', '--round-rows', '48', '--drafts', '1',
                '--graphs', '1', '--graph-cap', '128', '--side', '1', '--prefetch', '1', '--aio', '1']
    argv = ['docker', 'run', '-d', '--pull=never', '--name', f'deepseek1017-tp3-r{rank}',
            '--init', '--gpus', 'all', '--ipc=host', '--network=host', '--read-only', '--log-driver=none',
            '--ulimit', 'core=0', '--ulimit', 'memlock=-1', '--ulimit', 'stack=67108864',
            '--cap-add=IPC_LOCK', '--device=/dev/infiniband', '--tmpfs', '/tmp:rw,nosuid,nodev,size=256m']
    for key, dest in [('runtime','/runtime'), ('kit','/runtime/kit'), ('rank_cache','/rank-cache'),
                      ('engram','/engram'), ('serving','/serving'), ('spill','/spill')]:
        argv += ['--mount', f'type=bind,src={local[key]},dst={dest}' + ('' if key == 'spill' else ',readonly')]
    for key, value in sorted(env.items()):
        argv += ['-e', key + '=' + str(value)]
    return argv + ['--entrypoint', '/serving/' + program, image, *args]


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--rank', required=True, type=int, choices=(0, 1, 2))
    a = p.parse_args()
    print(shlex.join(command(json.loads(a.config.read_text()), a.rank)))
