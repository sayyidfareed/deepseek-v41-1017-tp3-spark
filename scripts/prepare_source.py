#!/usr/bin/env python3
"""Reconstruct the pinned source locally; never downloads, builds or serves."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def prepare(archive, output):
    manifest = json.loads((ROOT / 'provenance.json').read_text())
    if digest(archive) != manifest['base']['archive_sha256']:
        raise ValueError('Upstream archive SHA256 differs')
    if output.exists():
        raise ValueError('Use a new output directory; existing work is never replaced')
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        prefixes = {Path(m.name).parts[0] for m in members}
        if len(prefixes) != 1:
            raise ValueError('Expected one archive root')
        for m in members:
            p = Path(m.name)
            if p.is_absolute() or '..' in p.parts or not (m.isdir() or m.isfile()):
                raise ValueError('Unsafe upstream archive entry')
        output.mkdir(parents=True)
        for m in members:
            rel = Path(*Path(m.name).parts[1:])
            if rel == Path('.'):
                continue
            target = output / rel
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(m) as src, target.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
    for entry in manifest['files']:
        rel = Path(entry['path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Unsafe overlay path')
        source, target = ROOT / 'overlay' / rel, output / rel
        expected = entry['base_sha256']
        if (expected is None and target.exists()) or (expected is not None and (not target.exists() or digest(target) != expected)):
            raise ValueError('Unexpected upstream source: ' + str(rel))
        if digest(source) != entry['export_sha256']:
            raise ValueError('Overlay digest differs: ' + str(rel))
    for entry in manifest['files']:
        rel = Path(entry['path'])
        (output / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / 'overlay' / rel, output / rel)
    for name in ('CREDITS.md', 'provenance.json'):
        shutil.copyfile(ROOT / name, output / name)
    shutil.copytree(ROOT / 'licenses', output / 'integration-licenses')
    with (output / 'NOTICE').open('a') as f:
        f.write('\nIntegration changes: see CREDITS.md, provenance.json and integration-licenses/.\n')
    print('Source reconstructed and hashes verified; no build or runtime action.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    prepare(a.archive, a.out)
