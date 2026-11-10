# Retained/adapted TP3 integration source; original copyrights retained. See CREDITS.md and licenses/.
"""CPU-only verification of pinned HF files and receipts for rank-cache export."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from io_budget import Budget, Reader


def fingerprint(path):
    s=path.stat()
    return dict(size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,
                inode=s.st_ino,device=s.st_dev)


def safe_path(root,name):
    if Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('Unsafe source path')
    p=(root/name).resolve()
    if not p.is_relative_to(root.resolve()): raise ValueError('Source escapes root')
    return p


def verify(manifest,root,selected=None,*,read_mib_s=0):
    root=root.resolve(); records={}
    budget=Budget(read_mib_s)
    for item in manifest['siblings']:
        name=item['rfilename']
        if selected is not None and name not in selected: continue
        p=safe_path(root,name); before=fingerprint(p)
        if before['size']!=item['size']: raise ValueError('Size mismatch: '+name)
        lfs=item.get('lfs')
        if lfs:
            expected=lfs['sha256']; h=hashlib.sha256(); algorithm='sha256'
        else:
            expected=item['blobId']; h=hashlib.sha1(); algorithm='git-blob-sha1'
            h.update(f'blob {before["size"]}\0'.encode())
        with Reader(p.open('rb'),budget) as f:
            while data:=f.read(8<<20): h.update(data)
        if fingerprint(p)!=before: raise ValueError('File changed while hashing: '+name)
        if h.hexdigest()!=expected: raise ValueError('Digest mismatch: '+name)
        records[name]={'fingerprint':before,'digest':expected,'algorithm':algorithm}
        print('VERIFIED '+name,flush=True)
    if selected is not None and set(records)!=set(selected): raise ValueError('Missing manifest entries')
    return {'schema':'ds41f.source-verification.v1','repo':manifest['id'],
            'revision':manifest['sha'],'root':str(root),'files':records}


def check_receipt(receipt,root,source,needed):
    """Fast metadata guard for trusted immutable inputs, NOT a content rehash.

    Equal stat metadata cannot prove unchanged bytes. Keep source directories
    immutable after verify(); re-run verify after any intended file mutation.
    """
    if receipt.get('schema')!='ds41f.source-verification.v1': raise ValueError('Unknown receipt')
    if receipt['repo']!=source['repo'] or receipt['revision']!=source['revision']:
        raise ValueError('Receipt revision mismatch')
    if Path(receipt['root']).resolve()!=root.resolve(): raise ValueError('Receipt root mismatch')
    for name in needed:
        record=receipt['files'].get(name)
        if record is None: raise ValueError('Unverified shard: '+name)
        if record['algorithm']!='sha256': raise ValueError('Weight shard requires SHA256')
        if fingerprint(safe_path(root,name))!=record['fingerprint']:
            raise ValueError('Source changed since verification: '+name)


def main():
    p=argparse.ArgumentParser()
    for n in ('manifest','root','out'): p.add_argument('--'+n,required=True)
    p.add_argument('--weights-only',action='store_true')
    p.add_argument('--shards',help='JSON array of selected shard names; mutually exclusive with --weights-only')
    p.add_argument('--read-mib-s',type=float,default=0,help='Shared read limit, 0=unlimited')
    a=p.parse_args(); raw=Path(a.manifest).read_bytes(); manifest=json.loads(raw)
    selected={x['rfilename'] for x in manifest['siblings'] if x['rfilename'].endswith('.safetensors')} if a.weights_only else None
    if a.shards:
        if a.weights_only: p.error('Choose either --shards or --weights-only')
        items=json.loads(Path(a.shards).read_bytes())
        if (not isinstance(items,list) or not items or
                any(not isinstance(x,str) or not x.endswith('.safetensors') for x in items)):
            p.error('--shards requires a nonempty JSON array of safetensors names')
        selected=set(items)
        if len(selected)!=len(items): p.error('Duplicate shard names')
    if Path(a.out).exists(): raise FileExistsError(a.out)
    result=verify(manifest,Path(a.root),selected,read_mib_s=a.read_mib_s)
    result['manifest_sha256']=hashlib.sha256(raw).hexdigest()
    with Path(a.out).open('x') as f:
        json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())


if __name__=='__main__': main()
