# Retained/adapted TP3 integration source; original copyrights retained. See CREDITS.md and licenses/.
"""Offline SM121 compilation of explicit launch specifications; never allocates GPU tensors.

Input cases name a source module/function, exact constexprs, tensor/scalar types
and compiler options. No shape guessing: coverage must be audited separately.
Produces the native aot.json schema and cubins, plus build provenance.
"""
import argparse
import hashlib
import importlib
import json
import re
from pathlib import Path
import struct


TYPES={'bfloat16':'*bf16','float16':'*fp16','float32':'*fp32','int64':'*i64',
       'int32':'*i32','int16':'*i16','uint8':'*u8','int8':'*i8','bool':'*i1',
       'float8_e4m3fn':'*fp8e4nv'}


def check_abi(ptx,params):
    entry=re.search(r'\.entry\s+\S+\s*\((.*?)\)\s*.*?\{',ptx,re.S)
    if entry is None: raise ValueError('Missing PTX entry')
    fields=[x.strip() for x in entry[1].split(',') if x.strip()]
    widths=[64 if p['type'].startswith('*') or p['type']=='u64' else 32 for p in params]+[64,64]
    if len(fields)!=len(widths): raise ValueError('Native ABI requires two Triton scratch pointers')
    for field,width in zip(fields,widths):
        m=re.search(r'\.(?:u|s|b|f)(\d+)\s+',field)
        if m is None or int(m[1])!=width: raise ValueError('Native ABI parameter width mismatch')


def constant(x):
    if isinstance(x,(int,bool)): return {'int':int(x)}
    if isinstance(x,float): return {'f32':struct.unpack('<I',struct.pack('<f',x))[0]}
    raise ValueError('Unsupported constexpr type')


def compile_case(case,out):
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    module=importlib.import_module(case['module']); fn=getattr(module,case['function'])
    consts=case['consts']; signature={}; attrs={}; params=[]
    runtime=case['params']
    if set(fn.arg_names)!=set(runtime)|set(consts): raise ValueError('Incomplete or extra function arguments')
    if set(runtime)&set(consts): raise ValueError('Runtime/constexpr overlap')
    for name in fn.arg_names:
        if name in consts:
            signature[name]='constexpr';continue
        p=runtime[name];ty=p['type'];signature[name]=ty
        aligned=p.get('div16',False)
        if aligned: attrs[(fn.arg_names.index(name),)]=[['tt.divisibility',16]]
        # All runtime scalars remain runtime (no equal-to-one specialization).
        params.append({'name':name,'type':ty,'div16':aligned,'nospec':True})
    source=ASTSource(fn=fn,signature=signature,constexprs=consts,attrs=attrs)
    compiled=triton.compile(source,target=GPUTarget('cuda',121,32),options=case['options'])
    md=compiled.metadata
    check_abi(compiled.asm['ptx'],params)
    data=compiled.asm['cubin']; digest=hashlib.sha256(data).hexdigest()
    path=out/'cubins'/(digest+'.cubin');path.write_bytes(data)
    return {'fn':fn.__name__,'hash':digest,'name':md.name,'num_warps':md.num_warps,
        'num_ctas':getattr(md,'num_ctas',1),'shared':md.shared,
        'global_scratch':getattr(md,'global_scratch_size',0),
        'global_align':getattr(md,'global_scratch_align',1),
        'profile_scratch':getattr(md,'profile_scratch_size',0),
        'pdl':getattr(md,'launch_pdl',False),'params':params,
        'consts':{k:constant(v) for k,v in consts.items()}}, {
        'module_file':module.__file__,'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        'compiler_hash':compiled.hash,'ptx_sha256':hashlib.sha256(compiled.asm['ptx'].encode()).hexdigest(),
        'source_dependencies':[{'file':m.__file__,'sha256':hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()}
                               for m in getattr(module,'SOURCE_DEPENDENCIES',())]}


def main():
    import triton
    p=argparse.ArgumentParser();p.add_argument('--cases',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();raw=Path(a.cases).read_bytes();cases=json.loads(raw)
    out=Path(a.out);out.mkdir(exist_ok=False);(out/'cubins').mkdir()
    kernels=[];provenance=[]
    for case in cases:
        kernel,receipt=compile_case(case,out);kernels.append(kernel);provenance.append(receipt)
        print('COMPILED '+case['function']+' '+kernel['hash'],flush=True)
    (out/'aot.json').write_text(json.dumps({'generator':'compile_triton.py','kernels':kernels},indent=2))
    (out/'provenance.json').write_text(json.dumps({'target':'cuda:121:32','triton':triton.__version__,
        'cases_sha256':hashlib.sha256(raw).hexdigest(),'status':'COMPILED_NOT_GPU_QUALIFIED',
        'sources':provenance},indent=2))


if __name__=='__main__':main()
