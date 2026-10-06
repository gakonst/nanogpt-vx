"""Compile real Vx source to LLVM IR, lower GPU ABI hooks, then LLVM NVPTX.

No PTX or arithmetic IR is synthesized. Only target metadata, calling convention
and four extern intrinsic names are changed. Intermediate artifacts are retained.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent

def run(args):
    result = subprocess.run([str(x) for x in args], text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f'{args}\n{result.stdout}\n{result.stderr}')
    return result.stdout

def build(vxc, llvm_bin, chip):
    version = run([vxc, '--version']).strip()
    if version != 'vxc 0.0.2':
        raise RuntimeError(f'Pinned vxc 0.0.2 required, got {version}')
    llvm_version = run([llvm_bin/'llc', '--version']).splitlines()[0]
    if not re.search(r'\b22\.', llvm_version):
        raise RuntimeError(f'LLVM 22 required, got {llvm_version}')
    out = ROOT / 'build'
    out.mkdir(exist_ok=True)
    source = ROOT / 'lerp.vx'
    ll = out / 'lerp.host.ll'
    ll.unlink(missing_ok=True)  # never accept stale output from a previous compiler run
    emitted = run([vxc, source, '--emit-llvm', '-o', ll])
    if not ll.exists():
        ll.write_text(emitted)
    ir = ll.read_text()
    if ir.lstrip().startswith('module {'):
        dialect = out / 'lerp.llvm.mlir'
        dialect.write_text(ir)
        run([llvm_bin/'mlir-translate', '--mlir-to-llvmir', dialect, '-o', ll])
        ir = ll.read_text()
    if 'define void @lerp_bf16' not in ir or 'define void @lerp_f32' not in ir:
        raise RuntimeError('Vx did not emit expected public kernel functions')
    gpu = re.sub(r'^target (?:datalayout|triple) = .*\n', '', ir, flags=re.M)
    gpu = 'target triple = "nvptx64-nvidia-cuda"\n' + gpu
    for name in ('lerp_bf16', 'lerp_f32'):
        gpu = gpu.replace(f'define void @{name}(', f'define ptx_kernel void @{name}(')
    for old, new in {'vx_tid':'llvm.nvvm.read.ptx.sreg.tid.x',
                     'vx_bid':'llvm.nvvm.read.ptx.sreg.ctaid.x',
                     'vx_bdim':'llvm.nvvm.read.ptx.sreg.ntid.x',
                     'vx_fma':'llvm.fma.f32'}.items():
        gpu = gpu.replace('@' + old + '(', '@' + new + '(')
    # Host CPU attributes must not constrain cross compilation.
    gpu = re.sub(r'"target-(?:cpu|features|abi)"="[^"]*"', '', gpu)
    (out / 'lerp.gpu.ll').write_text(gpu)
    run([llvm_bin / 'opt', '-passes=verify', '-disable-output', out / 'lerp.gpu.ll'])
    run([llvm_bin / 'llc', '-O3', '-march=nvptx64', f'-mcpu={chip}',
         '-mattr=+ptx80', out / 'lerp.gpu.ll', '-o', out / 'lerp.ptx'])
    ptx = (out / 'lerp.ptx').read_text()
    for required in ('.entry lerp_bf16', '.entry lerp_f32', 'fma.rn.f32', '%tid.x', 'st.'):
        if required not in ptx:
            raise RuntimeError(f'Missing expected compiled instruction: {required}')
    manifest = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'ptx_sha256': hashlib.sha256((out/'lerp.ptx').read_bytes()).hexdigest(),
                'chip': chip, 'vxc': run([vxc, '--version']).strip(),
                'llvm': run([llvm_bin/'llc', '--version']).splitlines()[0],
                'upstream_nanogpt': '4ea6b937337a4889b8cfe3f38a93d120048d8f71'}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--vxc', default=os.environ.get('VXC', str(Path.home()/'.vx/bin/vxc')))
    p.add_argument('--llvm-bin', type=Path, default=Path(os.environ.get('LLVM_BIN', '/opt/homebrew/opt/llvm@22/bin')))
    p.add_argument('--chip', default='sm_90')
    a = p.parse_args()
    build(a.vxc, a.llvm_bin, a.chip)
