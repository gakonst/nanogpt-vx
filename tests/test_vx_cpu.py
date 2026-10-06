"""Execute Vx-compiled functions against an independent C fmaf reference.

Requires vxc and LLVM 22 (VXC / LLVM_BIN overrides). No torch or GPU needed.
The host shim supplies only launch indices and the external fma hook.
"""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
VXC = Path(os.environ.get('VXC', str(Path.home()/'.vx/bin/vxc')))
LLVM = Path(os.environ.get('LLVM_BIN', '/opt/homebrew/opt/llvm@22/bin'))

SHIM = r'''
#include <stdint.h>
#include <math.h>
#include <string.h>
static int tid, bid;
int vx_tid(void) { return tid; }
int vx_bid(void) { return bid; }
int vx_bdim(void) { return 256; }
float vx_fma(float a, float b, float c) { return fmaf(a,b,c); }
extern void lerp_f32(const float*, float*, int, float);
extern void lerp_bf16(const uint16_t*, float*, int, float);
void launch(int bf, const void *src, float *dst, int n, float rate) {
    for (bid=0; bid<(n+255)/256; ++bid)
        for (tid=0; tid<256; ++tid) {
            if (bf) lerp_bf16(src,dst,n,rate);
            else lerp_f32(src,dst,n,rate);
        }
}
void reference(int bf, const void *src, float *dst, int n, float rate) {
    for (int i=0; i<n; ++i) {
        float e;
        if (bf) { uint32_t bits = ((const uint16_t*)src)[i]; bits <<= 16; memcpy(&e,&bits,4); }
        else e = ((const float*)src)[i];
        float delta = e - dst[i];
        dst[i] = fmaf(rate,delta,dst[i]);
    }
}
'''

class CompiledVxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not VXC.exists() or not (LLVM/'clang').exists():
            raise unittest.SkipTest('compiled differential test requires VXC and LLVM_BIN')
        subprocess.run([sys.executable, str(ROOT/'vx/build.py'), '--vxc', str(VXC), '--llvm-bin', str(LLVM)], check=True, capture_output=True)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        base = Path(cls.tmp.name)
        (base/'shim.c').write_text(SHIM)
        lib = base/('test.dylib' if sys.platform == 'darwin' else 'test.so')
        subprocess.run([str(LLVM/'clang'), '-shared', '-fPIC', '-O2', '-ffp-contract=off', str(ROOT/'vx/build/lerp.host.ll'), str(base/'shim.c'), '-lm', '-o', str(lib)], check=True, capture_output=True)
        cls.lib = C.CDLL(str(lib))
        for name in ('launch', 'reference'):
            getattr(cls.lib, name).argtypes = [C.c_int, C.c_void_p, C.POINTER(C.c_float), C.c_int, C.c_float]
            getattr(cls.lib, name).restype = None

    def test_compiled_differential_and_tail_guards(self):
        rng = random.Random(360)
        for bf in (False, True):
            for n in (0, 1, 31, 255, 256, 257, 4095, 4096, 4097):
                for rate in (0.0, 1.0, 0.5, 1/3, 0.001, -0.25, 1.25):
                    with self.subTest(bf16=bf, n=n, rate=rate):
                        # Offset pointers and sentinels catch prefix/tail writes.
                        src_values = [rng.uniform(-100,100) for _ in range(n)]
                        if bf:
                            src = (C.c_uint16 * max(1,n))(*[struct.unpack('I',struct.pack('f',v))[0] >> 16 for v in src_values])
                        else:
                            src = (C.c_float * max(1,n))(*src_values)
                        values = [123456.0] + [rng.uniform(-100,100) for _ in range(n)] + [-123456.0]*256
                        actual = (C.c_float * len(values))(*values)
                        expect = (C.c_float * len(values))(*values)
                        before = bytes(src)
                        self.lib.launch(bf, src, C.cast(C.byref(actual,4),C.POINTER(C.c_float)), n, rate)
                        self.lib.reference(bf, src, C.cast(C.byref(expect,4),C.POINTER(C.c_float)), n, rate)
                        self.assertEqual(bytes(actual),bytes(expect))
                        self.assertEqual(bytes(src),before)

    def test_emitted_ptx_and_manifest(self):
        out = ROOT/'vx/build'
        manifest = json.loads((out/'manifest.json').read_text())
        for path, key in ((ROOT/'vx/lerp.vx','source_sha256'),(out/'lerp.ptx','ptx_sha256')):
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),manifest[key])
        ptx = (out/'lerp.ptx').read_text()
        for instruction in ('.target sm_90', '.entry lerp_f32', '.entry lerp_bf16', 'sub.rn.f32', 'fma.rn.f32', 'cvt.f32.bf16', 'st.global.b32'):
            self.assertIn(instruction,ptx)
        self.assertNotIn('.extern .func',ptx)

if __name__ == '__main__':
    unittest.main()
