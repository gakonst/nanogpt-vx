"""Independent upstream Triton parity, current-stream and CUDA-graph checks.

VX_REQUIRE_CUDA=1 turns missing hardware/dependencies into a failure. No GPU test
is satisfied by the CPU shim. Run these tests on the actual benchmark host.
"""
import importlib.util
import os
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]


class GPUParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch is None or not torch.cuda.is_available():
            if os.environ.get('VX_REQUIRE_CUDA') == '1':
                raise RuntimeError('CUDA hardware and PyTorch required for the GPU correctness gate')
            raise unittest.SkipTest('GPU parity not exercised: CUDA/PyTorch unavailable')
        from nanogpt_vx import lerp_upcast_, warmup
        cls.vx = staticmethod(lerp_upcast_)
        warmup()
        path = ROOT/'vendor/modded-nanogpt/track_1_short/perf/kernels/lerp.py'
        spec = importlib.util.spec_from_file_location('independent_upstream_lerp', path)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        cls.original = staticmethod(original.lerp_upcast_)

    def test_bitwise_upstream_parity_and_guard_regions(self):
        gen = torch.Generator(device='cuda').manual_seed(360)
        for dtype in (torch.float32, torch.bfloat16):
            for n in (0, 1, 255, 256, 257, 4095, 4096, 4097, 65536, 65537, 1048576):
                for rate in (0.0, 1.0, 0.5, 4.0/53.0, 2.0/75.0):
                    with self.subTest(dtype=dtype, n=n, rate=rate):
                        src = torch.randn(n, device='cuda', dtype=dtype, generator=gen)
                        saved = src.clone()
                        storage = torch.randn(n+2, device='cuda', generator=gen)
                        expected = storage.clone()
                        self.vx(storage[1:-1], src, rate)
                        if n:
                            self.original(expected[1:-1], src, rate)
                        torch.cuda.synchronize()
                        self.assertTrue(torch.equal(storage.view(torch.int32), expected.view(torch.int32)))
                        self.assertTrue(torch.equal(src, saved))

    def test_nondefault_stream(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            src = torch.ones(65537, dtype=torch.bfloat16, device='cuda')
            actual = torch.zeros(65537, device='cuda')
            self.vx(actual, src, 0.5)
            # Consuming on the same stream must see this launch's result.
            answer = actual.sum()
        stream.synchronize()
        self.assertEqual(answer.item(), 65537*0.5)

    def test_graph_replay(self):
        src = torch.ones(4097, dtype=torch.bfloat16, device='cuda')
        actual = torch.zeros(4097, device='cuda')
        self.vx(actual, src, 0.5)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.vx(actual, src, 0.5)
        actual.zero_()
        graph.replay()
        self.assertTrue(torch.equal(actual, torch.full_like(actual, 0.5)))
        graph.replay()
        self.assertTrue(torch.equal(actual, torch.full_like(actual, 0.75)))

    def test_torch_compile_mutation(self):
        def apply(dst, src):
            self.vx(dst, src, 0.5)
            return dst + 1.0
        fn = torch.compile(apply)
        src = torch.ones(257, device='cuda', dtype=torch.bfloat16)
        dst = torch.zeros(257, device='cuda')
        result = fn(dst, src)
        self.assertTrue(torch.equal(dst, torch.full_like(dst, 0.5)))
        self.assertTrue(torch.equal(result, torch.full_like(result, 1.5)))

    def test_invalid_storage_contracts(self):
        x = torch.ones(512, device='cuda')
        with self.assertRaises(ValueError):
            self.vx(x, x, 0.5)
        with self.assertRaises(ValueError):
            self.vx(x[::2], x[:256].clone(), 0.5)
        with self.assertRaises(TypeError):
            self.vx(x.bfloat16(), x, 0.5)
        with self.assertRaises(ValueError):
            self.vx(x.cpu(), x, 0.5)

if __name__ == '__main__':
    unittest.main()
