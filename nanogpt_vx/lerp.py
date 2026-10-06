"""CUDA-driver launch of Vx-generated PTX on PyTorch's current stream."""
import ctypes as C
import hashlib
import json
from pathlib import Path
import threading
import torch

_CACHE = {}
_LOCK = threading.Lock()
_LAUNCHES = 0


def launch_statistics():
    return {"gpu_launches": _LAUNCHES}


def _check(code, operation):
    if code:
        raise RuntimeError(f'Vx CUDA driver {operation} failed (CUresult={code})')


def _load(device):
    with _LOCK:
        if device in _CACHE:
            return _CACHE[device]
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Call nanogpt_vx.warmup(device) before CUDA graph capture')
        root = Path(__file__).resolve().parent.parent / 'vx'
        ptx = (root/'build/lerp.ptx').read_bytes()
        manifest = json.loads((root/'build/manifest.json').read_text())
        if hashlib.sha256(ptx).hexdigest() != manifest['ptx_sha256']:
            raise RuntimeError('Vx PTX artifact hash mismatch; rebuild vx/build.py')
        if hashlib.sha256((root/'lerp.vx').read_bytes()).hexdigest() != manifest['source_sha256']:
            raise RuntimeError('Vx source changed since compilation; rebuild vx/build.py')
        major, minor = torch.cuda.get_device_capability(device)
        if major*10+minor < int(manifest['chip'].removeprefix('sm_')):
            raise RuntimeError('Device is older than the compiled Vx PTX target')
        driver = C.CDLL('libcuda.so.1')
        driver.cuModuleLoadData.argtypes = [C.POINTER(C.c_void_p), C.c_void_p]
        driver.cuModuleGetFunction.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p]
        driver.cuLaunchKernel.argtypes = [C.c_void_p] + [C.c_uint]*7 + [C.c_void_p, C.POINTER(C.c_void_p), C.c_void_p]
        module = C.c_void_p()
        _check(driver.cuModuleLoadData(C.byref(module), C.c_char_p(ptx)), 'cuModuleLoadData')
        functions = {}
        for dtype, name in ((torch.bfloat16, b'lerp_bf16'), (torch.float32, b'lerp_f32')):
            fn = C.c_void_p()
            _check(driver.cuModuleGetFunction(C.byref(fn), module, name), 'cuModuleGetFunction')
            functions[dtype] = fn
        # Keep module alive for the process lifetime, including captured graphs.
        _CACHE[device] = driver, module, functions
        return _CACHE[device]


def warmup(device=None):
    """Load PTX outside graph capture, once for each CUDA device used."""
    device = torch.cuda.current_device() if device is None else torch.device(device).index if not isinstance(device, int) else device
    if device is None:
        device = torch.cuda.current_device()
    with torch.cuda.device(device):
        torch.empty(0, device=device)  # establish the torch-owned CUDA context
        _load(device)


def _validate(dst, src):
    if not dst.is_cuda or not src.is_cuda or dst.device != src.device:
        raise ValueError('Vx lerp requires tensors on the same CUDA device')
    if dst.dtype != torch.float32 or src.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError('Vx lerp supports float32 dst and bfloat16/float32 src only')
    if not dst.is_contiguous() or not src.is_contiguous() or dst.numel() != src.numel():
        raise ValueError('Vx lerp requires contiguous tensors with equal element counts')
    if dst.numel() > 2147483392:
        raise ValueError('Vx lerp index exceeds signed i32 launch range')
    if dst.requires_grad or src.requires_grad:
        raise ValueError('Vx tail averaging is non-differentiable; detach input tensors')


@torch.library.custom_op('nanogpt_vx::lerp_upcast_', mutates_args=('dst',))
def lerp_upcast_(dst: torch.Tensor, src: torch.Tensor, rate: float) -> None:
    """In-place f32 fma(rate, float(src)-dst, dst); no fallback implementation."""
    global _LAUNCHES
    _validate(dst, src)
    if dst.numel() == 0:
        return
    da, sa = dst.data_ptr(), src.data_ptr()
    if da < sa + src.numel()*src.element_size() and sa < da + dst.numel()*dst.element_size():
        raise ValueError('Vx lerp requires non-overlapping source/destination storage')
    with torch.cuda.device(dst.device):
        driver, module, functions = _load(dst.device.index)
        args = [C.c_void_p(sa), C.c_void_p(da), C.c_int32(dst.numel()), C.c_float(rate)]
        params = (C.c_void_p * len(args))(*(C.cast(C.pointer(a), C.c_void_p) for a in args))
        stream = torch.cuda.current_stream(dst.device)
        _check(driver.cuLaunchKernel(functions[src.dtype], (dst.numel()+255)//256, 1, 1,
                                    256, 1, 1, 0, C.c_void_p(stream.cuda_stream), params, None),
               'cuLaunchKernel')
        _LAUNCHES += 1


@lerp_upcast_.register_fake
def _(dst, src, rate):
    _validate(dst, src)
    return None
