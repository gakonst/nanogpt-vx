"""Install the Vx replacement before the pinned trainer imports tail_average."""
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
_INSTALLED = False


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    if 'track_1_short.tail_average' in sys.modules:
        raise RuntimeError('Install Vx before importing the trainer/tail_average')
    from .lerp import lerp_upcast_, warmup
    # Module loading/JIT compilation stays outside the measured training loop.
    warmup()
    import track_1_short.perf.kernels.lerp as original
    original.lerp_upcast_ = lerp_upcast_
    _INSTALLED = True


def statistics():
    from .lerp import launch_statistics
    stats = launch_statistics()
    stats['installed'] = _INSTALLED
    stats['source_sha256'] = hashlib.sha256((ROOT/'vx/lerp.vx').read_bytes()).hexdigest()
    stats['ptx_sha256'] = hashlib.sha256((ROOT/'vx/build/lerp.ptx').read_bytes()).hexdigest()
    stats['scope'] = 'tail_weight_averaging_only'
    stats['counter_semantics'] = 'successful host cuLaunchKernel submissions; graph replays not counted'
    return stats
