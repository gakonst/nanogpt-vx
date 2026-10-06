#!/usr/bin/env python3
import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path
from evidence import hardware_ok, write_json

def inspect():
    import torch
    result = {'cuda_available': torch.cuda.is_available(), 'gpus': [],
              'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
              'torch_cuda': torch.version.cuda, 'environment': {k: os.environ.get(k) for k in
              ['CUDA_VISIBLE_DEVICES', 'NCCL_DEBUG', 'CUDA_HOME', 'TRAIN_SEED', 'NUM_SCHEDULED_ITERATIONS']}}
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        result['gpus'].append({'index': i, 'uuid': str(p.uuid), 'name': p.name, 'total_memory': p.total_memory,
                              'capability': list(torch.cuda.get_device_capability(i))})
    result['nvidia_smi'] = subprocess.check_output(['nvidia-smi', '-q'], text=True)
    result['topology'] = subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True)
    import ctypes
    ctypes.CDLL('libcudart.so.13')
    if torch.__version__.split('+')[0] != '2.10.0' or torch.version.cuda != '12.8':
        raise RuntimeError('Require pinned torch 2.10.0 cu128 and CUDA 13 attention runtime')
    if not hardware_ok(result):
        raise RuntimeError('Record runs require exactly eight visible H100 GPUs, CC 9.0 and >=75 GB each')
    if platform.system() != 'Linux':
        raise RuntimeError('Record training requires Linux')
    if os.environ.get('NUM_SCHEDULED_ITERATIONS', '1122') != '1122':
        raise RuntimeError('Refusing shortened training schedule')
    return result
if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); write_json(a.output, inspect())
