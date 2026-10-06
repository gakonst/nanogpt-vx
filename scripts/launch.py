#!/usr/bin/env python3
"""torchrun entry: install Vx before executing the unmodified pinned trainer."""
import argparse
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
from pathlib import Path
import runpy
import sys
import time
from evidence import write_json
from fetch_upstream import verify
ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(); p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--mode', choices=['baseline', 'vx'], required=True)
    a = p.parse_args(); source = a.upstream.resolve(); output = a.output.resolve()
    rank = int(os.environ['RANK']); local_rank = int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE']) != 8 or not 0 <= rank < 8:
        raise RuntimeError('exactly eight distributed ranks required')
    verify(source)
    import torch
    torch.cuda.set_device(local_rank)
    props = torch.cuda.get_device_properties(local_rank)
    if 'H100' not in props.name or torch.cuda.get_device_capability(local_rank) != (9, 0):
        raise RuntimeError('H100 required on every rank')
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(source))
    os.chdir(source)
    stats = lambda: {}
    if a.mode == 'vx':
        from nanogpt_vx.patch import install, statistics
        install()
        stats = statistics
    record = {'rank': rank, 'mode': a.mode, 'completed': False, 'started_unix': time.time(), 'local_rank': local_rank, 'device_uuid': str(props.uuid), 'device_name': props.name}
    try:
        sys.argv = [str(source / 'train_gpt.py')]
        runpy.run_path(sys.argv[0], run_name='__main__')
        torch.cuda.synchronize()
        record['completed'] = True
    finally:
        record['ended_unix'] = time.time()
        record['vx'] = stats()
        write_json(output / f'rank-{rank}.json', record)
if __name__ == '__main__': main()
