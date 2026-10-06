#!/usr/bin/env python3
"""Capture a full baseline/Vx run; no reduced-work or CPU record mode exists."""
import argparse
import datetime
import json
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
from pathlib import Path
import subprocess
import sys
import time
from evidence import sha256, write_json
from fetch_upstream import ROOT, LOCK, verify
from preflight import inspect

def main():
    p = argparse.ArgumentParser(); p.add_argument('--upstream', type=Path, default=ROOT / 'vendor/modded-nanogpt')
    p.add_argument('--output', type=Path, required=True); p.add_argument('--mode', choices=['baseline', 'vx'], required=True)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args(); source = a.upstream.resolve(); output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    verify(source)
    os.environ['TRAIN_SEED'] = str(a.seed)
    os.environ['NUM_SCHEDULED_ITERATIONS'] = '1122'
    os.environ['DATA_PATH'] = str(source)
    write_json(output / 'hardware.json', inspect())
    (output / 'pip-freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze', '--all'], text=True))
    source_hashes = {f: sha256(source / f) for f in LOCK['source_hashes']}
    # Hash all local integration source and Vx source, including code not tracked in git yet.
    own = [f for folder in ['scripts', 'nanogpt_vx', 'vx'] for f in (ROOT / folder).rglob('*')
           if f.is_file() and not any(x in f.parts for x in ['.git', '__pycache__', '.venv'])
           and f.suffix in ['.py', '.cpp', '.h', '.hpp', '.td', '.vx', '.toml', '.sh', '.ptx', '.ll', '.mlir', '.json']]
    own += [ROOT / 'toolchain.lock.json', ROOT / 'upstream.lock.json', ROOT / 'requirements.txt', ROOT / 'configs/track1.json']
    write_json(output / 'source-hashes.json', {'upstream': source_hashes,
               'integration': {str(f.relative_to(ROOT)): sha256(f) for f in own}})
    if a.mode == 'vx':
        gate_env = dict(os.environ, VX_REQUIRE_CUDA='1')
        gate = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
                               '-p', 'test_vx_gpu.py', '-v'], cwd=ROOT, env=gate_env,
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (output/'gpu-parity.log').write_text(gate.stdout)
        write_json(output/'gpu-parity.json', {'required_cuda': True, 'returncode': gate.returncode})
        if gate.returncode != 0:
            raise RuntimeError('GPU parity gate failed; see gpu-parity.log')
    data = sorted((source / 'data/fineweb10B').glob('fineweb_*.bin'))
    train = [f for f in data if 'train_' in f.name]; val = [f for f in data if 'val_' in f.name]
    if len(train) != 9 or len(val) != 1:
        raise RuntimeError('download the nine upstream training shards and validation data first')
    write_json(output / 'data-hashes.json', {f.name: {'sha256': sha256(f), 'bytes': f.stat().st_size} for f in data})
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
           '--log-dir', str(output / 'rank-logs'), '--redirects', '3', '--tee', '0',
           str(ROOT / 'scripts/launch.py'), '--upstream', str(source), '--output', str(output), '--mode', a.mode]
    manifest = {'schema': 1, 'upstream_commit': LOCK['commit'], 'mode': a.mode, 'seed': a.seed,
                'total_steps': LOCK['total_steps'], 'validation_tokens': LOCK['validation_tokens'],
                'command': cmd, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'completed': False, 'official_validation': False}
    write_json(output / 'manifest.json', manifest)
    begin = time.perf_counter()
    with (output / 'launcher.log').open('w') as log:
        child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in child.stdout:
            log.write(line); log.flush(); print(line, end='', flush=True)
        rc = child.wait()
    manifest.update(returncode=rc, completed=rc == 0, process_wall_seconds=time.perf_counter() - begin)
    evidence = ['hardware.json', 'pip-freeze.txt', 'source-hashes.json', 'data-hashes.json', 'train.log', 'launcher.log']
    evidence += [f'rank-{r}.json' for r in range(8)]
    if a.mode == 'vx':
        evidence += ['gpu-parity.json', 'gpu-parity.log']
    for rank in range(8):
        for stream in ['stdout', 'stderr']:
            logs = list((output / 'rank-logs').glob(f'*/attempt_0/{rank}/{stream}.log'))
            if len(logs) == 1:
                (output / f'rank-{rank}.{stream}.log').write_bytes(logs[0].read_bytes())
                evidence.append(f'rank-{rank}.{stream}.log')
    if (output / 'rank-0.stdout.log').exists():
        (output / 'train.log').write_bytes((output / 'rank-0.stdout.log').read_bytes())
    manifest['artifacts'] = {f: sha256(output / f) for f in evidence if (output / f).is_file()}
    write_json(output / 'manifest.json', manifest)
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/validate_record.py'), str(output)])
    sys.exit(rc or result.returncode)
if __name__ == '__main__': main()
