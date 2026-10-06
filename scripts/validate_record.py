#!/usr/bin/env python3
"""Audit local evidence, never certify an official record or operator honesty."""
import argparse
import json
import math
from pathlib import Path
import re
import sys
from evidence import hardware_ok, sha256, write_json
ROOT = Path(__file__).resolve().parents[1]
LOCK = json.loads((ROOT / 'upstream.lock.json').read_text())
FINAL = re.compile(r'^step:(\d+)/(\d+) val_loss:(\S+) train_time:(\S+)ms step_avg:', re.M)
REQUIRED = ['hardware.json', 'pip-freeze.txt', 'source-hashes.json', 'data-hashes.json', 'train.log'] + [f'rank-{r}.json' for r in range(8)] + [f'rank-{r}.{stream}.log' for r in range(8) for stream in ['stdout', 'stderr']]

def validate(folder):
    folder = Path(folder); errors = []; metric = None; m = {}
    def require(ok, reason):
        if not ok: errors.append(reason)
    def read(name): return json.loads((folder / name).read_text())
    try:
        m = read('manifest.json')
        require(m.get('schema') == 1, 'unsupported schema')
        require(m.get('upstream_commit') == LOCK['commit'], 'wrong upstream revision')
        require(m.get('completed') is True and m.get('returncode') == 0, 'training process incomplete or failed')
        require(m.get('total_steps') == LOCK['total_steps'], 'shortened or changed step count')
        require(m.get('validation_tokens') == LOCK['validation_tokens'], 'changed validation token count')
        require(m.get('mode') in ['baseline', 'vx'], 'unknown execution mode')
        required = REQUIRED + (['gpu-parity.json', 'gpu-parity.log'] if m.get('mode') == 'vx' else [])
        for name in required:
            require((folder / name).is_file(), f'missing evidence: {name}')
            if (folder / name).is_file():
                require(m.get('artifacts', {}).get(name) == sha256(folder / name), f'hash mismatch: {name}')
        if errors: raise ValueError('evidence integrity failed')
        hw = read('hardware.json')
        require(hw.get('torch', '').split('+')[0] == '2.10.0' and hw.get('torch_cuda') == '12.8', 'wrong torch/CUDA wheel stack')
        require(hardware_ok(hw), 'requires eight H100 GPUs with CC 9.0 and >=75GB')
        source = read('source-hashes.json')
        require(source.get('upstream', {}).get(LOCK['entry']) == LOCK['entry_sha256'], 'pinned training source hash mismatch')
        require(source.get('upstream') == LOCK['source_hashes'], 'incomplete or changed upstream source hashes')
        integration = source.get('integration', {})
        require(bool(integration), 'missing integration source hashes')
        require(all(isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v) for v in integration.values()), 'invalid integration hashes')
        if m['mode'] == 'vx':
            gate = read('gpu-parity.json')
            require(gate.get('required_cuda') is True and gate.get('returncode') == 0, 'GPU correctness gate did not pass')
            require(all(name in integration for name in ['vx/lerp.vx', 'vx/build/lerp.ptx', 'vx/build/manifest.json', 'nanogpt_vx/patch.py']), 'missing compiled Vx provenance')
        data = read('data-hashes.json')
        require(sum(k.startswith('fineweb_train_') for k in data) == 9 and sum(k.startswith('fineweb_val_') for k in data) == 1, 'missing training/validation shards')
        require(all(isinstance(v.get('bytes'), int) and v['bytes'] > 1024 and re.fullmatch('[0-9a-f]{64}', v.get('sha256', '')) for v in data.values()), 'invalid data evidence')
        for rank in range(8):
            r = read(f'rank-{rank}.json')
            require(r.get('rank') == rank and r.get('mode') == m['mode'] and r.get('completed') is True, f'rank {rank} incomplete or mismatched')
            require(r.get('local_rank') == rank and r.get('device_uuid') == hw['gpus'][rank]['uuid'], f'rank {rank} GPU assignment mismatch')
            require(isinstance(r.get('started_unix'), (int, float)) and isinstance(r.get('ended_unix'), (int, float)) and r['ended_unix'] > r['started_unix'], f'rank {rank} invalid execution interval')
            if m['mode'] == 'vx':
                vx = r.get('vx', {})
                require(vx.get('installed') is True, f'rank {rank} Vx patch not installed')
                require(vx.get('source_sha256') == integration.get('vx/lerp.vx') and vx.get('ptx_sha256') == integration.get('vx/build/lerp.ptx'), f'rank {rank} Vx artifact mismatch')
                calls = vx.get('gpu_launches', 0)
                require(isinstance(calls, int) and not isinstance(calls, bool) and calls > 0, f'rank {rank} has no measured Vx GPU launches')
        log = (folder / 'train.log').read_text()
        require(log == (folder / 'rank-0.stdout.log').read_text(), 'training log differs from rank 0 stdout')
        lines = FINAL.findall(log)
        require(bool(lines), 'missing final validation line')
        if lines:
            step, total, loss, milliseconds = lines[-1]
            loss = float(loss); seconds = float(milliseconds) / 1000
            require(int(step) == int(total) == LOCK['total_steps'], 'training did not reach final full-schedule validation')
            require(math.isfinite(loss) and loss > 0 and math.isfinite(seconds) and seconds > 0, 'nonfinite or nonpositive measurements')
            if math.isfinite(loss) and math.isfinite(seconds): metric = {'validation_loss': loss, 'training_seconds': seconds}
            require('peak memory allocated:' in log, 'missing successful end-of-training marker')
            progress = [int(s) for s in re.findall(r'^step:(\d+)/1194 train_time:', log, re.M)]
            require(25 in progress and 1175 in progress and 1194 in progress, 'missing full training progress evidence')
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError) as e:
        errors.append(f'invalid evidence: {e}')
    thresholds = metric is not None and metric['validation_loss'] <= 3.28 and metric['training_seconds'] < 39.9
    valid = not errors
    return {'evidence_valid': valid, 'meets_track1_thresholds': bool(valid and thresholds),
            'local_vx_candidate': False, 'matched_baseline_required': True,
            'official_record': False, 'official_validation_required': True, 'metrics': metric, 'errors': errors,
            'trust_note': 'Local self-reported evidence only. Hashes detect edits, not deliberate fabrication. Independent PrimeIntellect validation and rule review are required.'}

def main():
    p = argparse.ArgumentParser(); p.add_argument('bundle', type=Path); a = p.parse_args()
    result = validate(a.bundle); write_json(a.bundle / 'validation.json', result)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['evidence_valid'] else 1
if __name__ == '__main__': sys.exit(main())
