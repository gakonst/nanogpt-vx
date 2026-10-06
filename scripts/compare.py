#!/usr/bin/env python3
"""Audit a matched run pair; only a faster Vx run below 39.9s is a local candidate."""
import argparse
import json
from pathlib import Path
from validate_record import validate

def compare(baseline, vx):
    baseline, vx = Path(baseline), Path(vx)
    b, v = validate(baseline), validate(vx)
    errors = []
    def read(folder, name): return json.loads((folder / name).read_text())
    if not b['evidence_valid'] or not v['evidence_valid']:
        errors.append('Both bundles must pass evidence checks')
    else:
        mb, mv = read(baseline, 'manifest.json'), read(vx, 'manifest.json')
        if mb['mode'] != 'baseline' or mv['mode'] != 'vx' or mb.get('seed') != mv.get('seed') or not isinstance(mb.get('seed'), int):
            errors.append('Require baseline/Vx pair with the same explicit seed')
        for name in ['hardware.json', 'data-hashes.json', 'source-hashes.json']:
            x, y = read(baseline, name), read(vx, name)
            if name == 'hardware.json':
                keys = ['gpus', 'python', 'platform', 'torch', 'torch_cuda', 'environment', 'topology']
                x, y = ({k: z.get(k) for k in keys} for z in (x, y))
            if x != y: errors.append(f'Mismatched {name}')
        if (baseline/'pip-freeze.txt').read_text() != (vx/'pip-freeze.txt').read_text():
            errors.append('Mismatched Python packages')
        if b['metrics']['validation_loss'] > 3.28:
            errors.append('Baseline failed target validation loss')
    matched = not errors
    faster = matched and v['metrics']['training_seconds'] < b['metrics']['training_seconds']
    return {'matched_pair_valid': matched, 'baseline': b, 'vx': v,
            'speedup': b['metrics']['training_seconds'] / v['metrics']['training_seconds'] if matched else None,
            'beats_matched_baseline': bool(faster),
            'local_vx_candidate': bool(faster and v['meets_track1_thresholds']),
            'official_record': False, 'official_validation_required': True, 'errors': errors}

def main():
    p = argparse.ArgumentParser(); p.add_argument('baseline', type=Path); p.add_argument('vx', type=Path)
    a = p.parse_args(); result = compare(a.baseline, a.vx)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['matched_pair_valid'] else 1
if __name__ == '__main__': raise SystemExit(main())
