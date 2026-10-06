"""Evidence utilities. Hashes detect accidental changes, not malicious forgery."""
import hashlib
import json
from pathlib import Path

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def hardware_ok(hw):
    gpus = hw.get('gpus', [])
    return (hw.get('cuda_available') is True and len(gpus) == 8
            and len({g.get('uuid') for g in gpus}) == 8
            and all(isinstance(g.get('uuid'), str) and g['uuid'] and 'H100' in g.get('name', '') and g.get('capability') == [9, 0]
                    and g.get('total_memory', 0) >= 75_000_000_000 for g in gpus))
