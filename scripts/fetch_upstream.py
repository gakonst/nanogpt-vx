#!/usr/bin/env python3
"""Fetch exactly the reviewed upstream commit; never overwrite a changed checkout."""
import argparse
import json
from pathlib import Path
import tempfile
import urllib.request
ROOT = Path(__file__).resolve().parents[1]
LOCK = json.loads((ROOT / 'upstream.lock.json').read_text())
def verify(path):
    from evidence import sha256
    path = Path(path)
    for name, expected in LOCK['source_hashes'].items():
        if not (path / name).is_file() or sha256(path / name) != expected:
            raise RuntimeError(f'pinned upstream source changed or missing: {name}')
    actual = {str(f.relative_to(path)) for f in path.rglob('*.py') if '.git' not in f.parts}
    expected = {name for name in LOCK['source_hashes'] if name.endswith('.py')}
    if actual != expected:
        raise RuntimeError('unexpected Python source in upstream checkout')
def main():
    p = argparse.ArgumentParser(); p.add_argument('--destination', type=Path, default=ROOT / 'vendor/modded-nanogpt')
    a = p.parse_args(); path = a.destination.resolve()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
            stage = Path(tmp) / 'source'
            stage.mkdir()
            base = 'https://raw.githubusercontent.com/KellerJordan/modded-nanogpt/' + LOCK['commit'] + '/'
            for name in LOCK['source_hashes']:
                dst = stage / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                with urllib.request.urlopen(base + name, timeout=60) as response:
                    dst.write_bytes(response.read())
            verify(stage)
            stage.rename(path)
    verify(path)
    print(path)
if __name__ == '__main__': main()
