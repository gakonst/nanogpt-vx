"""Synthetic audit fixtures only: these are not measured benchmark results."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evidence import sha256, write_json
from validate_record import validate, LOCK, REQUIRED
from compare import compare

class RecordTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self, name='vx', mode='vx', seconds=39.0, loss=3.27):
        d = self.root/name; d.mkdir()
        hw = {'cuda_available': True, 'torch': '2.10.0+cu128', 'torch_cuda': '12.8',
              'gpus': [{'index': i, 'uuid': f'SYNTHETIC-{i}', 'name': 'NVIDIA H100', 'capability': [9, 0], 'total_memory': 80_000_000_000} for i in range(8)]}
        write_json(d/'hardware.json', hw)
        (d/'pip-freeze.txt').write_text('synthetic fixture only\n')
        write_json(d/'source-hashes.json', {'upstream': LOCK['source_hashes'], 'integration': {k: 'a'*64 for k in ['vx/lerp.vx', 'vx/build/lerp.ptx', 'vx/build/manifest.json', 'nanogpt_vx/patch.py']}})
        write_json(d/'data-hashes.json', {f'fineweb_{kind}_{i:06d}.bin': {'sha256': 'b'*64, 'bytes': 10000} for kind,n in [('train',9), ('val',1)] for i in range(n)})
        log = ''.join(f'step:{i}/1194 train_time:100ms step_avg:1ms\n' for i in [25,1175,1194])
        log += f'step:1194/1194 val_loss:{loss} train_time:{seconds*1000}ms step_avg:1ms\npeak memory allocated: 100 MiB\n'
        (d/'train.log').write_text(log)
        for i in range(8):
            write_json(d/f'rank-{i}.json', {'rank': i, 'local_rank': i, 'device_uuid': f'SYNTHETIC-{i}', 'mode': mode, 'completed': True, 'started_unix': 1, 'ended_unix': 100, 'vx': {'gpu_launches': 1, 'installed': True, 'source_sha256': 'a'*64, 'ptx_sha256': 'a'*64}})
            (d/f'rank-{i}.stdout.log').write_text(log if i == 0 else '')
            (d/f'rank-{i}.stderr.log').write_text('')
        write_json(d/'manifest.json', {'schema': 1, 'mode': mode, 'seed': 42, 'upstream_commit': LOCK['commit'], 'total_steps': 1194, 'validation_tokens': 10485760, 'completed': True, 'returncode': 0})
        if mode == 'vx':
            write_json(d/'gpu-parity.json', {'required_cuda': True, 'returncode': 0})
            (d/'gpu-parity.log').write_text('SYNTHETIC GPU gate fixture; not executed\n')
        self.rehash(d)
        return d

    def rehash(self, d):
        m=json.loads((d/'manifest.json').read_text()); m['artifacts']={k:sha256(d/k) for k in REQUIRED + ['gpu-parity.json', 'gpu-parity.log'] if (d/k).exists()}; write_json(d/'manifest.json',m)

    def change(self,d,name,fn):
        x=json.loads((d/name).read_text()); fn(x); write_json(d/name,x); self.rehash(d)

    def test_single_run_never_record_candidate(self):
        result=validate(self.fixture()); self.assertTrue(result['evidence_valid'],result)
        self.assertTrue(result['meets_track1_thresholds']); self.assertFalse(result['local_vx_candidate']); self.assertFalse(result['official_record'])

    def test_matched_pair_candidate(self):
        r=compare(self.fixture('b','baseline',40),self.fixture()); self.assertTrue(r['local_vx_candidate'],r); self.assertFalse(r['official_record'])

    def test_threshold_is_exclusive_39_9(self):
        self.assertFalse(validate(self.fixture(seconds=39.9))['meets_track1_thresholds'])

    def test_must_beat_baseline(self):
        for seconds in [38,39]:
            with self.subTest(seconds=seconds):
                r=compare(self.fixture(f'b{seconds}','baseline',seconds),self.fixture(f'v{seconds}')); self.assertFalse(r['local_vx_candidate'])

    def test_failed_loss(self): self.assertFalse(validate(self.fixture(loss=3.2801))['meets_track1_thresholds'])

    def test_nan_measurement(self): self.assertFalse(validate(self.fixture(loss=float('nan')))['evidence_valid'])

    def test_hash_tampering(self):
        d=self.fixture(); (d/'train.log').write_text('forged'); self.assertFalse(validate(d)['evidence_valid'])

    def test_missing_rank_log(self):
        d=self.fixture(); (d/'rank-7.stderr.log').unlink(); self.rehash(d); self.assertFalse(validate(d)['evidence_valid'])

    def test_rank_must_execute_vx(self):
        for count in [0,True,-1]:
            d=self.fixture(str(count)); self.change(d,'rank-4.json',lambda r:r['vx'].update(gpu_launches=count)); self.assertFalse(validate(d)['evidence_valid'])

    def test_duplicate_gpu_rejected(self):
        d=self.fixture(); self.change(d,'hardware.json',lambda h:h['gpus'][1].update(uuid='SYNTHETIC-0')); self.assertFalse(validate(d)['evidence_valid'])

    def test_wrong_device_assignment(self):
        d=self.fixture(); self.change(d,'rank-2.json',lambda r:r.update(device_uuid='SYNTHETIC-0')); self.assertFalse(validate(d)['evidence_valid'])

    def test_modified_support_source(self):
        d=self.fixture(); self.change(d,'source-hashes.json',lambda r:r['upstream'].update({'track_1_short/config.py':'c'*64})); self.assertFalse(validate(d)['evidence_valid'])

    def test_reject_shortened_schedule(self):
        d=self.fixture(); self.change(d,'manifest.json',lambda r:r.update(total_steps=100)); self.assertFalse(validate(d)['evidence_valid'])

    def test_different_seed_and_data(self):
        b=self.fixture('b','baseline',40); v=self.fixture()
        self.change(v,'manifest.json',lambda r:r.update(seed=1)); self.assertFalse(compare(b,v)['matched_pair_valid'])
        self.change(v,'manifest.json',lambda r:r.update(seed=42))
        self.change(v,'data-hashes.json',lambda r:r['fineweb_val_000000.bin'].update(sha256='c'*64)); self.assertFalse(compare(b,v)['matched_pair_valid'])

    def test_missing_gpu_fails_without_crashing(self):
        d=self.fixture(); self.change(d,'hardware.json',lambda h:h['gpus'].pop()); self.assertFalse(validate(d)['evidence_valid'])

    def test_wrong_wheel_stack(self):
        d=self.fixture(); self.change(d,'hardware.json',lambda h:h.update(torch_cuda='13.0')); self.assertFalse(validate(d)['evidence_valid'])

    def test_missing_ptx_provenance(self):
        d=self.fixture(); self.change(d,'source-hashes.json',lambda h:h['integration'].pop('vx/build/lerp.ptx')); self.assertFalse(validate(d)['evidence_valid'])

    def test_missing_gpu_gate(self):
        d=self.fixture(); (d/'gpu-parity.json').unlink(); self.rehash(d); self.assertFalse(validate(d)['evidence_valid'])

    def test_failed_gpu_gate(self):
        d=self.fixture(); self.change(d,'gpu-parity.json',lambda r:r.update(returncode=1)); self.assertFalse(validate(d)['evidence_valid'])

    def test_wrong_vx_binary(self):
        d=self.fixture(); self.change(d,'rank-1.json',lambda r:r['vx'].update(ptx_sha256='c'*64)); self.assertFalse(validate(d)['evidence_valid'])

    def test_malformed_json(self):
        d=self.fixture(); (d/'manifest.json').write_text('{'); self.assertFalse(validate(d)['evidence_valid'])

if __name__ == '__main__': unittest.main()
