"""Stage2 configuration, provenance, and resource guards."""
import hashlib
import json
import os
import shutil
from pathlib import Path
from common import ROOT, read_json


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)


def load_config(path):
    c = read_json(path)
    t = c['common_training']
    if t['micro_batch_size'] * t['gradient_accumulation_steps'] != t['effective_batch_size']:
        raise ValueError('Effective batch size is inconsistent')
    return c


def output_roots(smoke=False):
    name = 'stage2-smoke' if smoke else 'stage2'
    return {k: ROOT / k / name for k in ['artifacts', 'results', 'reports']}


def directory_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file()) if Path(path).exists() else 0


def check_space(c, roots, reserve_bytes=0):
    free = shutil.disk_usage(ROOT).free
    used = sum(directory_bytes(p) for p in roots.values())
    minimum = c['storage']['minimum_free_gib'] * 2**30
    maximum = c['storage']['maximum_additional_gib'] * 2**30
    if free - reserve_bytes < minimum or used + reserve_bytes > maximum:
        raise RuntimeError(f'Stage2 disk limit: free={free/2**30:.2f} GiB, used={used/2**30:.2f} GiB, reserve={reserve_bytes/2**30:.2f} GiB')


def gsm_prompt(row):
    return row['question'] + '\nSolve step by step and end with: #### <answer>.'


def stratified_first(rows, key, per_group):
    groups = {}
    for row in rows:
        groups.setdefault(row[key], []).append(row)
    if any(len(pool) < per_group for pool in groups.values()):
        raise ValueError('Insufficient examples in evaluation stratum')
    return [row for pool in groups.values() for row in pool[:per_group]]


def math_encode(tokenizer, row, max_length):
    from common import chat
    prefix = tokenizer.encode(chat(tokenizer, gsm_prompt(row)), add_special_tokens=False)
    target = tokenizer.encode(row['answer'] + tokenizer.eos_token, add_special_tokens=False)
    if len(prefix) + len(target) > max_length:
        raise ValueError('Recovery sample exceeds maximum length; never truncate the gold answer')
    return {'input_ids': prefix + target, 'attention_mask': [1]*(len(prefix)+len(target)),
            'labels': [-100]*len(prefix) + target}


def prepared_manifest(c):
    manifest = read_json(ROOT / 'data/stage2/manifest.json')
    if manifest['config_signature'] != signature(c):
        raise RuntimeError('Stage2 configuration changed after data preparation; use a new configuration/data version')
    for relative, expected in manifest['hashes'].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f'Input changed: {relative}')
    return manifest
