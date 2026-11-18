"""准备 M1 固定数据：AG News 与 DBpedia-14 的 train/val/test、replay buffer；按标签分层、去重、防跨划分泄漏。"""
import argparse
import random
from transformers import AutoTokenizer
from common import ROOT, read_json
from m1_common import atomic_json, digest, encode, load_config, roots, signature, stratified_buffer


def norm(s):
    return ' '.join(s.lower().split())


def rows_of(task, split):
    """Uniform row schema; keeps the official index as source_id."""
    texts = split['text'] if task == 'ag' else [s.strip() for s in split['content']]
    titles = [None] * len(texts) if task == 'ag' else split['title']
    return [{'text': t, 'title': h, 'label': y, 'source_id': i} for i, (t, h, y) in enumerate(zip(texts, titles, split['label']))]


def key(row):
    return norm((row['title'] or '') + ' || ' + row['text'])


def take_stratified(rows, sizes, n_labels, seed, banned):
    """Walk a seeded permutation; fill each split's per-class quota with unseen, non-banned texts."""
    quotas = {name: [n // n_labels + (k < n % n_labels) for k in range(n_labels)] for name, n in sizes.items()}
    out = {name: [] for name in sizes}
    seen = set(banned)
    order = list(range(len(rows))); random.Random(seed).shuffle(order)
    for i in order:
        r = rows[i]; k = key(r)
        if k in seen: continue
        for name in sizes:
            if quotas[name][r['label']]:
                quotas[name][r['label']] -= 1; out[name].append(r); seen.add(k); break
        if not any(any(q) for q in quotas.values()): break
    if any(any(q) for q in quotas.values()): raise RuntimeError('Not enough unique samples to fill stratified quotas')
    for name in out: random.Random(seed).shuffle(out[name])
    return out


def prepare(c, force=False):
    out = roots()['data']
    if (out / 'manifest.json').exists() and not force:
        verify(c); print('M1 data already prepared and verified', flush=True); return
    from datasets import load_dataset
    if digest(ROOT / c['base'] / 'model.safetensors') != c['source_model_sha256']:
        raise RuntimeError('Base model weight checksum mismatch')
    tok = AutoTokenizer.from_pretrained(ROOT / c['base'], local_files_only=True)
    s = c['split_sizes']; files = {}; stats = {}; fingerprints = {}
    for task, spec in c['tasks'].items():
        ds = load_dataset(spec['dataset'])
        n_labels = len(spec['labels'])
        if ds['train'].features['label'].num_classes != n_labels: raise RuntimeError(f'{task}: label count mismatch')
        fingerprints[task] = {k: v._fingerprint for k, v in ds.items()}
        test_rows = rows_of(task, ds['test'])
        test = take_stratified(test_rows, {'test': s['test']}, n_labels, c['seed'], set())['test']
        # Ban every official-test text from train/val, not only the sampled test items.
        banned = {key(r) for r in test_rows}
        tv = take_stratified(rows_of(task, ds['train']), {'train': s['train'], 'val': s['val']}, n_labels, c['seed'], banned)
        for split, rows in [('train', tv['train']), ('val', tv['val']), ('test', test)]:
            for r in rows: r['source_split'] = 'test' if split == 'test' else 'train'
            files[f'{task}_{split}'] = rows
            trunc = sum(encode(tok, task, r, spec['labels'], c['training']['max_length'])[1] for r in rows)
            stats[f'{task}_{split}'] = {'n': len(rows), 'truncated': trunc,
                                        'label_counts': [sum(r['label'] == k for r in rows) for k in range(n_labels)]}
        stats[task] = {'label_tokens_with_eos': {l: len(tok.encode(l + tok.eos_token, add_special_tokens=False)) for l in spec['labels']}}
    files['replay_buffer'] = stratified_buffer(files[c['replay']['source_task'] + '_train'], c['replay']['buffer_size'], c['seed'])
    hashes = {}
    for name, rows in files.items():
        atomic_json(out / f'{name}.json', rows); hashes[name] = digest(out / f'{name}.json')
    atomic_json(out / 'manifest.json', {'config_signature': signature(data_identity(c)), 'sha256': hashes,
                                        'fingerprints': fingerprints, 'stats': stats})
    verify(c); print('Prepared M1 data:', out, flush=True)


def data_identity(c):
    return {k: c[k] for k in ['seed', 'tasks', 'split_sizes', 'replay', 'base']} | {'max_length': c['training']['max_length']}


def verify(c):
    out = roots()['data']; m = read_json(out / 'manifest.json')
    if m['config_signature'] != signature(data_identity(c)): raise RuntimeError('M1 data config changed; prepare a new data version')
    for name, h in m['sha256'].items():
        if digest(out / f'{name}.json') != h: raise RuntimeError(f'M1 data changed: {name}')
    keys = {}
    for name in m['sha256']:
        if name == 'replay_buffer': continue
        task = name.split('_')[0]
        for r in read_json(out / f'{name}.json'):
            k = (task, key(r))
            if keys.setdefault(k, name) != name: raise RuntimeError(f'Leakage/duplicate between {keys[k]} and {name}')
    train_keys = {key(r) for r in read_json(out / f"{c['replay']['source_task']}_train.json")}
    if any(key(r) not in train_keys for r in read_json(out / 'replay_buffer.json')): raise RuntimeError('Replay buffer must come from training data only')
    return m


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', default='configs/m1.json')
    a = p.parse_args(); prepare(load_config(ROOT / a.config))
