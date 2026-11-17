"""准备 Stage2 的固定子集和恢复数据；不会修改 Stage1。"""
import argparse
import random
from common import ROOT, read_json
from stage2_common import atomic_json, digest, signature, load_config, stratified_first, math_encode, prepared_manifest


def prepare(c):
    out = ROOT / 'data/stage2'
    if (out / 'manifest.json').exists():
        prepared_manifest(c)
        print('Stage2 data already prepared and verified', flush=True)
        return
    from datasets import load_dataset
    from transformers import AutoTokenizer
    stage1 = ROOT / 'data/stage1'
    original = read_json(stage1 / 'manifest.json')
    for name, expected in original['sha256'].items():
        if digest(stage1 / (name + '.json')) != expected:
            raise RuntimeError(f'Stage1 data checksum mismatch: {name}')
    tokenizer = AutoTokenizer.from_pretrained(ROOT / c['references']['base'], local_files_only=True)
    weights = ROOT / c['references']['base'] / 'model.safetensors'
    if digest(weights) != c['references']['source_model_sha256']:
        raise RuntimeError('Base model weight checksum mismatch')
    collapsed = ROOT / c['references']['collapsed'] / 'model.safetensors'
    if not collapsed.exists():
        raise RuntimeError('Stage1 checkpoint-50 is required')
    full = {name: read_json(stage1 / (name + '.json')) for name in ['ag_test', 'gsm_test', 'mmlu_test', 'open']}
    e = c['evaluation']
    quick = {'ag_test': stratified_first(full['ag_test'], 'label', e['quick_ag_per_class']),
             'gsm_test': full['gsm_test'][:e['quick']['gsm']],
             'mmlu_test': stratified_first(full['mmlu_test'], 'subject', e['quick_mmlu_per_subject']),
             'open': full['open'][:e['quick']['open']]}
    norm = lambda s: ' '.join(s.lower().split())
    r = c['recovery']
    gsm = load_dataset(r['source_dataset'], r['source_subset'], split=r['source_split'])
    test_gsm = load_dataset(r['source_dataset'], r['source_subset'], split='test')
    seen = {norm(x['question']) for x in test_gsm}
    eligible = []
    order = list(range(len(gsm))); random.Random(c['seed']).shuffle(order)
    for i in order:
        row = dict(gsm[i], source_id=i)
        text = norm(row['question'])
        if text in seen:
            continue
        try:
            math_encode(tokenizer, row, r['max_length'])
        except ValueError:
            continue
        seen.add(text); eligible.append(row)
        if len(eligible) == r['train_pool_size'] + r['dev_size']:
            break
    if len(eligible) < r['train_pool_size'] + r['dev_size']:
        raise RuntimeError('Not enough unique length-safe GSM training samples')
    files = {'eval_full': full, 'eval_quick': quick,
             'gsm_dev': eligible[r['train_pool_size']:],
             **{f'gsm_train_{n}': eligible[:n] for n in r['nested_train_sizes']}}
    ag_train = read_json(stage1 / 'train.json')
    excluded = {norm(x['text']) for x in ag_train + full['ag_test']}
    ag = load_dataset('fancyzhx/ag_news', split='train')
    order = list(range(len(ag))); random.Random(c['seed']).shuffle(order)
    groups = {i: [] for i in range(4)}
    for i in order:
        row = dict(ag[i], source_id=i); text = norm(row['text'])
        if text in excluded or len(groups[row['label']]) >= 125:
            continue
        groups[row['label']].append(row); excluded.add(text)
        if all(len(v) == 125 for v in groups.values()): break
    if any(len(v) != 125 for v in groups.values()): raise RuntimeError('Cannot prepare AG dev')
    files['ag_dev'] = [x for pool in groups.values() for x in pool]
    for name, rows in files.items(): atomic_json(out / (name + '.json'), rows)
    paths = [out / (name+'.json') for name in files]
    paths += [stage1 / (name+'.json') for name in original['sha256']]
    paths += [stage1/'manifest.json', ROOT/'configs/stage1.json']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    # Keep model hashes separate: rechecking multi-GB weights is done once per invocation.
    atomic_json(out / 'manifest.json', {'config_signature': signature(c), 'hashes': hashes,
        'model_hashes': {str(weights.relative_to(ROOT)): digest(weights), str(collapsed.relative_to(ROOT)): digest(collapsed)},
        'fingerprints': {'gsm_train':gsm._fingerprint, 'gsm_test':test_gsm._fingerprint, 'ag_train':ag._fingerprint},
        'counts': {k: {t:len(v) for t,v in rows.items()} if isinstance(rows,dict) else len(rows) for k,rows in files.items()}})
    print('Stage2 data prepared', flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/stage2.json')
    prepare(load_config(p.parse_args().config))
