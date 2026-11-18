"""M1：A→B 顺序 SFT pilot 的提示、编码、replay 排程与指标。"""
import math
import random
import shutil
from common import ROOT, chat, read_json
from stage2_common import atomic_json, digest, signature  # noqa: F401  (re-exported)

def roots(smoke=False):
    """Data is shared (smoke reads a prefix); outputs are isolated for smoke runs."""
    name = 'm1-smoke' if smoke else 'm1'
    return {'data': ROOT / 'data/m1', **{k: ROOT / k / name for k in ['artifacts', 'results', 'reports']}}


def load_config(path):
    c = read_json(path)
    t = c['training']
    if t['micro_batch_size'] * t['gradient_accumulation_steps'] * t['steps_per_stage'] > c['split_sizes']['train']:
        raise ValueError('A stage must not exceed one pass over its training set')
    return c


def batch_size(c):
    t = c['training']
    return t['micro_batch_size'] * t['gradient_accumulation_steps']


def eval_steps(c, smoke=False):
    if smoke: return [1, 2]
    n = c['training']['steps_per_stage']
    return sorted({round(f * n) for f in c['eval_fractions']})


def prompt(task, row, labels):
    options = ', '.join(labels)
    if task == 'ag':
        return f'Classify the news article. Reply with exactly one category: {options}.\n\nArticle: ' + row['text']
    if task == 'dbpedia':
        return (f'Classify the entity described in the Wikipedia text. Reply with exactly one category: {options}.\n\n'
                f"Title: {row['title']}\nText: {row['text']}")
    raise ValueError(task)


def encode(tokenizer, task, row, labels, max_length):
    """Assistant-only loss on label+EOS; keeps instruction head and the last 16 prompt tokens if truncation is needed."""
    prefix = tokenizer.encode(chat(tokenizer, prompt(task, row, labels)), add_special_tokens=False)
    target = tokenizer.encode(labels[row['label']] + tokenizer.eos_token, add_special_tokens=False)
    budget = max_length - len(target)
    truncated = len(prefix) > budget
    if truncated: prefix = prefix[:budget - 16] + prefix[-16:]
    return {'input_ids': prefix + target, 'attention_mask': [1] * (len(prefix) + len(target)),
            'labels': [-100] * len(prefix) + target}, truncated


def replay_slots(total, fraction):
    """Deterministic interleaving: slot k is replay iff floor((k+1)f) > floor(kf); exact cumulative fraction."""
    return [math.floor((k + 1) * fraction) > math.floor(k * fraction) for k in range(total)]


def stage_order(new_rows, buffer_rows, total, fraction, seed):
    """Fixed sample order for one stage. Replay replaces new-task slots; new-task order is shared across methods."""
    rng = random.Random(seed)
    new = list(new_rows); rng.shuffle(new)
    slots = replay_slots(total, fraction)
    n_replay = sum(slots)
    if total - n_replay > len(new): raise ValueError('Not enough new-task samples for this stage')
    replay = []
    if n_replay:
        if not buffer_rows: raise ValueError('Replay requested but buffer is empty')
        while len(replay) < n_replay:
            chunk = list(buffer_rows); random.Random(seed + len(replay)).shuffle(chunk); replay += chunk
        replay = replay[:n_replay]
    it_new, it_replay = iter(new), iter(replay)
    return [('replay', next(it_replay)) if s else ('new', next(it_new)) for s in slots]


def stratified_buffer(rows, size, seed):
    groups = {}
    for r in rows: groups.setdefault(r['label'], []).append(r)
    rng = random.Random(seed); keys = sorted(groups)
    out = []
    for i, k in enumerate(keys):
        pool = list(groups[k]); rng.shuffle(pool)
        out += pool[:size // len(keys) + (i < size % len(keys))]
    rng.shuffle(out)
    return out


def macro_f1(gold, pred, n_labels):
    """Invalid predictions (None) count as errors; F1 averaged over gold classes."""
    scores = []
    for k in range(n_labels):
        tp = sum(g == k and p == k for g, p in zip(gold, pred))
        fp = sum(g != k and p == k for g, p in zip(gold, pred))
        fn = sum(g == k and p != k for g, p in zip(gold, pred))
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(scores) / n_labels


def paired_bootstrap(a, b, samples, seed):
    """95% CI of mean(a)-mean(b) over the same test items (per-item correctness 0/1)."""
    if len(a) != len(b): raise ValueError('Paired bootstrap needs aligned items')
    rng = random.Random(seed); n = len(a); diffs = []
    for _ in range(samples):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(a[i] - b[i] for i in idx) / n)
    diffs.sort()
    return {'mean': sum(x - y for x, y in zip(a, b)) / n, 'ci95': [diffs[int(0.025 * samples)], diffs[int(0.975 * samples) - 1]]}


def check_space(c, reserve_bytes=0):
    free = shutil.disk_usage(ROOT).free
    if free - reserve_bytes < c['storage']['minimum_free_gib'] * 2**30:
        raise RuntimeError(f'M1 disk limit: free={free/2**30:.2f} GiB')
