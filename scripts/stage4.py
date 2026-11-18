"""Stage4：在 AG News 上对照 RL 与 SFT 的遗忘（单 seed，全参，同一训练循环，只换损失与数据来源）。
臂：SFT（gold / 自采样）、GRPO（有/无 KL）、在线 RFT；探针：quick/full 评测、对 base 的 KL（新/旧任务分布）、
训练 token 惊讶度、GSM pass@k 与答案多样性、更新幅度与稀疏度。持锁运行，已完成的臂跳过。"""
import argparse
import csv
import fcntl
import gc
import json
import math
import random
import shutil
import time
from datetime import datetime, timezone
import torch
import torch.nn.functional as F
from common import ROOT, LABELS, ag_prompt, chat, read_json
from eval_checkpoint import number
from stage2_common import atomic_json, digest, gsm_prompt, signature
import stage3

OUT = ROOT / 'results/stage4'; DATA = ROOT / 'data/stage4'; REPORT = ROOT / 'reports/stage4'
EOS = [151645, 151643]; PAD = 151643


def log(*a): print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}]", *a, flush=True)


def status(phase, unit=None):
    atomic_json(OUT / 'running-status.json', {'phase': phase, 'unit': unit, 'updated_utc': datetime.now(timezone.utc).isoformat()})


def load(c):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(ROOT / c['models']['base'], local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    return m.cuda()


def prompt_ids(tok, text, cap):
    ids = tok.encode(chat(tok, ag_prompt(text)), add_special_tokens=False)
    return ids if len(ids) <= cap else ids[:cap - 16] + ids[-16:]


def cut(tokens):
    """截到第一个 EOS（含）；返回 (tokens, finished)。"""
    for i, t in enumerate(tokens):
        if t in EOS: return tokens[:i + 1], True
    return tokens, False


# ---------------- sampling & log-probs ----------------

@torch.no_grad()
def sample(model, prompts, n, max_new, seed, temperature=1.0):
    """纯温度采样（关闭 top-p/top-k/重复惩罚，保证与训练时的 log-prob 同分布）。prompts: token id 列表。"""
    ckpt = model.is_gradient_checkpointing; train_mode = model.training
    if ckpt: model.gradient_checkpointing_disable()
    model.eval(); model.config.use_cache = True
    torch.manual_seed(seed)
    L = max(len(p) for p in prompts)
    ids = torch.tensor([[PAD] * (L - len(p)) + p for p in prompts], device='cuda')
    mask = torch.tensor([[0] * (L - len(p)) + [1] * len(p) for p in prompts], device='cuda')
    out = model.generate(input_ids=ids, attention_mask=mask, do_sample=True, temperature=temperature, top_p=1.0, top_k=0,
                         repetition_penalty=1.0, max_new_tokens=max_new, num_return_sequences=n, eos_token_id=EOS, pad_token_id=PAD)
    res = [cut(row) for row in out[:, L:].tolist()]
    model.config.use_cache = False
    if ckpt: model.gradient_checkpointing_enable()
    model.train(train_mode)
    return res  # len(prompts)*n, prompt-major


def token_stats(model, prompts, comps, grad):
    """返回每条补全的逐 token logp（list of 1D tensors）与熵；只在补全位置算 lm_head。"""
    L = max(len(p) + len(q) for p, q in zip(prompts, comps))
    ids = torch.tensor([p + q + [PAD] * (L - len(p) - len(q)) for p, q in zip(prompts, comps)], device='cuda')
    mask = torch.tensor([[1] * (len(p) + len(q)) + [0] * (L - len(p) - len(q)) for p, q in zip(prompts, comps)], device='cuda')
    with torch.set_grad_enabled(grad):
        h = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
        rows = torch.cat([torch.full((len(q),), i, device='cuda') for i, q in enumerate(comps)])
        cols = torch.cat([torch.arange(len(p) - 1, len(p) - 1 + len(q), device='cuda') for p, q in zip(prompts, comps)])
        targets = torch.tensor([t for q in comps for t in q], device='cuda')
        logits = model.lm_head(h[rows, cols]).float()
        lp = F.log_softmax(logits, -1)
        tok_lp = lp.gather(1, targets[:, None]).squeeze(1)
        ent = -(lp.exp() * lp).sum(-1)
    sizes = [len(q) for q in comps]
    return tok_lp.split(sizes), ent.detach().split(sizes), lp


@torch.no_grad()
def exact_kl(model, ref, prompts, comps):
    """沿 policy 自己采样的路径，逐位置精确计算 KL(π‖π_base)（全词表）并求和；同时返回首 token 熵。"""
    _, ent, lp = token_stats(model, prompts, comps, False)
    _, _, lq = token_stats(ref, prompts, comps, False)
    kl = (lp.exp() * (lp - lq)).sum(-1).split([len(q) for q in comps])
    return [float(k.sum()) for k in kl], [float(e[0]) for e in ent]


# ---------------- probes ----------------

def offload(opt, device):
    for st in opt.state.values():
        for k, v in st.items():
            if torch.is_tensor(v): st[k] = v.to(device, non_blocking=True)
    torch.cuda.empty_cache()


def kl_probe(c, model, ref, tok, smoke):
    p = c['probes']['kl']; full = read_json(ROOT / 'data/stage2/eval_full.json'); res = {}
    sets = {'ag': ([prompt_ids(tok, r['text'], 2048) for r in full['ag_test'][:p['ag_prompts']]], p['ag_max_new_tokens']),
            'gsm': ([tok.encode(chat(tok, gsm_prompt(r)), add_special_tokens=False) for r in full['gsm_test'][-p['gsm_prompts']:]], p['gsm_max_new_tokens'])}
    for name, (prompts, max_new) in sets.items():
        kls, ents = [], []
        for i in range(0, len(prompts), 16):
            ps = prompts[i:i + 16]; comps = [q for q, _ in sample(model, ps, 1, max_new, c['seed'] + i)]
            comps = [q if q else [EOS[0]] for q in comps]
            k, e = exact_kl(model, ref, ps, comps); kls += k; ents += e
        res[name] = {'kl_seq_mean': sum(kls) / len(kls), 'first_token_entropy': sum(ents) / len(ents), 'n': len(kls)}
    return res


def passk(c, model, tok):
    p = c['probes']['passk']; rows = read_json(ROOT / 'data/stage2/eval_full.json')['gsm_test'][:p['questions']]; out = []
    for i in range(0, len(rows), p['questions_per_batch']):
        batch = rows[i:i + p['questions_per_batch']]
        prompts = [tok.encode(chat(tok, gsm_prompt(r)), add_special_tokens=False) for r in batch]
        comps = sample(model, prompts, p['k'], p['max_new_tokens'], c['seed'] + i, p['temperature'])
        for j, r in enumerate(batch):
            gold = number(r['answer'].split('####')[-1])
            answers = [number(tok.decode(q, skip_special_tokens=True)) for q, _ in comps[j * p['k']:(j + 1) * p['k']]]
            out.append({'source_id': r['source_id'], 'answers': answers, 'correct': [a == gold for a in answers]})
    k = p['k']
    return {'pass@1': sum(sum(x['correct']) / k for x in out) / len(out), f'pass@{k}': sum(any(x['correct']) for x in out) / len(out),
            'distinct_answers': sum(len({a for a in x['answers'] if a is not None}) for x in out) / len(out), 'n': len(out)}, out


def groups_of(name):
    if 'embed_tokens' in name: return 'embed'
    if '.self_attn.' in name: return 'attn'
    if '.mlp.' in name: return 'mlp'
    return 'norm'


@torch.no_grad()
def update_stats(model, ref):
    acc = {}
    for (n, p), (_, q) in zip(model.named_parameters(), ref.named_parameters()):
        g = acc.setdefault(groups_of(n), [0.0, 0.0, 0, 0]); d = (p.float() - q.float())
        g[0] += float(d.pow(2).sum()); g[1] += float(q.float().pow(2).sum()); g[2] += int((p != q).sum()); g[3] += p.numel()
    tot = [sum(v[i] for v in acc.values()) for i in range(4)]; acc['all'] = tot
    return {k: {'relative_delta': math.sqrt(v[0] / v[1]), 'fraction_changed': v[2] / v[3]} for k, v in acc.items()}


def probe(c, model, ref, tok, out, step, final, smoke):
    if (out / 'complete.json').exists(): return
    t0 = time.perf_counter(); ec = c
    m = {'step': step, 'quick': stage3.evaluate(ec, model, out / 'quick', 'quick', smoke)}
    m['kl'] = kl_probe(c, model, ref, tok, smoke)
    if final:
        m['full'] = stage3.evaluate(ec, model, out / 'full', 'full', smoke)
        m['passk'], recs = passk(c, model, tok); atomic_json(out / 'passk.json', recs)
        m['update'] = update_stats(model, ref)
    m['seconds'] = time.perf_counter() - t0
    atomic_json(out / 'metrics.json', m); atomic_json(out / 'complete.json', {'utc': datetime.now(timezone.utc).isoformat()})
    k = c['probes']['passk']['k']
    tail = f" | pass@1 {m['passk']['pass@1']:.3f} pass@{k} {m['passk'][f'pass@{k}']:.3f}" if final else ''
    log(f"probe {out.relative_to(ROOT)}: KL ag {m['kl']['ag']['kl_seq_mean']:.3f} gsm {m['kl']['gsm']['kl_seq_mean']:.3f}{tail} ({m['seconds']:.0f}s)")


# ---------------- data ----------------

def stream(c, smoke):
    t = c['training']; rows = read_json(ROOT / c['train_data']); random.Random(c['seed']).shuffle(rows)
    return rows[:t['steps'] * t['prompts_per_step']]


def self_data(c, ref, tok, smoke):
    path = DATA / ('sft_self-smoke.json' if smoke else 'sft_self.json')
    if path.exists(): return read_json(path)
    t = c['training']; rows = stream(c, smoke); keep = []; r = t['rollout']
    for i in range(0, len(rows), 64):
        batch = rows[i:i + 64]; ps = [prompt_ids(tok, x['text'], t['max_prompt_tokens']) for x in batch]
        for x, (q, fin) in zip(batch, sample(ref, ps, 1, r['max_new_tokens'], c['seed'] + 10_000 + i)):
            if smoke or (fin and tok.decode(q, skip_special_tokens=True).strip() == LABELS[x['label']]): keep.append({**x, 'completion': q})
    atomic_json(path, keep); log(f'sft_self data: kept {len(keep)}/{len(rows)} base samples (T=1)')
    return keep


# ---------------- training ----------------

def run_arm(c, arm, ref, tok, smoke):
    out = OUT / arm['id']
    if (out / 'complete.json').exists(): log('skip', arm['id']); return
    if out.exists(): shutil.rmtree(out)
    status('running', arm['id']); log('start', arm['id'])
    t = c['training']; torch.manual_seed(c['seed']); random.seed(c['seed'])
    model = load(c); model.gradient_checkpointing_enable(); model.config.use_cache = False; model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=arm['lr'], weight_decay=t['weight_decay'])
    rows = stream(c, smoke); P = t['prompts_per_step']; G = t['group_size']; r = t['rollout']
    selfrows = self_data(c, ref, tok, smoke) if arm['method'] == 'sft_self' else None
    steps = t['steps']; evals = set(c['probes']['eval_steps']); train_log = []
    probe_dir = lambda s: out / f'step-{s}'
    t_start = time.perf_counter()
    for step in range(1, steps + 1):
        ts = time.perf_counter()
        for g in opt.param_groups: g['lr'] = arm['lr'] * min(1.0, step / t['warmup_steps'])
        batch = rows[(step - 1) * P: step * P]
        rec = {'step': step}
        if arm['method'] in ('sft_gold', 'sft_self'):
            if arm['method'] == 'sft_gold':
                items = [(prompt_ids(tok, x['text'], t['max_prompt_tokens']), tok.encode(LABELS[x['label']] + tok.eos_token, add_special_tokens=False)) for x in batch]
            else:
                items = [(prompt_ids(tok, x['text'], t['max_prompt_tokens']), x['completion']) for x in (selfrows[((step - 1) * P + j) % len(selfrows)] for j in range(P))]
            adv = [1.0] * len(items); norm = sum(len(q) for _, q in items)
        else:
            ps = [prompt_ids(tok, x['text'], t['max_prompt_tokens']) for x in batch]
            comps = sample(model, ps, G, r['max_new_tokens'], c['seed'] * 100_003 + step)
            items, rewards = [], []
            for i, (q, fin) in enumerate(comps):
                x = batch[i // G]
                rewards.append(float(fin and tok.decode(q, skip_special_tokens=True).strip() == LABELS[x['label']]))
                items.append((ps[i // G], q if q else [EOS[0]]))
            if arm['method'] == 'grpo':
                adv = []
                for i in range(P):
                    rs = rewards[i * G:(i + 1) * G]; mu = sum(rs) / G; sd = (sum((v - mu) ** 2 for v in rs) / G) ** 0.5
                    adv += [(v - mu) / (sd + 1e-4) if sd > 0 else 0.0 for v in rs]
                norm = sum(len(q) for _, q in items)
            else:  # rft：只在答对的自采样上做 SFT
                adv = rewards; norm = sum(len(q) for (_, q), a in zip(items, adv) if a > 0)
            rec |= {'reward': sum(rewards) / len(rewards), 'zero_adv_groups': sum(all(a == 0 for a in adv[i * G:(i + 1) * G]) for i in range(P)) / P}
        beta = arm.get('beta', 0.0); sums = {'loss': 0.0, 'surprisal': 0.0, 'surprisal_w': 0.0, 'w': 0.0, 'entropy': 0.0, 'tokens': 0, 'kl_k3': 0.0}
        opt.zero_grad(set_to_none=True)
        if norm > 0:
            mb = t['micro_batch_size']
            for i in range(0, len(items), mb):
                chunk = items[i:i + mb]; a = adv[i:i + mb]
                if beta == 0 and all(v == 0 for v in a):
                    continue
                lps, ents, _ = token_stats(model, [p for p, _ in chunk], [q for _, q in chunk], True)
                loss = sum(-(av * lp).sum() for av, lp in zip(a, lps)) / norm
                if beta > 0:
                    with torch.no_grad():
                        rlps, _, _ = token_stats(ref, [p for p, _ in chunk], [q for _, q in chunk], False)
                    k3 = sum(((rl - lp).exp() - (rl - lp) - 1).sum() for rl, lp in zip(rlps, lps))
                    loss = loss + beta * k3 / norm; sums['kl_k3'] += float(k3)
                loss.backward(); sums['loss'] += float(loss.detach())
                for av, lp, en in zip(a, lps, ents):
                    s = float(-lp.detach().sum()); sums['surprisal'] += s; sums['tokens'] += len(lp); sums['entropy'] += float(en.sum())
                    sums['surprisal_w'] += abs(av) * s; sums['w'] += abs(av) * len(lp)
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), t['max_grad_norm'])); opt.step()
        else:
            gn = 0.0
        n = max(sums['tokens'], 1)
        rec |= {'loss': sums['loss'], 'grad_norm': gn, 'surprisal': sums['surprisal'] / n, 'surprisal_weighted': sums['surprisal_w'] / max(sums['w'], 1e-9),
                'entropy': sums['entropy'] / n, 'kl_k3_per_token': sums['kl_k3'] / n, 'lr': opt.param_groups[0]['lr'], 'seconds': time.perf_counter() - ts}
        train_log.append(rec)
        if step % 10 == 0 or smoke: log(arm['id'], {k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()})
        if step in evals or step == steps:
            opt.zero_grad(set_to_none=True); offload(opt, 'cpu')
            probe(c, model, ref, tok, probe_dir(step), step, step == steps, smoke)
            offload(opt, 'cuda'); model.train()
    atomic_json(out / 'train_log.json', train_log)
    atomic_json(out / 'complete.json', {'utc': datetime.now(timezone.utc).isoformat(), 'wall_seconds': time.perf_counter() - t_start})
    del opt, model; gc.collect(); torch.cuda.empty_cache()


# ---------------- report ----------------

def report(c):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    REPORT.mkdir(parents=True, exist_ok=True)
    M = lambda p: read_json(p / 'metrics.json') if (p / 'complete.json').exists() else None
    pc = lambda x: f'{100*x:.1f}'
    rows = []; ref = M(OUT / 'reference')
    if ref: rows.append({'arm': 'base', 'step': 0, **flat(ref)})
    for arm in c['arms']:
        for s in sorted(set(c['probes']['eval_steps']) | {c['training']['steps']}):
            if (m := M(OUT / arm['id'] / f'step-{s}')): rows.append({'arm': arm['id'], 'step': s, **flat(m)})
    if not rows: return
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('arm', 'step'), k))
    with (REPORT / 'probes.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    L = ['# Stage4：RL vs SFT 遗忘对照', '', '单 seed、全参、AG News 训练 200 步（每步 16 个提示；RL 每提示采样 8 条）。探索性。', '',
         '## 终点（full 评测 %，KL 为沿自身采样路径的序列 KL(π‖π_base)）', '',
         '| 臂 | AG | GSM | MMLU | 字母概率 | GSM 新闻标签 | 只回答数字遵循 | KL·AG | KL·GSM | GSM pass@1 / pass@8 / 不同答案数 | 相对更新 | 改变参数比例 |',
         '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|']
    for arm in [{'id': 'base'}] + c['arms']:
        m = ref if arm['id'] == 'base' else M(OUT / arm['id'] / f"step-{c['training']['steps']}")
        if not m or 'full' not in m: continue
        f_, p, k = m['full'], m['passk'], c['probes']['passk']['k']; u = m.get('update', {}).get('all', {'relative_delta': 0, 'fraction_changed': 0})
        L.append(f"| {arm['id']} | {pc(f_['ag_test']['accuracy'])} | {pc(f_['gsm_test']['accuracy'])} | {pc(f_['mmlu_test']['accuracy'])} | {f_['mmlu_test']['letter_probability_mass']:.2g} | "
                 f"{pc(f_['gsm_test']['news_label_rate'])} | {pc(f_['direct']['compliance'])} | {m['kl']['ag']['kl_seq_mean']:.3f} | {m['kl']['gsm']['kl_seq_mean']:.3f} | "
                 f"{pc(p['pass@1'])} / {pc(p[f'pass@{k}'])} / {p['distinct_answers']:.2f} | {u['relative_delta']:.2e} | {u['fraction_changed']:.3f} |")
    L += ['', '## 过程（quick 评测：AG / GSM / MMLU %，KL·AG）', '']
    for arm in c['arms']:
        cells = [f"{r['step']}: {pc(r['quick_ag'])}/{pc(r['quick_gsm'])}/{pc(r['quick_mmlu'])} KL {r['kl_ag']:.2f}" for r in rows if r['arm'] == arm['id']]
        if cells: L.append(f"- {arm['id']}：" + '；'.join(cells))
    L += ['', '## 训练 token 惊讶度（−log p，均值；RL 为采样 token，按 |优势| 加权）', '']
    logs = {}
    for arm in c['arms']:
        p = OUT / arm['id'] / 'train_log.json'
        if p.exists():
            lg = read_json(p); logs[arm['id']] = lg
            first = lg[:10]; last = lg[-10:]; mean = lambda xs, k: sum(x[k] for x in xs) / len(xs)
            extra = f"；平均奖励 {mean(first, 'reward'):.2f} → {mean(last, 'reward'):.2f}" if 'reward' in lg[0] else ''
            L.append(f"- {arm['id']}：前 10 步 {mean(first, 'surprisal_weighted'):.3f} → 后 10 步 {mean(last, 'surprisal_weighted'):.3f}；熵 {mean(first, 'entropy'):.3f} → {mean(last, 'entropy'):.3f}{extra}")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for arm in c['arms']:
        xs = [r for r in rows if r['arm'] == arm['id']]
        if not xs: continue
        base = [r for r in rows if r['arm'] == 'base']; xs = base + xs
        ax[0].plot([r['step'] for r in xs], [100 * r['quick_ag'] for r in xs], marker='o', ms=3, label=arm['id'])
        ax[1].plot([r['step'] for r in xs], [100 * r['quick_gsm'] for r in xs], marker='o', ms=3, label=arm['id'])
        ax[2].scatter([r['kl_ag'] for r in xs], [100 * r['quick_gsm'] for r in xs], s=14, label=arm['id'])
    ax[0].set_title('AG (quick) vs step'); ax[1].set_title('GSM (quick) vs step'); ax[2].set_title('GSM (quick) vs KL on AG prompts')
    ax[2].set_xscale('symlog', linthresh=0.01); ax[2].set_xlabel('KL(pi || base), nats per sequence'); ax[0].legend(fontsize=7)
    for a in ax[:2]: a.set_xlabel('step')
    fig.tight_layout(); fig.savefig(REPORT / 'curves.png', dpi=150); plt.close(fig)
    if logs:
        fig, ax = plt.subplots(figsize=(6, 4))
        for k, lg in logs.items(): ax.plot([x['step'] for x in lg], [x['surprisal_weighted'] for x in lg], label=k, lw=1)
        ax.set_yscale('log'); ax.set_xlabel('step'); ax.set_ylabel('trained-token surprisal (nats)'); ax.legend(fontsize=7); fig.tight_layout()
        fig.savefig(REPORT / 'surprisal.png', dpi=150); plt.close(fig)
    L += ['', '图：curves.png（AG/GSM 曲线与 GSM–KL 散点）、surprisal.png。逐点数据：probes.csv。']
    (REPORT / 'SUMMARY.md').write_text('\n'.join(L) + '\n'); log('Report:', REPORT / 'SUMMARY.md')


def flat(m):
    q = m['quick']
    return {'quick_ag': q['ag_test']['accuracy'], 'quick_gsm': q['gsm_test']['accuracy'], 'quick_mmlu': q['mmlu_test']['accuracy'],
            'quick_letter_mass': q['mmlu_test']['letter_probability_mass'], 'quick_gsm_news': q['gsm_test']['news_label_rate'],
            'quick_direct': q['direct']['compliance'], 'kl_ag': m['kl']['ag']['kl_seq_mean'], 'kl_gsm': m['kl']['gsm']['kl_seq_mean'],
            'ent_ag_first': m['kl']['ag']['first_token_entropy'], 'ent_gsm_first': m['kl']['gsm']['first_token_entropy']}


def main():
    global OUT, DATA, REPORT
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', default='configs/stage4.json'); p.add_argument('--smoke', action='store_true')
    p.add_argument('--arms', nargs='+'); p.add_argument('--report-only', action='store_true'); a = p.parse_args()
    c = read_json(ROOT / a.config)
    if a.smoke:
        OUT = ROOT / 'results/stage4-smoke'; REPORT = ROOT / 'reports/stage4-smoke'
        c['training'] |= {'steps': 2, 'prompts_per_step': 2, 'group_size': 2, 'warmup_steps': 1}
        c['probes'] = {'eval_steps': [2], 'kl': {'ag_prompts': 4, 'ag_max_new_tokens': 8, 'gsm_prompts': 2, 'gsm_max_new_tokens': 8},
                       'passk': {'questions': 2, 'k': 2, 'temperature': 0.7, 'max_new_tokens': 16, 'questions_per_batch': 2}}
        c['direct_probe'] |= {'quick_n': 4, 'full_n': 4}
        c['arms'] = [x for x in c['arms'] if x['id'] in ('sft_self_lr1e4', 'grpo_kl_lr1e5', 'rft_lr1e5')]
    if a.report_only: return report(c)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / '.lock').open('w') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('Another stage4 run is active')
        identity = {'config': c, 'uv_lock': digest(ROOT / 'uv.lock'), 'data': digest(ROOT / c['train_data']),
                    'code': {s: digest(ROOT / 'scripts' / s) for s in ['stage4.py', 'stage3.py', 'stage2_eval.py', 'stage2_common.py', 'common.py']}}
        ep = OUT / 'experiment.json'
        if ep.exists() and read_json(ep)['signature'] != signature(identity):
            raise RuntimeError('Code/config/data changed; archive results/stage4 first')
        atomic_json(ep, {'signature': signature(identity), 'identity': identity})
        if a.arms: c['arms'] = [x for x in c['arms'] if x['id'] in a.arms]
        try:
            tok = stage3.tokenizer(c); ref = load(c).eval()
            for prm in ref.parameters(): prm.requires_grad_(False)
            if not (OUT / 'reference' / 'complete.json').exists():
                status('running', 'reference'); probe(c, ref, ref, tok, OUT / 'reference', 0, True, a.smoke); report(c)
            for arm in c['arms']:
                run_arm(c, arm, ref, tok, a.smoke); report(c)
            status('complete')
        except BaseException as err:
            status('failed', repr(err)); raise
        finally:
            report(c)


if __name__ == '__main__': main()
