"""Stage5：(A) 在 GSM8K 上训练（长输出、gold 风格与模型风格差异大），对照 sft_gold / sft_self / rft / grpo 的遗忘；
(B) AG 上用临界学习率 3e-5 重跑 Stage4 的四种方法（直接复用 stage4.run_arm）。持锁运行，已完成的臂跳过。"""
import argparse
import fcntl
import gc
import random
import shutil
import time
from datetime import datetime, timezone
import torch
from common import ROOT, chat, read_json
from eval_checkpoint import number
from stage2_common import atomic_json, digest, gsm_prompt, signature
import stage3
import stage4 as S4
from stage4 import EOS, log, load, sample, token_stats, offload, probe

OUT = ROOT / 'results/stage5'; DATA = ROOT / 'data/stage5'; REPORT = ROOT / 'reports/stage5'


def status(phase, unit=None):
    atomic_json(OUT / 'running-status.json', {'phase': phase, 'unit': unit, 'updated_utc': datetime.now(timezone.utc).isoformat()})


# ---------------- GSM 任务 ----------------

def gold(x): return number(x['answer'].split('####')[-1])


def gsm_ids(tok, x): return tok.encode(chat(tok, gsm_prompt(x)), add_special_tokens=False)


def correct(tok, x, q, fin): return fin and number(tok.decode(q, skip_special_tokens=True)) == gold(x)


def gsm_stream(c, n_rows):
    t = c['gsm']['training']; rows = read_json(ROOT / c['gsm']['train_data'])[:n_rows]; out = []; e = 0
    while len(out) < t['steps'] * t['prompts_per_step']:
        r = list(rows); random.Random(c['seed'] + e).shuffle(r); out += r; e += 1
    return out[:t['steps'] * t['prompts_per_step']]


def rollout(c, model, tok, xs, n, seed):
    t = c['gsm']['training']; k = t['rollout_prompts_per_call']; res = []
    for i in range(0, len(xs), k):
        res += sample(model, [gsm_ids(tok, x) for x in xs[i:i + k]], n, t['rollout']['max_new_tokens'], seed + i)
    return res


def gsm_self(c, ref, tok, smoke):
    path = DATA / ('gsm_self-smoke.json' if smoke else 'gsm_self.json')
    if path.exists(): return read_json(path)
    rows = read_json(ROOT / c['gsm']['train_data'])[:c['gsm']['n_rows_smoke'] if smoke else None]
    keep = [{**x, 'completion': q} for x, (q, fin) in zip(rows, rollout(c, ref, tok, rows, 1, c['seed'] + 20_000)) if smoke or correct(tok, x, q, fin)]
    if not keep: raise RuntimeError('gsm_self kept nothing')
    atomic_json(path, keep); log(f'gsm_self data: kept {len(keep)}/{len(rows)} base samples (T=1)')
    return keep


def style(out):
    """从 full 评测的 GSM 输出统计风格：含 GSM8K 标注式 `<<…>>` 的比例、含 #### 的比例、平均输出 token 数。"""
    recs = read_json(out / 'full' / 'gsm_test.json')
    return {'gsm8k_calc_rate': sum('<<' in r['output'] for r in recs) / len(recs), 'hash_rate': sum('####' in r['output'] for r in recs) / len(recs),
            'mean_output_tokens': sum(r['output_tokens'] for r in recs) / len(recs)}


def run_gsm_arm(c, arm, ref, tok, smoke):
    out = OUT / 'gsm' / arm['id']
    if (out / 'complete.json').exists(): log('skip', arm['id']); return
    if out.exists(): shutil.rmtree(out)
    status('running', 'gsm/' + arm['id']); log('start gsm', arm['id'])
    t = c['gsm']['training']; torch.manual_seed(c['seed']); random.seed(c['seed'])
    model = load(c); model.gradient_checkpointing_enable(); model.config.use_cache = False; model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=arm['lr'], weight_decay=t['weight_decay'])
    rows = gsm_stream(c, c['gsm']['n_rows_smoke'] if smoke else None); P = t['prompts_per_step']; G = t['group_size']
    selfrows = gsm_self(c, ref, tok, smoke) if arm['method'] == 'sft_self' else None
    steps = t['steps']; evals = set(c['gsm']['eval_steps']); train_log = []; t_start = time.perf_counter()
    for step in range(1, steps + 1):
        ts = time.perf_counter()
        for g in opt.param_groups: g['lr'] = arm['lr'] * min(1.0, step / t['warmup_steps'])
        batch = rows[(step - 1) * P: step * P]; rec = {'step': step}
        if arm['method'] == 'sft_gold':
            items = [(gsm_ids(tok, x), tok.encode(x['answer'] + tok.eos_token, add_special_tokens=False)) for x in batch]
            adv = [1.0] * len(items); norm = sum(len(q) for _, q in items)
        elif arm['method'] == 'sft_self':
            xs = [selfrows[((step - 1) * P + j) % len(selfrows)] for j in range(P)]
            items = [(gsm_ids(tok, x), x['completion']) for x in xs]; adv = [1.0] * len(items); norm = sum(len(q) for _, q in items)
        else:
            comps = rollout(c, model, tok, batch, G, c['seed'] * 100_003 + step)
            rewards = [float(correct(tok, batch[i // G], q, fin)) for i, (q, fin) in enumerate(comps)]
            items = [(gsm_ids(tok, batch[i // G]), q if q else [EOS[0]]) for i, (q, _) in enumerate(comps)]
            if arm['method'] == 'grpo':
                adv = []
                for i in range(P):
                    rs = rewards[i * G:(i + 1) * G]; mu = sum(rs) / G; sd = (sum((v - mu) ** 2 for v in rs) / G) ** 0.5
                    adv += [(v - mu) / (sd + 1e-4) if sd > 0 else 0.0 for v in rs]
                norm = sum(len(q) for _, q in items)
            else:  # rft
                adv = rewards; norm = sum(len(q) for (_, q), a in zip(items, adv) if a > 0)
            rec |= {'reward': sum(rewards) / len(rewards), 'zero_adv_groups': sum(all(a == 0 for a in adv[i * G:(i + 1) * G]) for i in range(P)) / P,
                    'finished': sum(f for _, f in comps) / len(comps), 'mean_len': sum(len(q) for q, _ in comps) / len(comps)}
        sums = {'loss': 0.0, 'surprisal': 0.0, 'surprisal_w': 0.0, 'w': 0.0, 'entropy': 0.0, 'tokens': 0}
        opt.zero_grad(set_to_none=True)
        if norm > 0:
            mb = t['micro_batch_size']
            for i in range(0, len(items), mb):
                chunk = items[i:i + mb]; a = adv[i:i + mb]
                if all(v == 0 for v in a): continue
                lps, ents, _ = token_stats(model, [p for p, _ in chunk], [q for _, q in chunk], True)
                loss = sum(-(av * lp).sum() for av, lp in zip(a, lps)) / norm
                loss.backward(); sums['loss'] += float(loss.detach())
                for av, lp, en in zip(a, lps, ents):
                    s = float(-lp.detach().sum()); sums['surprisal'] += s; sums['tokens'] += len(lp); sums['entropy'] += float(en.sum())
                    sums['surprisal_w'] += abs(av) * s; sums['w'] += abs(av) * len(lp)
                del lps, ents, loss
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), t['max_grad_norm'])); opt.step()
        else:
            gn = 0.0
        n = max(sums['tokens'], 1)
        rec |= {'loss': sums['loss'], 'grad_norm': gn, 'surprisal': sums['surprisal'] / n, 'surprisal_weighted': sums['surprisal_w'] / max(sums['w'], 1e-9),
                'entropy': sums['entropy'] / n, 'lr': opt.param_groups[0]['lr'], 'seconds': time.perf_counter() - ts}
        train_log.append(rec)
        if step % 4 == 0 or smoke: log(arm['id'], {k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()})
        if step in evals or step == steps:
            opt.zero_grad(set_to_none=True); offload(opt, 'cpu'); torch.cuda.empty_cache()
            d = out / f'step-{step}'; probe(c, model, ref, tok, d, step, step == steps, smoke)
            if step == steps:
                m = read_json(d / 'metrics.json'); m['style'] = style(d); atomic_json(d / 'metrics.json', m)
            offload(opt, 'cuda'); model.train()
    atomic_json(out / 'train_log.json', train_log)
    atomic_json(out / 'complete.json', {'utc': datetime.now(timezone.utc).isoformat(), 'wall_seconds': time.perf_counter() - t_start})
    del opt, model; gc.collect(); torch.cuda.empty_cache()


# ---------------- report ----------------

def pk(p): return next(v for key, v in p.items() if key.startswith('pass@') and key != 'pass@1')


def report(c):
    REPORT.mkdir(parents=True, exist_ok=True)
    M = lambda p: read_json(p / 'metrics.json') if (p / 'complete.json').exists() else None
    pc = lambda x: f'{100*x:.1f}'
    ref = M(OUT / 'reference')
    if ref and 'style' not in ref: ref['style'] = style(OUT / 'reference')
    k = c['probes']['passk']['k']

    def row(name, m):
        f, p = m['full'], m['passk']; s = m.get('style'); u = m.get('update', {}).get('all', {'relative_delta': 0})
        sty = f"{pc(s['gsm8k_calc_rate'])} / {s['mean_output_tokens']:.0f}" if s else '–'
        return (f"| {name} | {pc(f['gsm_test']['accuracy'])} | {pc(p['pass@1'])} / {pc(pk(p))} / {p['distinct_answers']:.2f} | {sty} | "
                f"{pc(f['ag_test']['accuracy'])} | {pc(f['mmlu_test']['accuracy'])} | {f['mmlu_test']['letter_probability_mass']:.2g} | "
                f"{pc(f['direct']['compliance'])} / {pc(f['direct']['accuracy'])} | {m['kl']['gsm']['kl_seq_mean']:.3f} | {m['kl']['ag']['kl_seq_mean']:.3f} | {u['relative_delta']:.2e} |")
    head = ['| 臂 | GSM | GSM pass@1 / pass@8 / 不同答案数 | `<<>>` 比例 / 平均输出 token | AG | MMLU | 字母概率 | 只回答数字 遵循 / 准确 | KL·GSM | KL·AG | 相对更新 |',
            '|---|---:|---|---|---:|---:|---:|---|---:|---:|---:|']
    L = ['# Stage5：长输出任务上的 RL vs SFT，以及临界学习率', '', '单 seed、全参。探索性。KL 为沿自身采样路径的序列 KL(π‖π_base)。', '',
         f"## A. 在 GSM8K 上训练（{c['gsm']['training']['steps']} 步，每步 {c['gsm']['training']['prompts_per_step']} 题，RL 每题采样 {c['gsm']['training']['group_size']} 条）", ''] + head
    if ref: L.append(row('base', ref))
    for arm in c['gsm']['arms']:
        m = M(OUT / 'gsm' / arm['id'] / f"step-{c['gsm']['training']['steps']}")
        if m and 'full' in m: L.append(row(arm['id'], m))
    L += ['', '过程（quick：GSM / AG / MMLU / 只回答数字遵循 %，KL·GSM）：', '']
    for arm in c['gsm']['arms']:
        cells = []
        for s in sorted(set(c['gsm']['eval_steps'])):
            if (m := M(OUT / 'gsm' / arm['id'] / f'step-{s}')):
                q = m['quick']; cells.append(f"{s}: {pc(q['gsm_test']['accuracy'])}/{pc(q['ag_test']['accuracy'])}/{pc(q['mmlu_test']['accuracy'])}/{pc(q['direct']['compliance'])} KL {m['kl']['gsm']['kl_seq_mean']:.2f}")
        lp = OUT / 'gsm' / arm['id'] / 'train_log.json'
        if lp.exists():
            lg = read_json(lp); mean = lambda xs, key: sum(x[key] for x in xs) / len(xs); f, l = lg[:8], lg[-8:]
            extra = f"；奖励 {mean(f, 'reward'):.2f} → {mean(l, 'reward'):.2f}，零优势组 {mean(lg, 'zero_adv_groups'):.2f}" if 'reward' in lg[0] else ''
            cells.append(f"惊讶度 {mean(f, 'surprisal_weighted'):.3f} → {mean(l, 'surprisal_weighted'):.3f}，熵 {mean(f, 'entropy'):.3f} → {mean(l, 'entropy'):.3f}{extra}")
        if cells: L.append(f"- {arm['id']}：" + '；'.join(cells))
    L += ['', '## B. AG 上的临界学习率 3e-5（与 Stage4 的 1e-4 / 1e-5 并列）', '',
          '| 臂 | AG | GSM | MMLU | 字母概率 | GSM 新闻标签 | KL·AG | KL·GSM | GSM pass@8 |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    s4 = ROOT / 'results/stage4'; s4c = read_json(ROOT / 'configs/stage4.json')
    lines = [(a['id'], s4 / a['id'] / f"step-{s4c['training']['steps']}") for a in s4c['arms'] if a['method'] != 'grpo' or a.get('beta', 0) == 0]
    lines += [(a['id'], OUT / 'ag' / a['id'] / f"step-{c['training']['steps']}") for a in c['arms']]
    order = lambda x: (['sft_gold', 'sft_self', 'grpo', 'rft'].index(x[0].rsplit('_lr', 1)[0]), x[0])
    for name, p in sorted(lines, key=order):
        if (m := M(p)) and 'full' in m:
            f = m['full']
            L.append(f"| {name} | {pc(f['ag_test']['accuracy'])} | {pc(f['gsm_test']['accuracy'])} | {pc(f['mmlu_test']['accuracy'])} | {f['mmlu_test']['letter_probability_mass']:.2g} | "
                     f"{pc(f['gsm_test']['news_label_rate'])} | {m['kl']['ag']['kl_seq_mean']:.3f} | {m['kl']['gsm']['kl_seq_mean']:.3f} | {pc(pk(m['passk']))} |")
    L += ['', '过程（quick：AG / GSM / MMLU %）：', '']
    for arm in c['arms']:
        cells = [f"{s}: {pc(m['quick']['ag_test']['accuracy'])}/{pc(m['quick']['gsm_test']['accuracy'])}/{pc(m['quick']['mmlu_test']['accuracy'])}"
                 for s in sorted(set(c['probes']['eval_steps'])) if (m := M(OUT / 'ag' / arm['id'] / f'step-{s}'))]
        if cells: L.append(f"- {arm['id']}：" + '；'.join(cells))
    (REPORT / 'SUMMARY.md').write_text('\n'.join(L) + '\n'); log('Report:', REPORT / 'SUMMARY.md')


# ---------------- main ----------------

def main():
    global OUT, DATA, REPORT
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', default='configs/stage5.json'); p.add_argument('--smoke', action='store_true')
    p.add_argument('--parts', nargs='+', default=['gsm', 'ag']); p.add_argument('--report-only', action='store_true'); a = p.parse_args()
    c = read_json(ROOT / a.config)
    if a.smoke:
        OUT = ROOT / 'results/stage5-smoke'; REPORT = ROOT / 'reports/stage5-smoke'; DATA = ROOT / 'data/stage5-smoke'
        c['gsm']['training'] |= {'steps': 2, 'prompts_per_step': 2, 'group_size': 2, 'warmup_steps': 1, 'rollout_prompts_per_call': 2}
        c['gsm']['training']['rollout']['max_new_tokens'] = 32; c['gsm']['eval_steps'] = [2]
        c['training'] |= {'steps': 2, 'prompts_per_step': 2, 'group_size': 2, 'warmup_steps': 1}
        c['probes'] = {'eval_steps': [2], 'kl': {'ag_prompts': 4, 'ag_max_new_tokens': 8, 'gsm_prompts': 2, 'gsm_max_new_tokens': 8},
                       'passk': {'questions': 2, 'k': 2, 'temperature': 0.7, 'max_new_tokens': 16, 'questions_per_batch': 2}}
        c['direct_probe'] |= {'quick_n': 4, 'full_n': 4}
        c['gsm']['arms'] = [x for x in c['gsm']['arms'] if x['method'] in ('sft_gold', 'sft_self', 'grpo')]
        c['arms'] = c['arms'][:1]
    if a.report_only: return report(c)
    OUT.mkdir(parents=True, exist_ok=True)
    S4.OUT = OUT / 'ag'; S4.REPORT = REPORT / 'ag-stage4-format'  # B 部分复用 stage4.run_arm；sft_self 数据沿用 data/stage4（同 seed、同数据流）
    if a.smoke: S4.DATA = DATA
    with (OUT / '.lock').open('w') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('Another stage5 run is active')
        identity = {'config': c, 'uv_lock': digest(ROOT / 'uv.lock'), 'data': {x: digest(ROOT / x) for x in [c['train_data'], c['gsm']['train_data']]},
                    'code': {s: digest(ROOT / 'scripts' / s) for s in ['stage5.py', 'stage4.py', 'stage3.py', 'stage2_eval.py', 'stage2_common.py', 'common.py']}}
        ep = OUT / 'experiment.json'
        if ep.exists() and read_json(ep)['signature'] != signature(identity):
            raise RuntimeError('Code/config/data changed; archive results/stage5 first')
        atomic_json(ep, {'signature': signature(identity), 'identity': identity})
        try:
            tok = stage3.tokenizer(c); ref = load(c).eval()
            for prm in ref.parameters(): prm.requires_grad_(False)
            rd = OUT / 'reference'; src = ROOT / 'results/stage4/reference'
            if not (rd / 'complete.json').exists():
                if not a.smoke and (src / 'complete.json').exists() and read_json(ROOT / 'configs/stage4.json')['probes'] == c['probes']:
                    shutil.copytree(src, rd, dirs_exist_ok=True); log('reference: copied from stage4 (same base, same probes)')
                else:
                    status('running', 'reference'); probe(c, ref, ref, tok, rd, 0, True, a.smoke)
            if 'gsm' in a.parts:
                for arm in c['gsm']['arms']: run_gsm_arm(c, arm, ref, tok, a.smoke); report(c)
            if 'ag' in a.parts:
                for arm in c['arms']: S4.run_arm(c, arm, ref, tok, a.smoke); report(c)
            status('complete')
        except BaseException as err:
            status('failed', repr(err)); raise
        finally:
            report(c)


if __name__ == '__main__': main()
