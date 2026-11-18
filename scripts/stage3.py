"""Stage3 直觉实验队列（单 seed）：参考评测、权重插值、按层回退/移植、非数学恢复、MMLU 恢复、自蒸馏恢复、Stage1 混入通用数据。
持锁运行；每个单元写 complete.json，重跑时跳过；代码/配置/数据变化时拒绝混用。"""
import argparse
import fcntl
import gc
import math
import random
import re
import time
from datetime import datetime, timezone
import torch
from torch.utils.data import SequentialSampler
from common import ROOT, chat, read_json
from eval_checkpoint import request, number
from stage2_common import atomic_json, digest, gsm_prompt, signature, math_encode, load_config as load_stage2

MMLU_HEAD = 'The following is a multiple-choice question. Reply with the letter of the correct answer.\n'
DIRECT = re.compile(r'^\$?\s*-?[\d,]*\.?\d+\s*\.?$')
OUT = ROOT / 'results/stage3'; DATA = ROOT / 'data/stage3'; REPORT = ROOT / 'reports/stage3'
CACHE = {}


def done(path): return (path / 'complete.json').exists()


def status(phase, unit=None):
    atomic_json(OUT / 'running-status.json', {'phase': phase, 'unit': unit, 'updated_utc': datetime.now(timezone.utc).isoformat()})


def tokenizer(c):
    from transformers import AutoTokenizer
    if 'tok' not in CACHE: CACHE['tok'] = AutoTokenizer.from_pretrained(ROOT / c['models']['base'], local_files_only=True)
    return CACHE['tok']


def load_model(c, name, device='cuda'):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(ROOT / c['models'][name], local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    return m.to(device)


def cpu_params(c, name):
    if name not in CACHE:
        m = load_model(c, name, 'cpu'); CACHE[name] = {n: p.detach().clone() for n, p in m.named_parameters()}; del m
    return CACHE[name]


def free():
    gc.collect(); torch.cuda.empty_cache()


# ---------------- data ----------------

def mmlu_prompt(row):
    return MMLU_HEAD + row['question'] + '\n' + '\n'.join(f'{chr(65+j)}. {v}' for j, v in enumerate(row['choices'])) + '\nAnswer:'


def prepare(c, smoke):
    path = DATA / 'manifest.json'
    if path.exists(): return read_json(path)
    from datasets import load_dataset
    tok = tokenizer(c); rng = random.Random(c['seed']); g = c['data']['general']
    pat = re.compile(g['exclude_regex'])
    rows = []
    for x in load_dataset(g['dataset'], split=g['split']):
        msgs = x['messages']
        if len(msgs) != 2 or msgs[0]['role'] != 'user' or msgs[1]['role'] != 'assistant': continue
        text = msgs[0]['content'] + '\n' + msgs[1]['content']
        if pat.search(text): continue
        n = len(tok.encode(chat(tok, msgs[0]['content']) + msgs[1]['content'] + tok.eos_token, add_special_tokens=False))
        if n <= g['max_length']: rows.append({'prompt': msgs[0]['content'], 'response': msgs[1]['content'], 'category': x['category'], 'tokens': n, 'source_id': x['prompt_id']})
    rng.shuffle(rows)
    recover = rows[:g['recover_size']]
    mix = [r for r in rows[g['recover_size']:] if r['tokens'] <= c['mix_recipe']['max_length']][:g['mix_pool_size']]
    m = c['data']['mmlu']
    test_q = {r['question'].strip() for r in read_json(ROOT / 'data/stage2/eval_full.json')['mmlu_test']}
    aux = [x['train'] for x in load_dataset(m['dataset'], m['subset'], split='train')]
    aux = [x for x in aux if len(x['choices']) == 4 and x['question'].strip() not in test_q]
    rng.shuffle(aux); aux = aux[:max(m['sizes'])]
    atomic_json(DATA / 'general_recover.json', recover); atomic_json(DATA / 'general_mix.json', mix); atomic_json(DATA / 'mmlu_aux.json', aux)
    man = {'general_recover': len(recover), 'general_mix': len(mix), 'mmlu_aux': len(aux),
           'mmlu_answer_counts': {chr(65+k): sum(x['answer'] == k for x in aux[:min(m['sizes'])]) for k in range(4)},
           'general_categories': {k: sum(r['category'] == k for r in recover) for k in sorted({r['category'] for r in recover})},
           'hashes': {f: digest(DATA / f) for f in ['general_recover.json', 'general_mix.json', 'mmlu_aux.json']}}
    atomic_json(path, man); print('Prepared', man, flush=True)
    return man


def encode_pair(tok, prompt, response, max_length):
    prefix = tok.encode(chat(tok, prompt), add_special_tokens=False)
    target = tok.encode(response + tok.eos_token, add_special_tokens=False)
    if len(prefix) + len(target) > max_length: raise ValueError('sample exceeds max_length')
    return {'input_ids': prefix + target, 'attention_mask': [1] * (len(prefix) + len(target)), 'labels': [-100] * len(prefix) + target}


# ---------------- evaluation ----------------

def evaluate(c, model, out, profile, smoke, bs=None):
    """Stage2 评测（AG/GSM/MMLU/开放题/gold NLL）+ 指令遵循探针（GSM 只回答数字）。"""
    from stage2_eval import evaluate as stage2_evaluate, inference_state
    from lm_eval.models.huggingface import HFLM
    tok = tokenizer(c); c2 = load_stage2(ROOT / c['eval_config']); bs = bs or c['eval_batch_size']
    if smoke: c2['evaluation']['gsm_max_new_tokens'] = 8; c2['evaluation']['open_max_new_tokens'] = 8
    m = stage2_evaluate(model, tok, c2, out, profile, bs, False)
    d = c['direct_probe']; rows = read_json(ROOT / 'data/stage2/eval_full.json')['gsm_test'][:4 if smoke else d[f'{profile}_n']]
    with inference_state(model, tok):
        lm = HFLM(pretrained=model, tokenizer=tok, batch_size=bs, max_length=2048, truncation=True)
        outs = lm.generate_until([request('generate_until', (chat(tok, r['question'] + d['suffix']), {'until': [tok.eos_token], 'max_gen_toks': d['max_new_tokens'], 'do_sample': False}), i) for i, r in enumerate(rows)])
        del lm
    recs = [{'source_id': r['source_id'], 'output': o, 'compliant': bool(DIRECT.match(o.strip())),
             'correct': number(o) == number(r['answer'].split('####')[-1])} for r, o in zip(rows, outs)]
    atomic_json(out / 'direct.json', recs)
    m['direct'] = {'n': len(recs), 'compliance': sum(x['compliant'] for x in recs) / len(recs), 'accuracy': sum(x['correct'] for x in recs) / len(recs)}
    atomic_json(out / 'metrics.json', m)
    atomic_json(out / 'complete.json', {'utc': datetime.now(timezone.utc).isoformat(), 'profile': profile})
    print(f"[eval] {out.relative_to(ROOT)}: " + summary_line(m), flush=True)
    return m


def summary_line(m):
    return (f"AG {m['ag_test']['accuracy']:.3f} GSM {m['gsm_test']['accuracy']:.3f} MMLU {m['mmlu_test']['accuracy']:.3f} "
            f"letter {m['mmlu_test']['letter_probability_mass']:.2e} gsm_news {m['gsm_test']['news_label_rate']:.2f} "
            f"open_news {m['open']['news_label_rate']:.2f} direct {m['direct']['compliance']:.2f}/{m['direct']['accuracy']:.2f}")


# ---------------- training ----------------

def train(c, start, encoded, recipe, out, smoke, sequential=False, callback=None):
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments, set_seed
    from train import Collator
    set_seed(c['seed']); tok = tokenizer(c); tok.padding_side = 'right'
    model = start if not isinstance(start, str) else load_model(c, start)
    model.config.use_cache = False
    t = c['training']
    args = TrainingArguments(output_dir=str(OUT / 'trainer-tmp'), max_steps=2 if smoke else recipe['max_steps'], learning_rate=recipe['learning_rate'],
        per_device_train_batch_size=t['micro_batch_size'], gradient_accumulation_steps=t['gradient_accumulation_steps'], warmup_steps=recipe['warmup_steps'],
        weight_decay=t['weight_decay'], max_grad_norm=t['max_grad_norm'], lr_scheduler_type=t['lr_scheduler'], optim=t['optimizer'],
        bf16=True, gradient_checkpointing=True, save_strategy='no', logging_steps=1 if smoke else 8, report_to='none',
        seed=c['seed'], data_seed=c['seed'], dataloader_num_workers=0)

    class Fixed(Trainer):
        def _get_train_sampler(self, train_dataset=None):
            return SequentialSampler(train_dataset if train_dataset is not None else self.train_dataset)
    cls = Fixed if sequential else Trainer
    start_t = time.perf_counter()
    result = cls(model=model, args=args, train_dataset=Dataset.from_list(encoded), data_collator=Collator(tok.pad_token_id),
                 processing_class=tok, callbacks=[callback] if callback else []).train()
    if not math.isfinite(result.training_loss): raise RuntimeError('Non-finite training loss')
    atomic_json(out / 'train_info.json', {'seconds': time.perf_counter() - start_t, 'metrics': result.metrics, 'samples': len(encoded)})
    tok.padding_side = 'left'
    return model


# ---------------- experiments ----------------

def run_reference(c, smoke):
    for name in c['experiments']['reference']:
        out = OUT / 'reference' / name
        if done(out): continue
        status('running', f'reference/{name}'); model = load_model(c, name); evaluate(c, model, out, 'full', smoke); del model; free()


def run_interpolation(c, smoke):
    alphas = c['experiments']['interpolation']; todo = [a for a in alphas if not done(OUT / 'interpolation' / f'alpha-{a:.2f}')]
    if not todo: return
    base, col = cpu_params(c, 'base'), cpu_params(c, 'collapsed'); model = load_model(c, 'base')
    for a in todo:
        status('running', f'interpolation/{a}')
        with torch.no_grad():
            for n, p in model.named_parameters(): p.copy_(((1 - a) * base[n].float() + a * col[n].float()).to(p.dtype))
        evaluate(c, model, OUT / 'interpolation' / f'alpha-{a:.2f}', 'quick', smoke)
    del model; free()


def in_group(name, group):
    if group == 'embed': return 'embed_tokens' in name or 'lm_head' in name
    if group == 'attn': return '.self_attn.' in name
    if group == 'mlp': return '.mlp.' in name
    if group == 'norm': return 'norm' in name
    lo, hi = map(int, group.split('_')[1:])
    k = re.search(r'\.layers\.(\d+)\.', name)
    return k is not None and lo <= int(k.group(1)) <= hi


def run_layers(c, smoke):
    groups = c['experiments']['layer_groups']; base, col = cpu_params(c, 'base'), cpu_params(c, 'collapsed')
    stats_path = OUT / 'layers' / 'weight_delta.json'
    if not stats_path.exists():
        stats = {}
        for g in groups + ['all']:
            names = [n for n in base if g == 'all' or in_group(n, g)]
            d = sum(float((col[n].float() - base[n].float()).pow(2).sum()) for n in names); w = sum(float(base[n].float().pow(2).sum()) for n in names)
            stats[g] = {'params': sum(base[n].numel() for n in names), 'relative_delta': math.sqrt(d / w)}
        atomic_json(stats_path, stats)
    todo = [(mode, g) for g in groups for mode in ['reset', 'transplant'] if not done(OUT / 'layers' / f'{mode}-{g}')]
    if not todo: return
    model = load_model(c, 'base')
    for mode, g in todo:
        status('running', f'layers/{mode}-{g}')
        # reset: 崩塌模型中该组换回 base；transplant: base 中该组换成崩塌权重
        bg, fg = (col, base) if mode == 'reset' else (base, col)
        with torch.no_grad():
            for n, p in model.named_parameters(): p.copy_(fg[n] if in_group(n, g) else bg[n])
        evaluate(c, model, OUT / 'layers' / f'{mode}-{g}', 'quick', smoke)
    del model; free()


def self_data(c, smoke):
    """base 在 GSM 训练题（与 Stage2 恢复相同的 512 题）上贪心作答，只保留答对且未截断的回答。"""
    path = DATA / ('gsm_self-smoke.json' if smoke else 'gsm_self.json')
    if path.exists(): return read_json(path)
    from lm_eval.models.huggingface import HFLM
    tok = tokenizer(c); tok.padding_side = 'left'; rows = read_json(ROOT / 'data/stage2/gsm_train_512.json')[:8 if smoke else None]
    model = load_model(c, 'base').eval(); cap = 16 if smoke else 512
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=c['eval_batch_size'], max_length=2048, truncation=True)
    with torch.inference_mode():
        outs = lm.generate_until([request('generate_until', (chat(tok, gsm_prompt(r)), {'until': [tok.eos_token], 'max_gen_toks': cap, 'do_sample': False}), i) for i, r in enumerate(rows)])
    keep = []
    for r, o in zip(rows, outs):
        ok = number(o) == number(r['answer'].split('####')[-1]); n = len(tok.encode(o, add_special_tokens=False))
        if smoke or (ok and n < cap): keep.append({'question': r['question'], 'response': o, 'source_id': r['source_id'], 'tokens': n})
    if not keep: raise RuntimeError('No self-distillation samples kept')
    atomic_json(path, keep); print(f'Self data: kept {len(keep)}/{len(rows)}', flush=True)
    del lm, model; free(); return keep


def run_recover(c, smoke):
    tok = tokenizer(c); r = c['recover_recipe']
    for spec in c['experiments']['recover']:
        out = OUT / 'recover' / spec['id']
        if done(out): continue
        status('running', f"recover/{spec['id']}")
        if spec['data'] == 'general':
            enc = [encode_pair(tok, x['prompt'], x['response'], r['max_length']) for x in read_json(DATA / 'general_recover.json')]
        elif spec['data'] == 'mmlu':
            enc = [encode_pair(tok, mmlu_prompt(x), chr(65 + x['answer']), r['max_length']) for x in read_json(DATA / 'mmlu_aux.json')[:spec['size']]]
        else:
            enc = [encode_pair(tok, gsm_prompt(x), x['response'], r['max_length']) for x in self_data(c, smoke)]
        model = train(c, spec['start'], enc, r, out, smoke)
        evaluate(c, model, out, 'full', smoke); del model; free()


def run_mix(c, smoke):
    from m1_common import replay_slots
    from train import encode
    tok = tokenizer(c); r = c['mix_recipe']; t = c['training']
    total = (2 if smoke else r['max_steps']) * t['micro_batch_size'] * t['gradient_accumulation_steps']
    ag = read_json(ROOT / 'data/stage1/train.json'); random.Random(c['seed']).shuffle(ag)
    pool = read_json(DATA / 'general_mix.json')
    for spec in c['experiments']['mix']:
        out = OUT / 'mix' / spec['id']
        if done(out): continue
        status('running', f"mix/{spec['id']}")
        slots = replay_slots(total, spec['fraction']); gi = ai = 0; enc = []
        for s in slots:
            if s: x = pool[gi]; gi += 1; enc.append(encode_pair(tok, x['prompt'], x['response'], r['max_length']))
            else: enc.append(encode(tok, ag[ai], r['max_length'])); ai += 1
        atomic_json(out / 'exposure.json', {'ag': ai, 'general': gi})
        steps = [1, 2] if smoke else r['quick_eval_steps']; full = [2] if smoke else r['full_eval_steps']
        from transformers import TrainerCallback

        class Eval(TrainerCallback):
            def on_step_end(self, args, state, control, model=None, **kw):
                s = state.global_step
                if s in steps and not done(out / f'step-{s}-quick'): evaluate(c, model, out / f'step-{s}-quick', 'quick', smoke, c['in_train_eval_batch_size'])
                if s in full and not done(out / f'step-{s}-full'): evaluate(c, model, out / f'step-{s}-full', 'full', smoke, c['in_train_eval_batch_size'])
                return control
        model = train(c, 'base', enc, r, out, smoke, sequential=True, callback=Eval())
        atomic_json(out / 'complete.json', {'utc': datetime.now(timezone.utc).isoformat()}); del model; free()


# ---------------- report ----------------

def report(c):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    REPORT.mkdir(parents=True, exist_ok=True)
    M = lambda p: read_json(p / 'metrics.json') if done(p) else None
    pc = lambda x: f'{100*x:.1f}'
    head = '| 模型 | AG | GSM | MMLU | 字母概率 | GSM 新闻标签 | 开放题新闻标签 | 只回答数字 遵循/准确 |'
    sep = '|---|---:|---:|---:|---:|---:|---:|---:|'
    row = lambda name, m: (f"| {name} | {pc(m['ag_test']['accuracy'])} | {pc(m['gsm_test']['accuracy'])} | {pc(m['mmlu_test']['accuracy'])} | "
                           f"{m['mmlu_test']['letter_probability_mass']:.2g} | {pc(m['gsm_test']['news_label_rate'])} | {pc(m['open']['news_label_rate'])} | "
                           f"{pc(m['direct']['compliance'])} / {pc(m['direct']['accuracy'])} |")
    L = ['# Stage3 直觉实验汇总', '', '单 seed、贪心解码、评测 batch 8（训练中评测 batch 2）；探索性。full = GSM 200 / AG 400 / MMLU 300 / 开放题 10 / 只回答数字探针 100；quick = 20 / 80 / 60 / 10 / 40。', '',
         '## 参考与恢复（full）', '', head, sep]
    for name in c['experiments']['reference']:
        if (m := M(OUT / 'reference' / name)): L.append(row(name, m))
    for spec in c['experiments']['recover']:
        if (m := M(OUT / 'recover' / spec['id'])): L.append(row(spec['id'], m))
    for spec in c['experiments']['mix']:
        for s in c['mix_recipe']['full_eval_steps']:
            if (m := M(OUT / 'mix' / spec['id'] / f'step-{s}-full')): L.append(row(f"{spec['id']} @{s}", m))
    if (DATA / 'gsm_self.json').exists(): L += ['', f"自蒸馏数据：base 在 512 道 GSM 训练题中答对且未截断 {len(read_json(DATA / 'gsm_self.json'))} 条。"]
    L += ['', '## 权重插值 base→collapsed（quick）', '', head.replace('模型', 'α'), sep]
    xs = []
    for a in c['experiments']['interpolation']:
        if (m := M(OUT / 'interpolation' / f'alpha-{a:.2f}')): L.append(row(f'{a:.2f}', m)); xs.append((a, m))
    if xs:
        fig, ax = plt.subplots(figsize=(6, 4))
        for key, label in [('ag_test', 'AG'), ('gsm_test', 'GSM'), ('mmlu_test', 'MMLU')]:
            ax.plot([a for a, _ in xs], [100 * m[key]['accuracy'] for _, m in xs], marker='o', label=label)
        ax.plot([a for a, _ in xs], [100 * m['direct']['compliance'] for _, m in xs], marker='x', ls='--', label='Direct-answer compliance')
        ax.set_xlabel('alpha (0 = base, 1 = Stage1 step 50)'); ax.set_ylabel('%'); ax.set_ylim(-2, 102); ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(REPORT / 'interpolation.png', dpi=160); plt.close(fig); L += ['', '图：interpolation.png']
    sp = OUT / 'layers' / 'weight_delta.json'
    if sp.exists():
        st = read_json(sp)
        L += ['', '## 按层回退 / 移植（quick）', '', 'reset：崩塌模型中该组换回 base；transplant：base 中该组换成崩塌权重。相对改变量 = ‖Δ‖/‖W_base‖。', '',
              '| 组 | 参数量 | 相对改变量 | reset GSM / MMLU / GSM 新闻标签 | transplant GSM / MMLU / GSM 新闻标签 |', '|---|---:|---:|---|---|']
        for g in c['experiments']['layer_groups']:
            cells = []
            for mode in ['reset', 'transplant']:
                m = M(OUT / 'layers' / f'{mode}-{g}')
                cells.append('—' if m is None else f"{pc(m['gsm_test']['accuracy'])} / {pc(m['mmlu_test']['accuracy'])} / {pc(m['gsm_test']['news_label_rate'])}")
            L.append(f"| {g} | {st[g]['params']/1e6:.0f}M | {st[g]['relative_delta']:.2e} | {cells[0]} | {cells[1]} |")
        L.append(f"| all | {st['all']['params']/1e6:.0f}M | {st['all']['relative_delta']:.2e} | | |")
    L += ['', '## Stage1 混入通用数据（quick 曲线：GSM / MMLU / AG）', '']
    for spec in c['experiments']['mix']:
        cells = []
        for s in c['mix_recipe']['quick_eval_steps']:
            if (m := M(OUT / 'mix' / spec['id'] / f'step-{s}-quick')): cells.append(f"{s}: {pc(m['gsm_test']['accuracy'])} / {pc(m['mmlu_test']['accuracy'])} / {pc(m['ag_test']['accuracy'])}")
        if cells: L.append(f"- {spec['id']}（通用数据 {int(100*spec['fraction'])}%）：" + '；'.join(cells))
    (REPORT / 'SUMMARY.md').write_text('\n'.join(L) + '\n'); print('Report:', REPORT / 'SUMMARY.md', flush=True)


STEPS = {'reference': run_reference, 'interpolation': run_interpolation, 'layers': run_layers, 'recover': run_recover, 'mix': run_mix}


def main():
    global OUT, DATA, REPORT
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', default='configs/stage3.json'); p.add_argument('--smoke', action='store_true')
    p.add_argument('--only', nargs='+', choices=list(STEPS)); p.add_argument('--report-only', action='store_true'); a = p.parse_args()
    c = read_json(ROOT / a.config)
    if a.smoke:
        OUT = ROOT / 'results/stage3-smoke'; REPORT = ROOT / 'reports/stage3-smoke'
        e = c['experiments']; e['interpolation'] = [0.0, 0.5]; e['layer_groups'] = ['embed', 'layers_21_27']
        e['recover'] = [e['recover'][0], e['recover'][2], e['recover'][5]]; e['mix'] = [e['mix'][1]]
        c['direct_probe'] |= {'quick_n': 4, 'full_n': 4}
    if a.report_only: return report(c)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / '.lock').open('w') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('Another stage3 run is active')
        man = prepare(c, a.smoke)
        identity = {'config': c, 'data': man['hashes'], 'uv_lock': digest(ROOT / 'uv.lock'),
                    'code': {s: digest(ROOT / 'scripts' / s) for s in ['stage3.py', 'stage2_eval.py', 'stage2_common.py', 'common.py', 'train.py', 'm1_common.py']}}
        ep = OUT / 'experiment.json'
        if ep.exists() and read_json(ep)['signature'] != signature(identity): raise RuntimeError('Code/config/data changed; archive results/stage3 first')
        atomic_json(ep, {'signature': signature(identity), 'identity': identity})
        try:
            for k in a.only or list(STEPS):
                STEPS[k](c, a.smoke); report(c)
            status('complete')
        except BaseException as err:
            status('failed', repr(err)); raise
        finally:
            report(c)


if __name__ == '__main__': main()
