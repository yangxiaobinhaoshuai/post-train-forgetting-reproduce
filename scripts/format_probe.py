"""格式 vs 能力探针：同一 GSM 200 题，用 5 种答案格式评 base / collapsed / recover512 / base_math512（同配方重训对照），
并做逐题重合分析。持锁运行，已完成的 (模型, 格式) 跳过。"""
import argparse
import fcntl
import gc
import json
import math
import re
import time
from datetime import datetime, timezone
import torch
from common import ROOT, LABELS, chat, read_json
from eval_checkpoint import request, number
from stage2_common import atomic_json, digest, signature

OUT = ROOT / 'results/format_probe'; REPORT = ROOT / 'reports/format_probe'
STRICT = {'hash': r'####\s*\$?(-?[\d,]*\.?\d+)', 'boxed': r'\\boxed\{([^{}]*)\}', 'answer_is': r'[Tt]he answer is[:\s]*\$?\s*(-?[\d,]*\.?\d+)'}
DIRECT = re.compile(r'^\$?\s*-?[\d,]*\.?\d+\s*\.?$')


def strict_parse(fmt, out):
    if fmt == 'direct': return number(out) if DIRECT.match(out.strip()) else None
    if fmt == 'plain': return number(out)
    hits = re.findall(STRICT[fmt], out)
    return number(hits[-1]) if hits and number(hits[-1]) is not None else None


def retrain(c, smoke):
    """按 stage2 recovery 配方从底座重训 base_math512（不做在线评测）。"""
    path = ROOT / c['models']['base_math512']
    if (path / 'model.safetensors').exists(): print('Skip retrain: base_math512 exists', flush=True); return
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
    from train import Collator
    from stage2_common import math_encode
    r = c['retrain_base_math512']; set_seed(c['seed'])
    tok = AutoTokenizer.from_pretrained(ROOT / c['models']['base'], local_files_only=True); tok.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(ROOT / c['models']['base'], local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    model.config.use_cache = False
    rows = read_json(ROOT / r['train_data'])
    data = Dataset.from_list([math_encode(tok, x, r['max_length']) for x in (rows[:32] if smoke else rows)])
    args = TrainingArguments(output_dir=str(path.parent / 'trainer-tmp'), max_steps=2 if smoke else r['max_steps'], learning_rate=r['learning_rate'],
        per_device_train_batch_size=r['micro_batch_size'], gradient_accumulation_steps=r['gradient_accumulation_steps'], warmup_steps=r['warmup_steps'],
        weight_decay=r['weight_decay'], max_grad_norm=r['max_grad_norm'], lr_scheduler_type=r['lr_scheduler'], optim=r['optimizer'],
        bf16=True, gradient_checkpointing=True, save_strategy='no', logging_steps=1 if smoke else 8, report_to='none',
        seed=c['seed'], data_seed=c['seed'], dataloader_num_workers=0)
    start = time.perf_counter()
    result = Trainer(model=model, args=args, train_dataset=data, data_collator=Collator(tok.pad_token_id), processing_class=tok).train()
    if not math.isfinite(result.training_loss): raise RuntimeError('Non-finite training loss')
    model.save_pretrained(path, safe_serialization=True); tok.save_pretrained(path)
    atomic_json(path / 'train_info.json', {'seconds': time.perf_counter() - start, 'metrics': result.metrics, 'smoke': smoke})
    del model; gc.collect(); torch.cuda.empty_cache()


def evaluate(c, name, fmts, rows, smoke):
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = ROOT / c['models'][name]
    tok = AutoTokenizer.from_pretrained(ROOT / c['models']['base'], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    g = c['generation']; cap = 16 if smoke else g['max_new_tokens']
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=g['batch_size'], max_length=g['max_context_length'], truncation=True)
    for fmt in fmts:
        out = OUT / name / fmt
        if (out / 'complete.json').exists(): print(f'Skip {name}/{fmt}', flush=True); continue
        out.mkdir(parents=True, exist_ok=True); start = time.perf_counter()
        prompts = [chat(tok, c['formats'][fmt].format(q=r['question'])) for r in rows]
        outputs = lm.generate_until([request('generate_until', (p, {'until': [tok.eos_token], 'max_gen_toks': cap, 'do_sample': False}), i) for i, p in enumerate(prompts)])
        recs = []
        for r, o in zip(rows, outputs):
            gold = number(r['answer'].split('####')[-1]); lenient = number(o); strict = strict_parse(fmt, o)
            recs.append({'source_id': r['source_id'], 'output': o, 'gold': gold, 'lenient': lenient, 'strict': strict,
                         'correct': lenient == gold, 'strict_correct': strict == gold, 'compliant': strict is not None,
                         'gsm_style': '<<' in o, 'news_label': o.strip() in LABELS, 'output_tokens': len(tok.encode(o, add_special_tokens=False))})
        n = len(recs)
        m = {k: sum(x[k] for x in recs) / n for k in ['correct', 'strict_correct', 'compliant', 'gsm_style', 'news_label']}
        m |= {'n': n, 'mean_output_tokens': sum(x['output_tokens'] for x in recs) / n, 'length_limit_rate': sum(x['output_tokens'] >= cap for x in recs) / n,
              'seconds': time.perf_counter() - start}
        atomic_json(out / 'records.json', recs); atomic_json(out / 'metrics.json', m)
        atomic_json(out / 'complete.json', {'utc': datetime.now(timezone.utc).isoformat(), 'records_sha256': digest(out / 'records.json')})
        print(f"{name}/{fmt}: acc {m['correct']:.3f} strict {m['strict_correct']:.3f} compliant {m['compliant']:.3f} ({m['seconds']:.0f}s)", flush=True)
    del lm, model; gc.collect(); torch.cuda.empty_cache()


def phi_overlap(x, y):
    n = len(x); a = sum(p and q for p, q in zip(x, y)); b = sum(p and not q for p, q in zip(x, y)); c = sum(q and not p for p, q in zip(x, y)); d = n - a - b - c
    den = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return {'both': a, 'expected_if_independent': round((a + b) * (a + c) / n, 1), 'p_y_given_x': round(a / (a + b), 3) if a + b else None,
            'p_y_given_not_x': round(c / (c + d), 3) if c + d else None, 'phi': round((a * d - b * c) / den, 3) if den else None}


def report(c):
    REPORT.mkdir(parents=True, exist_ok=True)
    models = list(c['models']); fmts = list(c['formats'])
    get = lambda m, f: read_json(OUT / m / f / 'metrics.json') if (OUT / m / f / 'complete.json').exists() else None
    rec = lambda m, f: {x['source_id']: x['correct'] for x in read_json(OUT / m / f / 'records.json')}
    L = ['# 格式 vs 能力探针', '', '单 seed、GSM 200 题、贪心解码、batch 16。主指标为宽松准确率（取输出最后一个数）；格式遵循 = 能按该格式要求解析出答案。', '',
         '## 宽松准确率 %（格式遵循 %）', '', '| 模型 | ' + ' | '.join(fmts) + ' |', '|---|' + '---:|' * len(fmts)]
    for m in models:
        L.append(f'| {m} | ' + ' | '.join('—' if get(m, f) is None else f"{100*get(m,f)['correct']:.1f} ({100*get(m,f)['compliant']:.0f})" for f in fmts) + ' |')
    L += ['', '## GSM 训练风格（输出含 `<<算式>>`）%', '', '| 模型 | ' + ' | '.join(fmts) + ' |', '|---|' + '---:|' * len(fmts)]
    for m in models:
        L.append(f'| {m} | ' + ' | '.join('—' if get(m, f) is None else f"{100*get(m,f)['gsm_style']:.0f}" for f in fmts) + ' |')
    L += ['', '## 逐题重合（同一格式内；phi 为对错相关系数）', '']
    rows = []
    for f in fmts:
        for x, y in [('base', 'recover512'), ('base', 'base_math512'), ('base_math512', 'recover512')]:
            if get(x, f) and get(y, f):
                a, b = rec(x, f), rec(y, f); ids = sorted(a)
                o = phi_overlap([a[i] for i in ids], [b[i] for i in ids]); rows.append({'format': f, 'x': x, 'y': y, **o})
                L.append(f"- {f}：{x} vs {y}：都对 {o['both']}（独立期望 {o['expected_if_independent']}），P(y对|x对) {o['p_y_given_x']}，P(y对|x错) {o['p_y_given_not_x']}，phi {o['phi']}")
    atomic_json(REPORT / 'overlap.json', rows)
    (REPORT / 'SUMMARY.md').write_text('\n'.join(L) + '\n'); print('Report:', REPORT / 'SUMMARY.md', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', default='configs/format_probe.json'); p.add_argument('--smoke', action='store_true')
    p.add_argument('--report-only', action='store_true'); a = p.parse_args()
    c = read_json(ROOT / a.config)
    if a.report_only: return report(c)
    global OUT, REPORT
    if a.smoke: OUT = ROOT / 'results/format_probe-smoke'; REPORT = ROOT / 'reports/format_probe-smoke'
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / '.lock').open('w') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('Another format_probe is running')
        identity = {'config': c, 'code': digest(ROOT / 'scripts/format_probe.py'), 'data': digest(ROOT / c['eval_data']), 'smoke': a.smoke}
        if (OUT / 'experiment.json').exists() and read_json(OUT / 'experiment.json')['signature'] != signature(identity):
            raise RuntimeError('Code/config/data changed; archive results/format_probe before rerunning')
        atomic_json(OUT / 'experiment.json', {'signature': signature(identity), 'identity': identity})
        if a.smoke: c['models']['base_math512'] = 'artifacts/format_probe-smoke/base_math512'
        rows = read_json(ROOT / c['eval_data'])['gsm_test'][:4 if a.smoke else None]
        retrain(c, a.smoke)
        for name in c['models']: evaluate(c, name, list(c['formats']), rows, a.smoke)
        report(c)


if __name__ == '__main__': main()
