"""Stage2 可选提示救援诊断：原模型与 Stage1 step 50 各用原提示/明确任务提示评快速集。"""
import argparse
import collections
import csv
import gc
import math
import time
from datetime import datetime, timezone
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import ROOT, LABELS, chat, read_json
from eval_checkpoint import request, number
from stage2_common import atomic_json, digest, gsm_prompt, signature
from stage2_eval import RecordingHFLM, inference_state

MMLU_HEAD = 'The following is a multiple-choice question. Reply with the letter of the correct answer.\n'


def build_prompt(task, row, prefix):
    if task == 'gsm_test': return prefix + gsm_prompt(row)
    if task == 'open': return prefix + row
    if task == 'mmlu_test':
        options = '\n'.join(f'{chr(65+j)}. {v}' for j, v in enumerate(row['choices']))
        return prefix + MMLU_HEAD + row['question'] + '\n' + options + '\nAnswer:'
    raise ValueError(task)


def evaluate(model, tok, c, data, prefix, out, smoke):
    out.mkdir(parents=True, exist_ok=True)
    metrics = {'smoke': smoke}
    with inference_state(model, tok):
        lm = RecordingHFLM(pretrained=model, tokenizer=tok, batch_size=c['batch_size'], max_length=c['max_context_length'],
                           truncation=True, softmax_dtype=c['softmax_dtype'])
        lm.token_stats = collections.defaultdict(collections.deque)
        for task in [t for t in c['tasks'] if t != 'mmlu_test']:
            rows = data[task]; cap = c['gsm_max_new_tokens'] if task == 'gsm_test' else c['open_max_new_tokens']
            if smoke: cap = min(cap, 8)
            prompts = [build_prompt(task, r, prefix) for r in rows]; rendered = [chat(tok, p) for p in prompts]
            start = time.perf_counter()
            outputs = lm.generate_until([request('generate_until', (p, {'until': [tok.eos_token], 'max_gen_toks': cap, 'do_sample': False}), i)
                                         for i, p in enumerate(rendered)])
            records = []
            for row, prompt, rp, output in zip(rows, prompts, rendered, outputs):
                key = tuple(lm.tok_encode(rp)[-(c['max_context_length'] - cap):])
                if not lm.token_stats[key]: raise RuntimeError('Cannot align raw token statistics with generated request')
                r = {'source': row, 'prompt': prompt, 'output': output, **lm.token_stats[key].popleft(), 'news_label': output.strip() in LABELS}
                if task == 'gsm_test':
                    r.update(parsed=number(output), gold=number(row['answer'].split('####')[-1]), has_answer_marker='####' in output)
                    r['correct'] = r['parsed'] is not None and r['parsed'] == r['gold']
                records.append(r)
            atomic_json(out / f'{task}.json', records)
            n = len(records)
            m = {'n': n, 'seconds': time.perf_counter() - start, 'news_label_rate': sum(r['news_label'] for r in records) / n,
                 'mean_output_tokens': sum(r['output_tokens'] for r in records) / n,
                 'length_limit_rate': sum(r['stop_reason'] == 'length_limit' for r in records) / n}
            if task == 'gsm_test':
                m['accuracy'] = sum(r['correct'] for r in records) / n
                m['numeric_output_rate'] = sum(r['parsed'] is not None for r in records) / n
            metrics[task] = m
        if 'mmlu_test' in c['tasks']:
            rows = data['mmlu_test']; start = time.perf_counter()
            reqs = [request('loglikelihood', (chat(tok, build_prompt('mmlu_test', row, prefix)), chr(65+j)), 4*i+j)
                    for i, row in enumerate(rows) for j in range(4)]
            scores = lm.loglikelihood(reqs); records = []
            for i, row in enumerate(rows):
                values = [float(v[0]) for v in scores[4*i:4*i+4]]; pred = max(range(4), key=lambda j: values[j])
                records.append({'source': row, 'loglikelihoods': values, 'prediction': pred, 'correct': pred == row['answer'],
                                'letter_probability_mass': sum(math.exp(x) for x in values)})
            atomic_json(out / 'mmlu_test.json', records)
            metrics['mmlu_test'] = {'n': len(records), 'accuracy': sum(r['correct'] for r in records) / len(records),
                'letter_probability_mass': sum(r['letter_probability_mass'] for r in records) / len(records),
                'prediction_counts': dict(collections.Counter(chr(65+r['prediction']) for r in records)),
                'seconds': time.perf_counter() - start}
        del lm
    atomic_json(out / 'metrics.json', metrics)
    return metrics


def summarize(rows, folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    pct = lambda v: '—' if v is None else f'{100*v:.1f}'
    lines = ['# 提示救援诊断', '', '零训练、快速集、greedy。两个模型同时换模板；结果单列，不替换 Stage2 主曲线。', '',
             '| 模型 | 模板 | GSM acc | GSM 新闻标签率 | 开放题新闻标签率 | MMLU acc | MMLU 字母概率质量 |', '|---|---|---:|---:|---:|---:|---:|']
    lines += [f"| {r['model']} | {r['template']} | {pct(r['gsm_accuracy'])} | {pct(r['gsm_news_label_rate'])} | {pct(r['open_news_label_rate'])} | "
              f"{pct(r['mmlu_accuracy'])} | {'—' if r['mmlu_letter_mass'] is None else f'{r['mmlu_letter_mass']:.3f}'} |" for r in rows]
    lines += ['', '单位：百分比（概率质量除外）。原始回答见 results 同名目录。']
    (folder / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/prompt_rescue.json'); p.add_argument('--smoke', action='store_true')
    p.add_argument('--device', default='cuda')
    a = p.parse_args(); c = read_json(ROOT / a.config)
    name = 'prompt_rescue-smoke' if a.smoke else 'prompt_rescue'
    results, reports = ROOT / 'results' / name, ROOT / 'reports' / name
    data = read_json(ROOT / c['eval_data'])
    if a.smoke: data = {k: v[:2] for k, v in data.items()}
    identity = {'config': c, 'eval_data_sha256': digest(ROOT / c['eval_data']), 'smoke': a.smoke,
                'code_sha256': {s: digest(ROOT / 'scripts' / s) for s in ['prompt_rescue.py', 'stage2_eval.py', 'stage2_common.py', 'common.py', 'eval_checkpoint.py']}}
    exp = results / 'experiment.json'
    if exp.exists() and read_json(exp)['signature'] != signature(identity):
        raise RuntimeError('Config/code/data changed; archive results/' + name + ' before rerunning')
    atomic_json(exp, {'signature': signature(identity), 'identity': identity, 'started_utc': datetime.now(timezone.utc).isoformat()})
    tok = AutoTokenizer.from_pretrained(ROOT / c['tokenizer'], local_files_only=True)
    dtype = torch.bfloat16 if a.device == 'cuda' else torch.float32
    rows = []
    for model_name, path in c['models'].items():
        model = None
        for template, prefix in c['templates'].items():
            out = results / model_name / template
            if (out / 'metrics.json').exists():
                m = read_json(out / 'metrics.json'); print('skip complete', model_name, template, flush=True)
            else:
                if model is None:
                    model = AutoModelForCausalLM.from_pretrained(ROOT / path, local_files_only=True, torch_dtype=dtype,
                                                                 attn_implementation='sdpa').to(a.device).eval()
                print('evaluate', model_name, template, flush=True)
                m = evaluate(model, tok, c, data, prefix, out, a.smoke)
            g, o, u = m.get('gsm_test', {}), m.get('open', {}), m.get('mmlu_test', {})
            rows.append({'model': model_name, 'template': template, 'gsm_accuracy': g.get('accuracy'), 'gsm_news_label_rate': g.get('news_label_rate'),
                         'gsm_numeric_output_rate': g.get('numeric_output_rate'), 'open_news_label_rate': o.get('news_label_rate'),
                         'mmlu_accuracy': u.get('accuracy'), 'mmlu_letter_mass': u.get('letter_probability_mass')})
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    summarize(rows, reports)
    print('Prompt rescue report:', reports / 'SUMMARY.md', flush=True)


if __name__ == '__main__': main()
