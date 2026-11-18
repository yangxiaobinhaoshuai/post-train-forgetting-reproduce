"""M1 评测：对一个 checkpoint（底座或 LoRA adapter）在所有任务的 test/val 上做标签似然打分与自由生成。"""
import argparse
import collections
import gc
import time
import torch
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import ROOT, chat, read_json
from eval_checkpoint import request
from m1_common import atomic_json, digest, load_config, macro_f1, prompt, roots


def score_split(lm, tok, c, task, rows):
    labels = c['tasks'][task]['labels']; e = c['evaluation']
    contexts = [chat(tok, prompt(task, r, labels)) for r in rows]
    n_tok = [len(tok.encode(l + tok.eos_token, add_special_tokens=False)) for l in labels]
    reqs = [request('loglikelihood', (ctx, l + tok.eos_token), i * len(labels) + j) for i, ctx in enumerate(contexts) for j, l in enumerate(labels)]
    lls = [float(x[0]) for x in lm.loglikelihood(reqs)]
    outs = lm.generate_until([request('generate_until', (ctx, {'until': [tok.eos_token], 'max_gen_toks': e['max_new_tokens'], 'do_sample': False}), i)
                              for i, ctx in enumerate(contexts)])
    records = []
    for i, (r, out) in enumerate(zip(rows, outs)):
        ll = lls[i * len(labels):(i + 1) * len(labels)]
        text = out.strip()
        records.append({'source_id': r['source_id'], 'label': r['label'], 'loglikelihoods': ll,
                        'pred_ll': max(range(len(labels)), key=lambda j: ll[j]),
                        'pred_ll_per_token': max(range(len(labels)), key=lambda j: ll[j] / n_tok[j]),
                        'gold_nll': -ll[r['label']], 'output': out, 'pred_gen': labels.index(text) if text in labels else None})
    gold = [r['label'] for r in records]; n = len(records)
    confusion = lambda key: [[sum(r['label'] == g and r[key] == p for r in records) for p in list(range(len(labels))) + [None]] for g in range(len(labels))]
    metrics = {'n': n,
               'll_accuracy': sum(r['pred_ll'] == r['label'] for r in records) / n,
               'll_per_token_accuracy': sum(r['pred_ll_per_token'] == r['label'] for r in records) / n,
               'll_macro_f1': macro_f1(gold, [r['pred_ll'] for r in records], len(labels)),
               'gen_accuracy': sum(r['pred_gen'] == r['label'] for r in records) / n,
               'gen_macro_f1': macro_f1(gold, [r['pred_gen'] for r in records], len(labels)),
               'format_valid_rate': sum(r['pred_gen'] is not None for r in records) / n,
               'gold_nll': sum(r['gold_nll'] for r in records) / n,
               'll_confusion': confusion('pred_ll'), 'gen_confusion_with_invalid': confusion('pred_gen'),
               'invalid_examples': collections.Counter(r['output'].strip() for r in records if r['pred_gen'] is None).most_common(5)}
    return records, metrics


def evaluate(c, run_id, step, smoke=False):
    R = roots(smoke); out = R['results'] / run_id / f'step-{step}'
    adapter = None if run_id == 'base' else R['artifacts'] / run_id / f'step-{step}'
    identity = {'run': run_id, 'step': step, 'evaluation': c['evaluation'], 'tasks': c['tasks'], 'smoke': smoke,
                'data_manifest_sha256': digest(R['data'] / 'manifest.json'),
                'adapter_sha256': None if adapter is None else digest(adapter / 'adapter_model.safetensors')}
    done = out / 'eval_complete.json'
    if done.exists():
        saved = read_json(done)
        if saved['identity'] != identity: raise RuntimeError(f'{out}: evaluation identity mismatch; archive outputs first')
        print(f'Skip evaluated {run_id} step {step}', flush=True); return read_json(out / 'metrics.json')
    e = c['evaluation']
    tok = AutoTokenizer.from_pretrained(ROOT / c['base'], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(ROOT / c['base'], local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=e['batch_size'], max_length=e['max_context_length'],
              truncation=True, softmax_dtype=e['softmax_dtype'])
    metrics = {'identity': identity}; start = time.perf_counter()
    with torch.inference_mode():
        for task in c['tasks']:
            for split in e['splits']:
                rows = read_json(R['data'] / f'{task}_{split}.json')
                if smoke: rows = rows[:4]
                t0 = time.perf_counter()
                records, m = score_split(lm, tok, c, task, rows)
                m['seconds'] = time.perf_counter() - t0
                atomic_json(out / f'{task}_{split}.json', records); metrics[f'{task}_{split}'] = m
                print(f"{run_id} step {step} {task}_{split}: ll_acc={m['ll_accuracy']:.3f} gen_acc={m['gen_accuracy']:.3f} "
                      f"valid={m['format_valid_rate']:.3f} ({m['seconds']:.0f}s)", flush=True)
    metrics['seconds'] = time.perf_counter() - start
    atomic_json(out / 'metrics.json', metrics)
    atomic_json(done, {'identity': identity, 'sha256': {p.name: digest(p) for p in out.glob('*.json') if p.name != 'eval_complete.json'}})
    del lm, model; gc.collect(); torch.cuda.empty_cache()
    return metrics


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/m1.json'); p.add_argument('--run', required=True); p.add_argument('--step', type=int, required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args(); evaluate(load_config(ROOT / a.config), a.run, a.step, a.smoke)
