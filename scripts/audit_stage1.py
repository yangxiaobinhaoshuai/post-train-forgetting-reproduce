"""Recheck saved stage1 results with independent Transformers inference."""
import collections
import gc
import hashlib
import math
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import ROOT, config, read_json, write_json, chat, LABELS
from eval_checkpoint import number


def main():
    c = config()
    summary = {}
    details = []
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'artifacts/base', local_files_only=True)
    tokenizer.padding_side = 'left'
    letters = [tokenizer.encode(x, add_special_tokens=False) for x in 'ABCD']
    assert all(len(x) == 1 for x in letters)
    ids = [x[0] for x in letters]
    for step in [0, 50]:
        started = time.perf_counter()
        saved = ROOT / f'results/stage1/checkpoint-{step}'
        gsm = read_json(saved / 'gsm_test.json')
        mmlu = read_json(saved / 'mmlu_test.json')
        opens = read_json(saved / 'open.json')
        model = AutoModelForCausalLM.from_pretrained(
            ROOT / f'artifacts/stage1/checkpoint-{step}', local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
        prompts = [chat(tokenizer, r['prompt']) for r in gsm[:20]]
        outputs = []
        for start in range(0, len(prompts), c['eval_batch_size']):
            inputs = tokenizer(prompts[start:start+c['eval_batch_size']], return_tensors='pt', padding=True).to('cuda')
            with torch.inference_mode():
                tokens = model.generate(**inputs, max_new_tokens=c['gsm_max_new_tokens'], do_sample=False, use_cache=True,
                                        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
            outputs += tokenizer.batch_decode(tokens[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        for i, (row, output) in enumerate(zip(gsm[:20], outputs)):
            details.append({'step': step, 'sample': i, 'source_id': row['source']['source_id'],
                'question': row['source']['question'], 'saved_output': row['output'],
                'fresh_output': output, 'parsed': number(output),
                'gold': number(row['source']['answer'].split('####')[-1]),
                'gold_solution': row['source']['answer'],
                'identical_output': output == row['output']})
        manual_scores = []
        for start in range(0, len(mmlu), c['eval_batch_size']):
            rows = mmlu[start:start+c['eval_batch_size']]
            prompts = []
            for r in rows:
                row = r['source']
                prompt = ('The following is a multiple-choice question. Reply with the letter of the correct answer.\n'
                          + row['question'] + '\n'
                          + '\n'.join(f'{chr(65+j)}. {v}' for j,v in enumerate(row['choices'])) + '\nAnswer:')
                prompts.append(chat(tokenizer, prompt))
            inputs = tokenizer(prompts, return_tensors='pt', padding=True).to('cuda')
            with torch.inference_mode():
                # Single-token continuations: direct next-token probability is exact conditional scoring.
                logits = model(**inputs).logits[:, -1, :].float()
                scores = torch.log_softmax(logits, dim=-1)[:, ids].cpu().tolist()
            manual_scores += scores
        predictions = [max(range(4), key=lambda j: scores[j]) for scores in manual_scores]
        fresh = [r for r in details if r['step'] == step]
        summary[str(step)] = {
            'gsm_n':len(gsm), 'gsm_numeric_output_n':sum(number(r['output']) is not None for r in gsm),
            'gsm_news_label_n':sum(r['output'].strip() in LABELS for r in gsm),
            'gsm_saved_accuracy':sum(r['correct'] for r in gsm)/len(gsm),
            'gsm_label_counts':dict(collections.Counter(r['output'].strip() for r in gsm if r['output'].strip() in LABELS)),
            'fresh_gsm_n':len(fresh), 'fresh_gsm_exact_match_n':sum(r['identical_output'] for r in fresh),
            'fresh_gsm_accuracy':sum(r['parsed'] is not None and r['parsed']==r['gold'] for r in fresh)/len(fresh),
            'open_news_label_n':sum(r['output'].strip() in LABELS for r in opens),
            'mmlu_n':len(mmlu), 'mmlu_direct_accuracy':sum(p==r['source']['answer'] for p,r in zip(predictions,mmlu))/len(mmlu),
            'mmlu_harness_accuracy':sum(r['correct'] for r in mmlu)/len(mmlu),
            'mmlu_prediction_agreement_n':sum(p==r['prediction'] for p,r in zip(predictions,mmlu)),
            'mmlu_harness_predictions':dict(collections.Counter(chr(65+r['prediction']) for r in mmlu)),
            'mmlu_direct_predictions':dict(collections.Counter(chr(65+p) for p in predictions)),
            'mean_best_letter_logprob_direct':sum(max(s) for s in manual_scores)/len(manual_scores),
            'mmlu_mean_probability_mass_ABCD':sum(sum(math.exp(v) for v in s) for s in manual_scores)/len(manual_scores),
            'seconds':time.perf_counter()-started,
        }
        write_json(ROOT/f'results/audit-stage1/mmlu-direct-{step}.json',manual_scores)
        del model
        gc.collect(); torch.cuda.empty_cache()
    summary['protocol']={'gsm_max_new_tokens':c['gsm_max_new_tokens'], 'gsm_new_sample_selection':'first 20 saved GSM items, same IDs across checkpoints',
                         'chat_template_sha256':hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                         'mmlu_scoring':'direct Transformers next-token logits, float32 log_softmax; harness used BF16 log_softmax'}
    write_json(ROOT/'results/audit-stage1/gsm20.json',details)
    write_json(ROOT/'reports/stage1/audit.json',summary)
    print(summary)

if __name__ == '__main__': main()
