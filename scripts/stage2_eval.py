"""Stage2 在线评测：复用模型，恢复训练状态与随机数状态。"""
import collections
from contextlib import contextmanager
import gc
import math
import random
import time
import numpy as np
import torch
from lm_eval.models.huggingface import HFLM
from common import ROOT, LABELS, ag_prompt, chat, read_json
from eval_checkpoint import request, number
from stage2_common import atomic_json, gsm_prompt


@contextmanager
def inference_state(model, tokenizer):
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    modes = [(m, m.training) for m in model.modules()]
    cache = model.config.use_cache
    checkpointing = model.is_gradient_checkpointing
    padding, truncation = tokenizer.padding_side, tokenizer.truncation_side
    try:
        if checkpointing: model.gradient_checkpointing_disable()
        model.config.use_cache = True
        model.eval(); tokenizer.padding_side = 'left'; tokenizer.truncation_side = 'left'
        with torch.inference_mode(): yield
    finally:
        model.config.use_cache = cache
        if checkpointing: model.gradient_checkpointing_enable()
        for module, training in modes: module.training = training
        tokenizer.padding_side, tokenizer.truncation_side = padding, truncation
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
        if rng[3]: torch.cuda.set_rng_state_all(rng[3])
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


class RecordingHFLM(HFLM):
    def _model_generate(self, context, max_length, stop, **kwargs):
        output = super()._model_generate(context, max_length, stop, **kwargs)
        mask = kwargs.get('attention_mask')
        eos = set()
        for value in [self.tokenizer.eos_token_id, self.model.generation_config.eos_token_id]:
            eos.update(value if isinstance(value, list) else ([] if value is None else [value]))
        for i, tokens in enumerate(output[:, context.shape[1]:].tolist()):
            prefix = context[i][mask[i].bool()].tolist() if mask is not None else context[i].tolist()
            end = next((j for j,t in enumerate(tokens) if t in eos), None)
            stats = {'output_tokens':len(tokens) if end is None else end,
                     'stop_reason':'eos' if end is not None else ('length_limit' if len(tokens)>=max_length-context.shape[1] else 'stop_sequence')}
            self.token_stats[tuple(prefix)].append(stats)
        return output


def evaluate(model, tokenizer, c, out, profile='quick', batch_size=1, smoke=False):
    out.mkdir(parents=True, exist_ok=True)
    e = c['evaluation']
    data = read_json(ROOT / f'data/stage2/eval_{profile}.json')
    if smoke: data = {name:rows[:2] for name,rows in data.items()}
    started = time.perf_counter()
    metrics = {'profile':profile, 'smoke':smoke, 'batch_size':batch_size,
               'softmax_dtype':'float32', 'gsm_max_new_tokens':min(e['gsm_max_new_tokens'],8) if smoke else e['gsm_max_new_tokens']}
    with inference_state(model, tokenizer):
        lm = RecordingHFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size,
                          max_length=e['max_context_length'], truncation=True, softmax_dtype='float32')
        lm.token_stats = collections.defaultdict(collections.deque)
        for task, limit in [('ag_test',e['ag_max_new_tokens']),('gsm_test',e['gsm_max_new_tokens']),('open',e['open_max_new_tokens'])]:
            rows = data[task]
            prompts = [ag_prompt(r['text']) if task=='ag_test' else gsm_prompt(r) if task=='gsm_test' else r for r in rows]
            rendered = [chat(tokenizer, p) for p in prompts]
            cap = min(limit,8) if smoke else limit
            start = time.perf_counter()
            outputs = lm.generate_until([request('generate_until',(p,{'until':[tokenizer.eos_token], 'max_gen_toks':cap,'do_sample':False}),i) for i,p in enumerate(rendered)])
            records = []
            for row, prompt, rendered_prompt, output in zip(rows,prompts,rendered,outputs):
                key = tuple(lm.tok_encode(rendered_prompt)[-(e['max_context_length']-cap):])
                if not lm.token_stats[key]: raise RuntimeError('Cannot align raw token statistics with generated request')
                r = {'source':row,'prompt':prompt,'output':output, **lm.token_stats[key].popleft()}
                r['news_label'] = output.strip() in LABELS
                if task=='ag_test': r.update(correct=output.strip()==LABELS[row['label']],valid_format=r['news_label'])
                if task=='gsm_test':
                    r.update(parsed=number(output), gold=number(row['answer'].split('####')[-1]))
                    r['correct'] = r['parsed'] is not None and r['parsed']==r['gold']
                    r['has_answer_marker'] = '####' in output
                records.append(r)
            atomic_json(out / (task+'.json'),records)
            m = {'n':len(rows),'seconds':time.perf_counter()-start,
                 'news_label_rate':sum(r['news_label'] for r in records)/len(rows),
                 'mean_output_tokens':sum(r['output_tokens'] for r in records)/len(rows),
                 'length_limit_rate':sum(r['stop_reason']=='length_limit' for r in records)/len(rows)}
            if task!='open': m['accuracy']=sum(r['correct'] for r in records)/len(rows)
            if task=='gsm_test': m['numeric_output_rate']=sum(r['parsed'] is not None for r in records)/len(rows)
            if task=='ag_test': m['valid_format_rate']=sum(r['valid_format'] for r in records)/len(rows)
            metrics[task] = m
        start = time.perf_counter(); reqs = []
        for i,row in enumerate(data['mmlu_test']):
            prompt = ('The following is a multiple-choice question. Reply with the letter of the correct answer.\n'+row['question']+'\n'+'\n'.join(f'{chr(65+j)}. {v}' for j,v in enumerate(row['choices']))+'\nAnswer:')
            reqs += [request('loglikelihood',(chat(tokenizer,prompt),chr(65+j)),4*i+j) for j in range(4)]
        scores = lm.loglikelihood(reqs); records = []
        for i,row in enumerate(data['mmlu_test']):
            values = [float(v[0]) for v in scores[4*i:4*i+4]]
            pred = max(range(4),key=lambda j:values[j])
            records.append({'source':row,'loglikelihoods':values,'prediction':pred,'correct':pred==row['answer'],
                            'letter_probability_mass':sum(math.exp(x) for x in values)})
        atomic_json(out/'mmlu_test.json',records)
        metrics['mmlu_test'] = {'n':len(records),'accuracy':sum(r['correct'] for r in records)/len(records),
            'letter_probability_mass':sum(r['letter_probability_mass'] for r in records)/len(records),
            'prediction_counts':dict(collections.Counter(chr(65+r['prediction']) for r in records)),
            'seconds':time.perf_counter()-start}
        # Conditional answer likelihood is a separate diagnostic, not generation accuracy.
        start=time.perf_counter(); reqs=[]; answers=[]
        for i,row in enumerate(data['gsm_test']):
            rationale, sep, answer = row['answer'].rpartition('####')
            if not sep: raise ValueError('GSM gold answer has no #### marker')
            context=chat(tokenizer,gsm_prompt(row))+rationale+sep
            token_count=len(lm._encode_pair(context,answer)[1])
            reqs.append(request('loglikelihood',(context,answer),i)); answers.append((row,token_count))
        scores=lm.loglikelihood(reqs)
        nll=[{'source_id':row['source_id'],'answer_nll':-float(score[0]),'answer_tokens':count} for (row,count),score in zip(answers,scores)]
        atomic_json(out/'gold_answer_nll.json',nll)
        metrics['gold_answer_nll']={'n':len(nll),'mean_sequence_nll':sum(r['answer_nll'] for r in nll)/len(nll),
            'mean_token_nll':sum(r['answer_nll'] for r in nll)/sum(r['answer_tokens'] for r in nll),'seconds':time.perf_counter()-start}
        del lm
    if profile == 'full':
        quick = read_json(ROOT / 'data/stage2/eval_quick.json')
        def key(task, row):
            if task == 'open': return row
            return (row.get('subject'), row['source_id'])
        view = {}
        for task in ['ag_test', 'gsm_test', 'mmlu_test', 'open']:
            allowed = {key(task, row) for row in quick[task]}
            records = [r for r in read_json(out/(task+'.json')) if key(task,r['source']) in allowed]
            m = {'n':len(records)}
            if task != 'open': m['accuracy']=sum(r['correct'] for r in records)/len(records)
            if task != 'mmlu_test':
                m['news_label_rate']=sum(r['news_label'] for r in records)/len(records)
                m['mean_output_tokens']=sum(r['output_tokens'] for r in records)/len(records)
                m['length_limit_rate']=sum(r['stop_reason']=='length_limit' for r in records)/len(records)
            if task=='gsm_test': m['numeric_output_rate']=sum(r['parsed'] is not None for r in records)/len(records)
            if task=='mmlu_test': m['letter_probability_mass']=sum(r['letter_probability_mass'] for r in records)/len(records)
            view[task]=m
        metrics['quick_view']=view
    metrics['total_seconds']=time.perf_counter()-started
    atomic_json(out/'metrics.json',metrics)
    return metrics
