"""M1/M2 单阶段训练：同一 LoRA adapter 延续，优化器与调度器每阶段重置；按固定顺序喂样本，里程碑保存 adapter。"""
import argparse
import collections
import gc
import math
import time
import torch
from datasets import Dataset
from torch.utils.data import SequentialSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
from common import ROOT, read_json
from train import Collator
from m1_common import atomic_json, batch_size, buffer_name, check_space, digest, encode, eval_steps, labels_of, load_config, roots, stage_order
from prepare_m1 import verify


class FixedOrderTrainer(Trainer):
    """The sample order is decided up front (incl. replay interleaving); never reshuffle."""
    def _get_train_sampler(self, train_dataset=None):
        return SequentialSampler(train_dataset if train_dataset is not None else self.train_dataset)


class SaveMilestones(TrainerCallback):
    def __init__(self, c, folder, steps):
        self.c, self.folder, self.steps = c, folder, steps

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step in self.steps:
            check_space(self.c, 200 * 2**20)
            model.save_pretrained(self.folder / f'step-{state.global_step}', safe_serialization=True)
        return control


def spec_of(c, run_id):
    return next(r for r in c['runs'] if r['id'] == run_id)


def final_adapter(c, run_id, smoke):
    return roots(c, smoke)['artifacts'] / run_id / f'step-{eval_steps(c, smoke)[-1]}'


def train_run(c, run_id, smoke=False):
    R = roots(c, smoke); spec = spec_of(c, run_id)
    res, art = R['results'] / run_id, R['artifacts'] / run_id
    identity = {'spec': spec, 'training': c['training'], 'lora': c['lora'], 'seed': c['seed'], 'smoke': smoke,
                'data_manifest_sha256': digest(R['data'] / 'manifest.json'),
                'init_complete': None if spec['init'] == 'base' else read_json(R['results'] / spec['init'] / 'train_complete.json')}
    done = res / 'train_complete.json'
    if done.exists():
        saved = read_json(done)
        if saved['identity'] != identity: raise RuntimeError(f'{run_id}: completed training identity mismatch; archive outputs first')
        final = str((final_adapter(c, run_id, smoke) / 'adapter_model.safetensors').relative_to(ROOT))
        for rel, h in saved['adapter_sha256'].items():
            # Intermediate adapters may be deleted after evaluation (storage policy); the stage endpoint must remain.
            if (rel == final or (ROOT / rel).exists()) and digest(ROOT / rel) != h: raise RuntimeError(f'Adapter changed: {rel}')
        print(f'Skip completed training: {run_id}', flush=True); return
    verify(c); check_space(c)
    atomic_json(res / 'run.json', identity)
    t = c['training']; task = spec['task']
    steps = 2 if smoke else t['steps_per_stage']; total = steps * batch_size(c)
    new = read_json(R['data'] / f'{task}_train.json')
    new = [dict(r, task=task) for r in new]
    buffer = read_json(R['data'] / f'{buffer_name(spec)}.json') if spec['replay_fraction'] else []
    order = stage_order(new, buffer, total, spec['replay_fraction'], c['seed'])
    tok = AutoTokenizer.from_pretrained(ROOT / c['base'], local_files_only=True); tok.padding_side = 'right'
    encoded = [encode(tok, r['task'], r, labels_of(c, r['task']), t['max_length'])[0] for _, r in order]
    exposure = {'new_samples': sum(k == 'new' for k, _ in order), 'replay_samples': sum(k == 'replay' for k, _ in order),
                'new_label_counts': dict(collections.Counter(r['label'] for k, r in order if k == 'new')),
                'replay_label_counts': dict(collections.Counter(f"{r['task']}:{r['label']}" for k, r in order if k == 'replay')),
                'new_target_tokens': sum(sum(x != -100 for x in e['labels']) for (k, _), e in zip(order, encoded) if k == 'new'),
                'replay_target_tokens': sum(sum(x != -100 for x in e['labels']) for (k, _), e in zip(order, encoded) if k == 'replay'),
                'new_input_tokens': sum(len(e['input_ids']) for (k, _), e in zip(order, encoded) if k == 'new'),
                'replay_input_tokens': sum(len(e['input_ids']) for (k, _), e in zip(order, encoded) if k == 'replay')}
    atomic_json(res / 'order.json', [{'kind': k, 'task': r['task'], 'source_id': r['source_id']} for k, r in order])
    atomic_json(res / 'exposure.json', exposure)
    set_seed(c['seed'])
    model = AutoModelForCausalLM.from_pretrained(ROOT / c['base'], local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    model.config.use_cache = False
    if spec['init'] == 'base':
        from peft import LoraConfig, get_peft_model
        l = c['lora']
        model = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', r=l['r'], lora_alpha=l['alpha'], lora_dropout=l['dropout'],
                                                 target_modules=l['target_modules'], bias=l['bias']))
    else:
        from peft import PeftModel
        # Continue the same adapter; optimizer/scheduler are fresh because this is a new Trainer.
        model = PeftModel.from_pretrained(model, final_adapter(c, spec['init'], smoke), is_trainable=True)
    model.enable_input_require_grads()
    params = {'all': sum(p.numel() for p in model.parameters()), 'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad)}
    if not 0 < params['trainable'] < params['all'] * 0.05: raise RuntimeError(f'Unexpected trainable parameter count: {params}')
    args = TrainingArguments(output_dir=str(art), max_steps=steps, learning_rate=t['learning_rate'],
        per_device_train_batch_size=t['micro_batch_size'], gradient_accumulation_steps=t['gradient_accumulation_steps'],
        warmup_steps=math.ceil(t['warmup_ratio'] * steps), weight_decay=t['weight_decay'], max_grad_norm=t['max_grad_norm'],
        lr_scheduler_type=t['lr_scheduler'], bf16=True, gradient_checkpointing=t['gradient_checkpointing'], optim=t['optimizer'],
        save_strategy='no', logging_steps=1 if smoke else 10, report_to='none', seed=c['seed'], data_seed=c['seed'], dataloader_num_workers=0)
    trainer = FixedOrderTrainer(model=model, args=args, train_dataset=Dataset.from_list(encoded), data_collator=Collator(tok.pad_token_id),
                                processing_class=tok, callbacks=[SaveMilestones(c, art, eval_steps(c, smoke))])
    print(f'M1 train {run_id}: {params} exposure={exposure}', flush=True)
    torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
    result = trainer.train(); torch.cuda.synchronize()
    if not math.isfinite(result.training_loss): raise RuntimeError('Non-finite training loss')
    atomic_json(res / 'train_log.json', trainer.state.log_history)
    atomic_json(res / 'timing.json', {'train_seconds': time.perf_counter() - start, 'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                                      'parameters': params, 'train_metrics': result.metrics, 'optimizer_resume_supported': False})
    adapters = {str(p.relative_to(ROOT)): digest(p) for p in art.rglob('adapter_model.safetensors')}
    if len(adapters) != len(eval_steps(c, smoke)): raise RuntimeError('Missing milestone adapters')
    atomic_json(done, {'identity': identity, 'adapter_sha256': adapters})
    print(f'Completed training {run_id}', flush=True)
    del trainer, model; gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/m1.json'); p.add_argument('--run', required=True); p.add_argument('--smoke', action='store_true')
    a = p.parse_args(); train_run(load_config(ROOT / a.config), a.run, a.smoke)
