"""Stage2 单个训练 run；训练中在线评测，完成的 run 可跳过。"""
import argparse
import gc
import math
import time
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
from common import ROOT, read_json
from train import encode, Collator
from stage2_common import atomic_json, load_config, output_roots, prepared_manifest, signature, digest, check_space, math_encode, directory_bytes
from stage2_eval import evaluate


def model_fingerprint(model):
    import hashlib
    h=hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        if param.requires_grad:
            h.update(name.encode());h.update(str((str(param.dtype),list(param.shape))).encode())
            h.update(param.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def run_specs(c):
    sensitivity=[dict(r, group='sensitivity', start='base', data='stage1_ag', train_size=8000) for r in c['sensitivity']['runs']]
    recovery=[dict(r, group='recovery', method='full', learning_rate=c['recovery']['learning_rate']) for r in c['recovery']['runs']]
    return sensitivity+recovery


class OnlineEvaluation(TrainerCallback):
    def __init__(self,c,spec,tokenizer,roots,smoke):
        self.c,self.spec,self.tokenizer,self.roots,self.smoke=c,spec,tokenizer,roots,smoke
        self.eval_seconds=0
        self.phase=c[spec['group']]
        self.quick=[1,2] if smoke else self.phase['quick_eval_steps']
        self.full=[2] if smoke else self.phase['full_eval_steps']

    def on_step_end(self,args,state,control,model=None,**kwargs):
        step=state.global_step
        check_space(self.c,self.roots)
        if step not in self.quick and step not in self.full: return control
        run_id=self.spec['id']
        out=self.roots['results']/run_id/f'step-{step}'
        profile='full' if step in self.full else 'quick'
        fingerprint=model_fingerprint(model)
        metrics=evaluate(model,self.tokenizer,self.c,out,profile=profile,
                         batch_size=self.c['evaluation']['in_process_batch_size'],smoke=self.smoke)
        self.eval_seconds+=metrics['total_seconds']
        atomic_json(out/'state.json',{'step':step,'trainable_weight_sha256':fingerprint,
            'weight_hash_scope':'trainable_parameters_only','learning_rate':kwargs['optimizer'].param_groups[0]['lr']})
        if self.spec['method']=='lora':
            if step==self.phase['max_steps'] or (self.smoke and step==2):
                with model.disable_adapter():
                    disabled=evaluate(model,self.tokenizer,self.c,out/'adapter-disabled',profile='quick',
                        batch_size=self.c['evaluation']['in_process_batch_size'],smoke=self.smoke)
                    self.eval_seconds+=disabled['total_seconds']
            if self.c['storage']['keep_lora_eval_checkpoints']:
                path=self.roots['artifacts']/run_id/f'step-{step}'
                reserve=sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad)
                check_space(self.c,self.roots,max(0,reserve-directory_bytes(path)))
                model.save_pretrained(path,safe_serialization=True)
        return control


def train_run(c,run_id,smoke=False):
    roots=output_roots(smoke)
    experiment=read_json(roots['results']/'experiment.json')
    if experiment['config_signature']!=signature(c): raise RuntimeError('Experiment/config mismatch')
    manifest=prepared_manifest(c)
    spec=next(r for r in run_specs(c) if r['id']==run_id)
    out=roots['results']/run_id
    identity={'experiment_signature':experiment['signature'],'spec':spec,'smoke':smoke}
    complete=out/'complete.json'
    if complete.exists():
        saved=read_json(complete)
        if saved['identity']!=identity: raise RuntimeError('Completed run identity mismatch')
        for relative,expected in saved['output_hashes'].items():
            if digest(ROOT/relative)!=expected: raise RuntimeError(f'Completed output changed: {relative}')
        print(f'Skip completed run: {run_id}',flush=True);return
    check_space(c,roots)
    atomic_json(out/'run.json',identity)
    set_seed(c['seed'])
    tokenizer=AutoTokenizer.from_pretrained(ROOT/c['references']['base'],local_files_only=True)
    tokenizer.padding_side='right'
    source=ROOT/c['references'][spec['start']]
    model=AutoModelForCausalLM.from_pretrained(source,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa')
    model.config.use_cache=False
    if spec['method']=='lora':
        from peft import LoraConfig, get_peft_model
        l=c['sensitivity']['lora']
        model=get_peft_model(model,LoraConfig(task_type='CAUSAL_LM',r=l['r'],lora_alpha=l['alpha'],
            lora_dropout=l['dropout'],target_modules=l['target_modules'],bias=l['bias'],modules_to_save=l['modules_to_save']))
        # Frozen embeddings still need an activation gradient for reentrant checkpointing.
        model.enable_input_require_grads()
    phase=c[spec['group']]
    if spec['data']=='stage1_ag':
        rows=read_json(ROOT/c['sensitivity']['train_data'])
        encoded=[encode(tokenizer,r,phase['max_length']) for r in (rows[:32] if smoke else rows)]
    else:
        rows=read_json(ROOT/f"data/stage2/gsm_train_{spec['train_size']}.json")
        encoded=[math_encode(tokenizer,r,phase['max_length']) for r in (rows[:32] if smoke else rows)]
    atomic_json(out/'initial_reference.json',{'reference':spec['start'],'path':str((roots['results']/'references'/spec['start']).relative_to(ROOT)),
        'source_weight_sha256':manifest['model_hashes'][str((source/'model.safetensors').relative_to(ROOT))]})
    t=c['common_training']
    args=TrainingArguments(output_dir=str(roots['artifacts']/run_id),max_steps=2 if smoke else phase['max_steps'],
        learning_rate=spec['learning_rate'],per_device_train_batch_size=t['micro_batch_size'],
        gradient_accumulation_steps=t['gradient_accumulation_steps'],warmup_steps=phase['warmup_steps'],
        weight_decay=t['weight_decay'],max_grad_norm=t['max_grad_norm'],lr_scheduler_type=t['lr_scheduler'],
        bf16=True,gradient_checkpointing=t['gradient_checkpointing'],optim=t['optimizer'],
        save_strategy='no',logging_steps=1 if smoke else 10,report_to='none',seed=c['seed'],data_seed=c['seed'],
        dataloader_num_workers=0)
    callback=OnlineEvaluation(c,spec,tokenizer,roots,smoke)
    trainer=Trainer(model=model,args=args,train_dataset=Dataset.from_list(encoded),data_collator=Collator(tokenizer.pad_token_id),
                    processing_class=tokenizer,callbacks=[callback])
    params={'all':sum(p.numel() for p in model.parameters()),'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad)}
    print(f'Stage2 run {run_id}: {params}',flush=True)
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    result=trainer.train();torch.cuda.synchronize()
    wall=time.perf_counter()-start
    if not math.isfinite(result.training_loss): raise RuntimeError('Non-finite training loss')
    weights={}
    if not smoke and run_id in c['storage']['keep_full_weight_run_endpoints']:
        path=roots['artifacts']/run_id/'final'
        reserve=sum(p.numel()*p.element_size() for p in model.parameters())
        check_space(c,roots,max(0,reserve-directory_bytes(path)))
        trainer.save_model(str(path))
        weights={str(p.relative_to(ROOT)):digest(p) for p in path.glob('*.safetensors')}
    atomic_json(out/'timing.json',{'wall_seconds':wall,'evaluation_seconds':callback.eval_seconds,
        'wall_excluding_evaluation_seconds':wall-callback.eval_seconds,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
        'parameters':params,'train_metrics':result.metrics,'optimizer_resume_supported':False})
    # Completion marker is written last; an interrupted run is retrained from its declared start.
    paths=[p for p in out.rglob('*.json') if p.name!='complete.json']
    hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}
    hashes.update(weights)
    for p in (roots['artifacts']/run_id).rglob('adapter_model.safetensors'):
        hashes[str(p.relative_to(ROOT))]=digest(p)
    atomic_json(complete,{'identity':identity,'output_hashes':hashes,'completed_steps':2 if smoke else phase['max_steps']})
    print(f'Completed {run_id}',flush=True)
    del trainer,model;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/stage2.json');p.add_argument('--run',required=True);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();train_run(load_config(a.config),a.run,a.smoke)
