"""Offline full-size architecture check; random weights, NOT experimental evidence."""

import time
import torch
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from lm_eval.models.huggingface import HFLM
from common import ROOT, config, write_json
from train import encode, Collator
from eval_checkpoint import request

c = config()
set_seed(c["seed"])
tok = AutoTokenizer.from_pretrained(ROOT / "artifacts/base")
model = AutoModelForCausalLM.from_config(
    AutoConfig.from_pretrained(ROOT / "artifacts/base"),
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
model.config.use_cache = False
rows = [
    encode(
        tok,
        {"text": "Scientists report a new computing discovery. " * 200, "label": 3},
        c["max_length"],
    )
    for _ in range(16)
]
args = TrainingArguments(
    output_dir=str(ROOT / "artifacts/runtime-validation"),
    max_steps=1,
    per_device_train_batch_size=c["micro_batch_size"],
    gradient_accumulation_steps=c["gradient_accumulation_steps"],
    learning_rate=c["learning_rate"],
    bf16=True,
    gradient_checkpointing=True,
    save_strategy="no",
    report_to="none",
    optim="adamw_torch",
    logging_steps=1,
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=Dataset.from_list(rows),
    data_collator=Collator(tok.pad_token_id),
    processing_class=tok,
)
torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()
result = trainer.train()
torch.cuda.synchronize()
assert torch.isfinite(torch.tensor(result.training_loss)) and result.training_loss > 0
record = {
    "random_weights": True,
    "model": c["model"],
    "parameters": sum(p.numel() for p in model.parameters()),
    "optimizer_steps": 1,
    "train_seconds": time.perf_counter() - start,
    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    "loss": result.training_loss,
}
# Free optimizer state before exercising the actual harness backend.
trainer.optimizer = None
trainer.lr_scheduler = None
del trainer
model.gradient_checkpointing_disable()
model.config.use_cache = True
model.eval()
lm = HFLM(pretrained=model, tokenizer=tok, batch_size=1, max_length=1024)
ll = lm.loglikelihood([request("loglikelihood", ("Question: 1+1? Answer:", " A"), 0)])
gen = lm.generate_until(
    [
        request(
            "generate_until",
            (
                "Hello",
                {"max_gen_toks": 4, "until": [tok.eos_token], "do_sample": False},
            ),
            0,
        )
    ]
)
assert len(ll) == 1 and len(gen) == 1
record["harness_generation_and_likelihood_ok"] = True
write_json(ROOT / "reports/runtime-validation.json", record)
print(record)
