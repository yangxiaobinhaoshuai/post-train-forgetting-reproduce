import argparse
import time
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from common import ROOT, LABELS, ag_prompt, chat, config, read_json, write_json


class Milestones(TrainerCallback):
    def __init__(self, steps):
        self.steps = steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.steps:
            control.should_save = True
        return control


def encode(tokenizer, row, max_length):
    prefix = tokenizer.encode(
        chat(tokenizer, ag_prompt(row["text"])), add_special_tokens=False
    )
    suffix = tokenizer.encode(
        LABELS[row["label"]] + tokenizer.eos_token, add_special_tokens=False
    )
    # Preserve instruction and leave room for the complete response.
    budget = max_length - len(suffix)
    if len(prefix) > budget:
        prefix = prefix[: budget - 16] + prefix[-16:]
    return {
        "input_ids": prefix + suffix,
        "attention_mask": [1] * (len(prefix) + len(suffix)),
        "labels": [-100] * len(prefix) + suffix,
    }


class Collator:
    def __init__(self, pad):
        self.pad = pad

    def __call__(self, rows):
        n = max(len(x["input_ids"]) for x in rows)
        return {
            k: torch.tensor(
                [
                    x[k]
                    + [self.pad if k == "input_ids" else (-100 if k == "labels" else 0)]
                    * (n - len(x[k]))
                    for x in rows
                ]
            )
            for k in rows[0]
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Separate 2-step run; never modifies formal checkpoints",
    )
    a = p.parse_args()
    c = config()
    set_seed(c["seed"])
    if not (ROOT / "data/stage1/manifest.json").exists():
        raise SystemExit("Run prepare.py first")
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "artifacts/base")
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        ROOT / "artifacts/base", torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.config.use_cache = False
    assert all(p.requires_grad for p in model.parameters()), (
        "Full-parameter training required"
    )
    out = ROOT / ("artifacts/smoke" if a.smoke else "artifacts/stage1")
    if out.exists() and any(out.glob("checkpoint-*")):
        raise SystemExit(
            "Output already contains checkpoints; use a fresh run directory."
        )
    rows = read_json(ROOT / "data/stage1/train.json")
    ds = Dataset.from_list(
        [
            encode(tokenizer, r, c["max_length"])
            for r in (rows[:32] if a.smoke else rows)
        ]
    )
    args = TrainingArguments(
        output_dir=str(out),
        max_steps=2 if a.smoke else c["max_steps"],
        learning_rate=c["learning_rate"],
        per_device_train_batch_size=c["micro_batch_size"],
        gradient_accumulation_steps=c["gradient_accumulation_steps"],
        warmup_steps=0 if a.smoke else c["warmup_steps"],
        weight_decay=c["weight_decay"],
        max_grad_norm=c["max_grad_norm"],
        lr_scheduler_type="constant_with_warmup",
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        save_strategy="no",
        save_only_model=True,
        logging_steps=1 if a.smoke else 10,
        report_to="none",
        seed=c["seed"],
        data_seed=c["seed"],
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=Collator(tokenizer.pad_token_id),
        processing_class=tokenizer,
        callbacks=[Milestones([2] if a.smoke else c["checkpoint_steps"])],
    )
    if not a.smoke:
        trainer.save_model(str(out / "checkpoint-0"))
        write_json(out / "config.json", c)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = trainer.train()
    torch.cuda.synchronize()
    write_json(
        out / "timing.json",
        {
            "seconds": time.perf_counter() - start,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "metrics": result.metrics,
            "smoke": a.smoke,
        },
    )


if __name__ == "__main__":
    main()
