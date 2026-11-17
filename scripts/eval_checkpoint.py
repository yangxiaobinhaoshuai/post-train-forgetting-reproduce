"""Fixed sample evaluations through lm-evaluation-harness HFLM backend."""

import argparse
from decimal import Decimal
import re
import time
import torch
from lm_eval.api.instance import Instance
from lm_eval.models.huggingface import HFLM
from common import ROOT, LABELS, config, read_json, write_json, chat, ag_prompt


def request(kind, args, i):
    return Instance(
        request_type=kind, doc={}, arguments=args, idx=i, metadata=("stage1", i, 1)
    )


def number(text):
    values = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return format(Decimal(values[-1]).normalize(), "f") if values else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", nargs="+", type=int)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    c = config()
    data = ROOT / "data/stage1"
    from transformers import set_seed

    set_seed(c["seed"])
    for step in a.steps if a.steps is not None else c["checkpoint_steps"]:
        path = ROOT / "artifacts/stage1" / f"checkpoint-{step}"
        if step == 0 and not path.exists():
            path = ROOT / "artifacts/base"
        out = (
            ROOT
            / ("results/smoke" if a.smoke else "results/stage1")
            / f"checkpoint-{step}"
        )
        if (out / "metrics.json").exists():
            print("Already evaluated:", step)
            continue
        started = time.perf_counter()
        lm = HFLM(
            pretrained=str(path),
            tokenizer=str(ROOT / "artifacts/base"),
            device="cuda:0",
            dtype="bfloat16",
            batch_size=c["eval_batch_size"],
            max_length=2048,
            truncation=True,
        )
        metrics = {
            "step": step,
            "load_seconds": time.perf_counter() - started,
            "smoke": a.smoke,
        }
        for name, limit in [
            ("ag_test", 16),
            ("gsm_test", c["gsm_max_new_tokens"]),
            ("open", c["open_max_new_tokens"]),
        ]:
            rows = read_json(data / (name + ".json"))
            if a.smoke:
                rows = rows[:2]
            prompts = [
                ag_prompt(r["text"])
                if name == "ag_test"
                else (
                    r["question"] + "\nSolve step by step and end with: #### <answer>."
                    if name == "gsm_test"
                    else r
                )
                for r in rows
            ]
            start = time.perf_counter()
            outputs = lm.generate_until(
                [
                    request(
                        "generate_until",
                        (
                            chat(lm.tokenizer, q),
                            {
                                "until": [lm.tokenizer.eos_token],
                                "max_gen_toks": min(limit, 32) if a.smoke else limit,
                                "do_sample": False,
                            },
                        ),
                        i,
                    )
                    for i, q in enumerate(prompts)
                ]
            )
            records = []
            for row, prompt, output in zip(rows, prompts, outputs):
                record = {"source": row, "prompt": prompt, "output": output}
                if name == "ag_test":
                    record["correct"] = output.strip() == LABELS[row["label"]]
                    record["valid_format"] = output.strip() in LABELS
                if name == "gsm_test":
                    record["correct"] = number(output) is not None and number(
                        output
                    ) == number(row["answer"].split("####")[-1])
                    record["has_answer_marker"] = "####" in output
                records.append(record)
            write_json(out / (name + ".json"), records)
            metrics[name] = {"n": len(records), "seconds": time.perf_counter() - start}
            if name != "open":
                metrics[name]["accuracy"] = sum(r["correct"] for r in records) / len(
                    records
                )
            if name == "ag_test":
                metrics[name]["valid_format_rate"] = sum(
                    r["valid_format"] for r in records
                ) / len(records)
        rows = read_json(data / "mmlu_test.json")
        rows = rows[:2] if a.smoke else rows
        requests = []
        for i, row in enumerate(rows):
            prompt = (
                "The following is a multiple-choice question. Reply with the letter of the correct answer.\n"
                + row["question"]
                + "\n"
                + "\n".join(f"{chr(65 + j)}. {v}" for j, v in enumerate(row["choices"]))
                + "\nAnswer:"
            )
            for j in range(4):
                requests.append(
                    request(
                        "loglikelihood",
                        (chat(lm.tokenizer, prompt), chr(65 + j)),
                        4 * i + j,
                    )
                )
        start = time.perf_counter()
        scores = lm.loglikelihood(requests)
        records = []
        for i, row in enumerate(rows):
            values = [float(x[0]) for x in scores[4 * i : 4 * i + 4]]
            pred = max(range(4), key=lambda j: values[j])
            records.append(
                {
                    "source": row,
                    "loglikelihoods": values,
                    "prediction": pred,
                    "correct": pred == row["answer"],
                }
            )
        write_json(out / "mmlu_test.json", records)
        metrics["mmlu_test"] = {
            "n": len(rows),
            "accuracy": sum(r["correct"] for r in records) / len(rows),
            "seconds": time.perf_counter() - start,
        }
        metrics["total_seconds"] = time.perf_counter() - started
        write_json(out / "metrics.json", metrics)
        print(metrics)
        del lm
        import gc

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
