import hashlib
import random
from common import ROOT, config, write_json
from datasets import load_dataset
import os

OPEN = [
    "Explain why the sky appears blue in three sentences.",
    "What is the difference between correlation and causation? Give an example.",
    "Write a Python function that returns the factorial of a nonnegative integer.",
    "Translate into French: I would like a glass of water, please.",
    "A shop sells 3 apples for 2 dollars. How much do 12 apples cost? Explain your reasoning.",
    "Summarize the main causes of the First World War in a short paragraph.",
    "Give three practical tips for learning a new language.",
    "Write a four-line poem about the ocean.",
    "Explain photosynthesis to a ten-year-old.",
    "请用中文解释什么是机器学习，并举一个生活中的例子。",
]


def balanced(ds, n, seed):
    groups = {}
    for i, row in enumerate(ds):
        groups.setdefault(row["label"], []).append(i)
    rng = random.Random(seed)
    ids = []
    for k, pool in sorted(groups.items()):
        rng.shuffle(pool)
        ids += pool[: n // len(groups) + (k < n % len(groups))]
    rng.shuffle(ids)
    return [dict(ds[i], source_id=i) for i in ids]


def main():
    c = config()
    out = ROOT / "data/stage1"
    if (out / "manifest.json").exists():
        raise SystemExit(
            "Prepared data already exists; retain it for fixed checkpoint comparisons."
        )
    if c["model_source"] != "modelscope":
        raise ValueError("This preparation script uses ModelScope")
    os.environ["MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS"] = str(
        c["model_download_workers"]
    )
    from modelscope import snapshot_download

    revision = c["model_revision"]
    print(f"Downloading {c['model']} from ModelScope ({revision})", flush=True)
    snapshot_download(
        c["model"],
        revision=revision,
        local_dir=str(ROOT / "artifacts/base"),
        allow_file_pattern=["*.json", "*.safetensors", "*.txt", "*.md", "LICENSE"],
    )
    model_hashes = {}
    for path in (ROOT / "artifacts/base").glob("*.safetensors"):
        with path.open("rb") as stream:
            model_hashes[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    if not model_hashes:
        raise RuntimeError("ModelScope download contains no safetensors weights")
    write_json(
        ROOT / "artifacts/base-source.json",
        {
            "source": "modelscope",
            "model": c["model"],
            "revision": revision,
            "weight_sha256": model_hashes,
        },
    )
    ag = load_dataset("fancyzhx/ag_news")
    train = balanced(ag["train"], c["train_size"], c["seed"])
    test = balanced(ag["test"], c["ag_test_size"], c["seed"])
    norm = lambda s: " ".join(s.lower().split())
    train_text = {norm(x["text"]) for x in train}
    # Keep fixed size by sampling from a deduplicated candidate pool.
    safe = ag["test"].filter(lambda r: norm(r["text"]) not in train_text)
    test = balanced(safe, c["ag_test_size"], c["seed"])
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    ids = random.Random(c["seed"]).sample(range(len(gsm)), c["gsm_test_size"])
    gsm_rows = [dict(gsm[i], source_id=i) for i in ids]
    mmlu_rows = []
    fingerprints = {
        "ag_train": ag["train"]._fingerprint,
        "ag_test": ag["test"]._fingerprint,
        "gsm8k": gsm._fingerprint,
    }
    for subject in c["mmlu_subjects"]:
        ds = load_dataset("cais/mmlu", subject, split="test")
        fingerprints[subject] = ds._fingerprint
        ids = random.Random(c["seed"]).sample(
            range(len(ds)), min(c["mmlu_per_subject"], len(ds))
        )
        mmlu_rows += [dict(ds[i], subject=subject, source_id=i) for i in ids]
    files = {
        "train": train,
        "ag_test": test,
        "gsm_test": gsm_rows,
        "mmlu_test": mmlu_rows,
        "open": OPEN,
    }
    hashes = {}
    for name, rows in files.items():
        path = out / (name + ".json")
        write_json(path, rows)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(
        out / "manifest.json",
        {
            "config": c,
            "model_source": "modelscope",
            "model_revision": revision,
            "model_weight_sha256": model_hashes,
            "fingerprints": fingerprints,
            "sha256": hashes,
            "counts": {k: len(v) for k, v in files.items()},
        },
    )
    print("Prepared fixed samples and local model:", out)


if __name__ == "__main__":
    main()
