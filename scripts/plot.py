import csv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import ROOT, read_json

paths = sorted(
    (ROOT / "results/stage1").glob("checkpoint-*/metrics.json"),
    key=lambda p: int(p.parent.name.split("-")[-1]),
)
rows = []
for p in paths:
    m = read_json(p)
    rows.append(
        {
            "step": m["step"],
            **{k: m[k]["accuracy"] for k in ["ag_test", "gsm_test", "mmlu_test"]},
            "eval_seconds": m["total_seconds"],
        }
    )
if not rows:
    raise SystemExit("No formal evaluations found")
out = ROOT / "reports/stage1"
out.mkdir(parents=True, exist_ok=True)
with (out / "curve.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for k in ["ag_test", "gsm_test", "mmlu_test"]:
    plt.plot([r["step"] for r in rows], [100 * r[k] for r in rows], marker="o", label=k)
plt.xlabel("Optimizer steps")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.tight_layout()
plt.savefig(out / "curve.png", dpi=160)
print(out)
