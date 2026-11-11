#!/usr/bin/env python3
"""Create tables and figures from local or cluster scaling measurements."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--summary", type=Path, default=ROOT / "results" / "summary.csv")
parser.add_argument("--output-dir", type=Path, default=ROOT / "report")
parser.add_argument("--label-prefix", default="",
                    help="prefix for LaTeX table labels when including multiple result sets")
args = parser.parse_args()
report = args.output_dir.expanduser()
with args.summary.expanduser().open() as f:
    rows = list(csv.DictReader(f))
if not rows:
    parser.error("summary CSV is empty")
for row in rows:
    for key in ("ranks", "threads", "n"):
        row[key] = int(row[key])
    for key in ("median_seconds", "min_seconds", "max_seconds",
                "median_communication_seconds", "speedup", "efficiency"):
        row[key] = float(row[key])

strong = [r for r in rows if r["kind"] == "strong"]
weak = [r for r in rows if r["kind"] == "weak"]
if not strong or not weak:
    parser.error("summary CSV needs strong and weak scaling rows")
metadata_path = args.summary.parent / "machine.json"
metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
repetitions = metadata.get("repetitions", 5)
report.mkdir(parents=True, exist_ok=True)
with (report / "results_tables.tex").open("w") as f:
    f.write("\\begin{table}[H]\\centering\\small\n")
    f.write(f"\\caption{{Strong scaling, $N={strong[0]['n']}$, "
            f"$D={strong[0]['d']}$, $K={strong[0]['k']}$, "
            f"{strong[0]['iterations']} iterations. Times are medians of "
            f"{repetitions} runs; ranges show minimum and maximum.}}\n")
    f.write("\\begin{tabular}{rrrrrrr}\\toprule\n")
    f.write("MPI & OMP & Workers & Time (s) & Range (s) & Speedup & Efficiency \\\\\n\\midrule\n")
    for r in strong:
        f.write(f"{r['ranks']} & {r['threads']} & {r['ranks']*r['threads']} & "
                f"{r['median_seconds']:.3f} & {r['min_seconds']:.3f}--{r['max_seconds']:.3f} & "
                f"{r['speedup']:.2f} & {100*r['efficiency']:.1f}\\% \\\\\n")
    f.write(f"\\bottomrule\\end{{tabular}}\\label{{tab:{args.label_prefix}strong}}\\end{{table}}\n")
    f.write("\\begin{table}[H]\\centering\\small\n")
    f.write(f"\\caption{{Weak scaling with {weak[0]['n'] // weak[0]['ranks']:,} "
            "observations per MPI process and one OpenMP thread.}\n")
    f.write("\\begin{tabular}{rrrrr}\\toprule\n")
    f.write("MPI & Observations & Time (s) & Range (s) & Weak efficiency \\\\\n\\midrule\n")
    for r in weak:
        f.write(f"{r['ranks']} & {r['n']:,} & {r['median_seconds']:.3f} & "
                f"{r['min_seconds']:.3f}--{r['max_seconds']:.3f} & "
                f"{100*r['efficiency']:.1f}\\% \\\\\n")
    f.write(f"\\bottomrule\\end{{tabular}}\\label{{tab:{args.label_prefix}weak}}\\end{{table}}\n")

plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 180})
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.35), constrained_layout=True)
ax = axes[0]
for name, selected, marker in [
    ("OpenMP only", [r for r in strong if r["ranks"] == 1], "o"),
    ("MPI only", [r for r in strong if r["threads"] == 1], "s"),
    ("Hybrid", [r for r in strong if r["ranks"] > 1 and r["threads"] > 1], "^"),
]:
    selected.sort(key=lambda r: r["ranks"] * r["threads"])
    if selected:
        ax.plot([r["ranks"] * r["threads"] for r in selected],
                [r["speedup"] for r in selected], marker=marker, linewidth=1.8, label=name)
workers = sorted({r["ranks"] * r["threads"] for r in strong})
max_workers = max(workers)
ax.plot([1, max_workers], [1, max_workers], "--", color="0.5", linewidth=1, label="Ideal")
ax.set(xlabel="Total workers", ylabel="Strong speedup", title="Strong scaling",
       xticks=workers, ylim=(0, 1.04 * max(max_workers, max(r["speedup"] for r in strong))))
ax.grid(alpha=0.25)
ax.legend(fontsize=7, loc="upper left")
ax = axes[1]
ax.plot([r["ranks"] for r in weak], [r["median_seconds"] for r in weak],
        marker="s", linewidth=1.8, color="tab:orange", label="Measured")
ax.axhline(weak[0]["median_seconds"], linestyle="--", color="0.5", linewidth=1,
           label="Ideal")
ax.set(xlabel="MPI processes", ylabel="Time (s)", title="Weak scaling",
       xticks=sorted({r["ranks"] for r in weak}))
ax.grid(alpha=0.25)
ax.legend(fontsize=7)
fig.savefig(report / "scaling.pdf", bbox_inches="tight")
print(f"Wrote {report / 'results_tables.tex'} and {report / 'scaling.pdf'}")
