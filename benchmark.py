#!/usr/bin/env python3
"""Run reproducible single-node or cluster strong/weak scaling experiments."""

import argparse
import csv
import json
import os
import platform
import shutil
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXE = ROOT / "kmeans"


def parse_configurations(value):
    configurations = []
    for item in value.split(","):
        try:
            ranks, threads = (int(part) for part in item.strip().lower().split("x"))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid configuration: {item}") from exc
        if ranks < 1 or threads < 1:
            raise argparse.ArgumentTypeError("rank and thread counts must be positive")
        configurations.append((ranks, threads))
    if not configurations or (1, 1) not in configurations or len(set(configurations)) != len(configurations):
        raise argparse.ArgumentTypeError("configurations must be unique and include 1x1")
    return configurations


def parse_rank_list(value):
    try:
        ranks = [int(part) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("weak ranks must be comma-separated integers") from exc
    if not ranks or 1 not in ranks or any(rank < 1 for rank in ranks) or len(set(ranks)) != len(ranks):
        raise argparse.ArgumentTypeError("weak ranks must be unique, positive, and include 1")
    return ranks


def physical_cores():
    if platform.system() == "Darwin":
        result = subprocess.run(["sysctl", "-n", "hw.physicalcpu"], text=True,
                                capture_output=True, check=False)
        if result.returncode == 0:
            return int(result.stdout.strip())
    return os.cpu_count()


def run(ranks, threads, n, d, k, iterations, validate=False, bind_to_none=False):
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS=str(threads), OMP_DYNAMIC="FALSE")
    cmd = ["mpirun"]
    if bind_to_none:
        cmd += ["--bind-to", "none"]
    cmd += ["-np", str(ranks), str(EXE), "--n", str(n),
           "--d", str(d), "--k", str(k), "--iterations", str(iterations)]
    if validate:
        cmd.append("--validate")
    completed = subprocess.run(cmd, env=env, text=True, capture_output=True, check=True)
    return json.loads(completed.stdout.strip().splitlines()[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--strong-n", type=int, default=1000000)
    parser.add_argument("--weak-n-per-rank", type=int, default=250000)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--strong-configs", type=parse_configurations,
                        default=parse_configurations("1x1,1x2,1x4,1x8,2x1,4x1,2x2,4x2"))
    parser.add_argument("--weak-ranks", type=parse_rank_list,
                        default=parse_rank_list("1,2,4"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--bind-to-none", action="store_true",
                        help="allow OpenMP threads to use all CPUs assigned to the PBS job")
    args = parser.parse_args()
    if (args.repetitions < 1 or args.strong_n < args.k or args.weak_n_per_rank < args.k
            or args.d < 1 or args.k < 1 or args.iterations < 1):
        parser.error("require positive repetitions, d, k, iterations and dataset sizes >= k")
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = output_dir / "raw.jsonl"
    summary = output_dir / "summary.csv"
    cases = [("strong", r, t, args.strong_n) for r, t in args.strong_configs]
    cases += [("weak", r, 1, args.weak_n_per_rank * r) for r in args.weak_ranks]
    compiler = os.environ.get("CXX") or ("/opt/homebrew/bin/g++-16"
                                        if Path("/opt/homebrew/bin/g++-16").exists()
                                        else shutil.which("g++"))
    machine = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "mpirun_version": subprocess.run(["mpirun", "--version"], text=True,
                                         capture_output=True, check=True).stdout.splitlines()[0],
        "compiler_version": subprocess.run([compiler, "--version"],
                                           text=True, capture_output=True, check=True).stdout.splitlines()[0],
        "physical_cores": physical_cores(),
        "strong_configurations": args.strong_configs,
        "weak_ranks": args.weak_ranks,
        "repetitions": args.repetitions,
        "strong_n": args.strong_n,
        "weak_n_per_rank": args.weak_n_per_rank,
        "d": args.d,
        "k": args.k,
        "iterations": args.iterations,
        "mpi_bind_to_none": args.bind_to_none,
    }
    (output_dir / "machine.json").write_text(json.dumps(machine, indent=2) + "\n")
    rows = []
    with raw.open("w") as f:
        for kind, ranks, threads, n in cases:
            measurements = []
            for repetition in range(args.repetitions):
                result = run(ranks, threads, n, args.d, args.k, args.iterations,
                             bind_to_none=args.bind_to_none)
                result.update(kind=kind, repetition=repetition + 1)
                f.write(json.dumps(result) + "\n")
                f.flush()
                measurements.append(result)
                print(kind, f"{ranks}x{threads}", repetition + 1,
                      f"{result['seconds']:.6f}s", flush=True)
            rows.append({"kind": kind, "ranks": ranks, "threads": threads,
                         "n": n, "d": args.d, "k": args.k, "iterations": args.iterations,
                         "median_seconds": statistics.median(x["seconds"] for x in measurements),
                         "min_seconds": min(x["seconds"] for x in measurements),
                         "max_seconds": max(x["seconds"] for x in measurements),
                         "median_communication_seconds": statistics.median(
                             x["communication_seconds"] for x in measurements),
                         "median_inertia": statistics.median(x["inertia"] for x in measurements),
                         "median_label_agreement": statistics.median(
                             x["label_agreement"] for x in measurements)})
    strong_baseline = next(x["median_seconds"] for x in rows
                           if x["kind"] == "strong" and x["ranks"] == x["threads"] == 1)
    weak_baseline = next(x["median_seconds"] for x in rows
                         if x["kind"] == "weak" and x["ranks"] == x["threads"] == 1)
    for row in rows:
        if row["kind"] == "strong":
            row["speedup"] = strong_baseline / row["median_seconds"]
            row["efficiency"] = row["speedup"] / (row["ranks"] * row["threads"])
        else:
            row["speedup"] = weak_baseline / row["median_seconds"]
            row["efficiency"] = row["speedup"]
    with summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {raw} and {summary}")


if __name__ == "__main__":
    main()
