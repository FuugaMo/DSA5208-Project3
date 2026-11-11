#!/usr/bin/env python3
"""Record objective quality after different fixed iteration counts."""
import csv
import os
import subprocess
import json
from pathlib import Path

root = Path(__file__).resolve().parent
env = os.environ.copy()
env.update(OMP_NUM_THREADS="2", OMP_DYNAMIC="FALSE")
rows = []
for iterations in (1, 2, 5, 15):
    command = ["mpirun", "-np", "2", str(root / "kmeans"),
               "--n", "200000", "--d", "32", "--k", "16",
               "--iterations", str(iterations)]
    result = subprocess.run(command, env=env, text=True,
                            capture_output=True, check=True)
    rows.append(json.loads(result.stdout.strip().splitlines()[-1]))
with (root / "results" / "convergence.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(root / "results" / "convergence.csv")
