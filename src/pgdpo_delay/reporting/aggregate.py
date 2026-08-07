"""Aggregation: scan outputs/runs/**/manifest.json + metric artifacts into ONE
tidy results.csv (rows = problem, method, seed, metric, value). Figures and
tables consume ONLY this file; they never touch run directories directly, so
adding a method later means writing artifacts in the standard layout and
nothing else."""
import csv, json
from pathlib import Path

def collect(outroot="outputs/runs", dest="outputs/results.csv"):
    rows = []
    for man_path in Path(outroot).glob("**/manifest.json"):
        man = json.loads(man_path.read_text())
        mfile = man_path.parent / "metrics.json"
        if not mfile.exists():
            continue
        metrics = json.loads(mfile.read_text())
        for key, val in metrics.items():
            rows.append(dict(problem=man.get("problem"), method=man.get("method"),
                             seed=(man.get("seeds") or {}).get("train"),
                             config_hash=man.get("config_hash"),
                             metric=key, value=val))
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["problem", "method", "seed",
                                           "config_hash", "metric", "value"])
        w.writeheader(); w.writerows(rows)
    return dest, len(rows)
