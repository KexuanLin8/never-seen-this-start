"""Summary of the real-robot evaluation: python analyze_real.py [results/real_eval.jsonl]

Success per model and condition (95% Wilson interval) and the paired comparison between the models: both faced the
same task, cube position and start pose in each trial, so an exact McNemar test applies.
"""

import json
import math
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONDITIONS = ("nominal", "perturbed", "chain")


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def mcnemar(b, c):
    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def main():
    path = HERE / (sys.argv[1] if len(sys.argv) > 1 else "results/real_eval.jsonl")
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    runs = defaultdict(dict)
    for r in rows:
        runs[(r["model"], r["condition"])][r["trial"]] = r
    models = sorted({m for m, _ in runs})
    print(f"{len(rows)} trials; models {models}")
    print(f"{'condition':<11}" + "".join(f"{m:>26}" for m in models))
    for c in CONDITIONS:
        cells = []
        for m in models:
            v = list(runs.get((m, c), {}).values())
            if not v:
                cells.append(f"{'-':>26}")
                continue
            k = sum(x["success"] for x in v)
            lo, hi = wilson(k, len(v))
            cells.append(f"{f'{k}/{len(v)} = {100 * k / len(v):.0f}% [{lo:.0f},{hi:.0f}]':>26}")
        print(f"{c:<11}" + "".join(cells))
    if len(models) == 2:
        a, b = models
        print(f"\npaired: {b} vs {a}")
        for c in CONDITIONS:
            ra, rb = runs.get((a, c), {}), runs.get((b, c), {})
            common = sorted(set(ra) & set(rb))
            gained = sum(rb[t]["success"] and not ra[t]["success"] for t in common)
            lost = sum(ra[t]["success"] and not rb[t]["success"] for t in common)
            print(f"  {c:<10} n={len(common):>2}  {b} only: {gained:>2}  {a} only: {lost:>2}  McNemar p={mcnemar(gained, lost):.3g}")
    print("\nper task (success / trials)")
    for c in CONDITIONS:
        for m in models:
            v = list(runs.get((m, c), {}).values())
            if v:
                print(f"  {c:<10} {m:<11} " + "  ".join(
                    f"{t}: {sum(x['success'] for x in v if x['task'] == t)}/{sum(x['task'] == t for x in v)}" for t in "AB"))
    ms = [r["inference_ms"] for r in rows]
    print(f"\nmean inference {sum(ms) / len(ms):.0f} ms; mean trial length {sum(r['steps'] for r in rows) / len(rows) / 20:.1f} s")


if __name__ == "__main__":
    main()
