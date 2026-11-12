"""Split the task-chaining start configurations by SOURCE TASK for the held-out-task experiment.

In every suite the even task ids are the "seen" source tasks and the odd ids the held-out ones:
  chain_pool_even.json   synthesis pool: demos 10-49, end configurations of even tasks only
  chain_starts_even.json evaluation starts from seen source tasks (demos 0-9)
  chain_starts_odd.json  evaluation starts from held-out source tasks (demos 0-9), never used for synthesis

  python split_chain_pool.py [--dir $W/outputs/probe]
"""

import os
import argparse
import json
from pathlib import Path


def subset(data, parity):
    return {suite: {t: qs for t, qs in tasks.items() if int(t) % 2 == parity} for suite, tasks in data.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=os.path.join(os.environ.get("W", "work"), "outputs/probe"))
    a = p.parse_args()
    d = Path(a.dir)
    pool = json.loads((d / "chain_pool_train.json").read_text())
    starts = json.loads((d / "chain_starts.json").read_text())
    outputs = {"chain_pool_even.json": subset(pool, 0), "chain_starts_even.json": subset(starts, 0),
               "chain_starts_odd.json": subset(starts, 1)}
    for name, data in outputs.items():
        (d / name).write_text(json.dumps(data))
        print(name, {s: f"{len(t)} tasks, {sum(len(q) for q in t.values())} poses" for s, t in data.items()})


if __name__ == "__main__":
    main()
