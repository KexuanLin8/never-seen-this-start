"""Does success depend on how many padding tokens a prompt leaves? (observational check)

Evo-1 pads the VLM sequence to 1024 tokens and its action head attends to the padding
positions (~19% of attention mass; masking them collapses success). If padding acts as an
implicit register, longer instructions (fewer padding tokens) should do worse. This script
counts padding tokens for every LIBERO-plus baseline episode, using the exact InternVL3 prompt
layout of Evo-1 (3 views x 256 image tokens + instruction), and relates them to success.

Run in the Evo1 env (needs the InternVL3 tokenizer, cached in HF_HOME):
  python prompt_len.py --tag evo1_official
"""

import os
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

CATS = ["camera", "robot", "language", "light", "background", "noise", "layout"]


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return gt / (len(pos) * len(neg))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="evo1_official")
    p.add_argument("--out-root", default=os.path.join(os.environ.get("W", "work"), "outputs/lp_subset"))
    p.add_argument("--max-len", type=int, default=1024)
    a = p.parse_args()

    tok = AutoTokenizer.from_pretrained("OpenGVLab/InternVL3-1B", trust_remote_code=True, use_fast=False)
    image_part = "".join(f"Image-{i + 1}: <img>" + "<IMG_CONTEXT>" * 256 + "</img>\n" for i in range(3))

    recs = {}
    for f in (Path(a.out_root) / a.tag).glob("results_shard*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                recs[(r["suite"], r["task_id"])] = r
    cache = {}
    rows = []
    for r in recs.values():
        ins = r["instruction"].strip()
        if ins not in cache:
            cache[ins] = len(tok(image_part + ins)["input_ids"])
        n_tok = cache[ins]
        rows.append((r["suite"], r["category"], a.max_len - n_tok, r["success"]))
    print(f"{len(rows)} episodes, {len(cache)} distinct instructions")

    print("\n## padding tokens per category (median [p10, p90])")
    by_cat = defaultdict(list)
    for s, c, npad, ok in rows:
        by_cat[c].append(npad)
    for c in CATS:
        v = np.array(by_cat[c])
        if len(v):
            print(f"- {c:<11} median={np.median(v):.0f} [{np.percentile(v, 10):.0f}, {np.percentile(v, 90):.0f}]"
                  f"  min={v.min()}  truncated(<0)={int((v < 0).sum())}")

    print("\n## success vs padding-token quartile, per category (low padding = long prompt)")
    for c in CATS + ["ALL"]:
        sub = [(npad, ok) for s, cc, npad, ok in rows if c == "ALL" or cc == c]
        if len(sub) < 8:
            continue
        d = np.array([x[0] for x in sub], float)
        sc = np.array([x[1] for x in sub], float)
        edges = np.unique(np.percentile(d, [0, 25, 50, 75, 100]))
        cells = []
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            m = (d >= lo) & ((d <= hi) if i == len(edges) - 2 else (d < hi))
            cells.append(f"[{lo:.0f},{hi:.0f}] {100 * sc[m].mean():.0f}%(n={int(m.sum())})")
        print(f"- {c:<11} " + "  ".join(cells) + f"  | AUROC(more padding -> success)={auroc(d[sc == 1], d[sc == 0]):.2f}")

    print("\n## within-suite check for 'language' (removes between-suite differences)")
    for suite in sorted({s for s, *_ in rows}):
        sub = [(npad, ok) for s, c, npad, ok in rows if s == suite and c == "language"]
        if len(sub) >= 8:
            d = np.array([x[0] for x in sub], float)
            sc = np.array([x[1] for x in sub], float)
            print(f"- {suite:<15} n={len(sub)} success={100 * sc.mean():.0f}%  "
                  f"AUROC(more padding -> success)={auroc(d[sc == 1], d[sc == 0]):.2f}")


if __name__ == "__main__":
    main()
