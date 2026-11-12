"""Aggregate P1 diagnostic runs into tables.

Reads <out-root>/<bench>/<condition>/results_shard*.jsonl written by p1_client.py and prints
(and saves to <out-root>/p1_report.md):

  T1  clean-LIBERO success per condition, with paired change vs C0_full
      -> language / state / view reliance
  T2  LIBERO-plus success per category for each executed variant
      -> e.g. does masking the agentview help under camera shift? (oracle modality selection)
  T3  AUROC of branch disagreement for separating clean vs each perturbation category
      -> hypothesis A: is disagreement a usable per-modality reliability signal?
  T4  AUROC of branch disagreement for predicting failure inside LIBERO-plus
  T5  action-head attention mass per context segment (padding / masked-view leakage)
  T6  initial end-effector displacement of robot-init perturbations (calibrates state offsets)

Runs with plain python (numpy only), e.g. in the libero_plus env:
  python p1_analyze.py --out-root $W/outputs/p1
"""

import os
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

BRANCHES = ["no_lang", "no_state", "no_agent", "no_wrist"]
CATS = ["camera", "robot", "language", "light", "background", "noise", "layout"]


def load(out_root):
    runs = defaultdict(dict)  # (bench, condition) -> {(suite, task, init): rec}
    for f in Path(out_root).glob("*/*/results_shard*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                runs[(r["bench"], r["condition"])][(r["suite"], r["task_id"], r["init_idx"])] = r
    return runs


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def auroc(pos, neg):
    """P(score_pos > score_neg) with ties counted half (Mann-Whitney U)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):  # average ranks of ties
        idx = allv == v
        ranks[idx] = ranks[idx].mean()
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def ep_feature(rec, branch, key="all", first_k=5):
    d = rec.get("dists") or []
    vals = [step[branch][key] for step in d[:first_k] if branch in step]
    return float(np.mean(vals)) if vals else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", default=os.path.join(os.environ.get("W", "work"), "outputs/p1"))
    p.add_argument("--first-k", type=int, default=5, help="policy steps averaged into an episode score")
    a = p.parse_args()
    runs = load(a.out_root)
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    # ---------- T1 ----------
    emit("## T1 clean LIBERO: success per condition (paired vs C0_full)")
    ref = runs.get(("libero", "C0_full"), {})
    suites = sorted({k[0] for (b, _), recs in runs.items() if b == "libero" for k in recs})
    emit("| condition | " + " | ".join(suites) + " | ALL [95% CI] | n | lost vs C0 | gained vs C0 |")
    emit("|" + "---|" * (len(suites) + 5))
    for (bench, cond), recs in sorted(runs.items()):
        if bench != "libero":
            continue
        cells = []
        for s in suites:
            sub = [r["success"] for k, r in recs.items() if k[0] == s]
            cells.append(f"{100 * np.mean(sub):.1f}" if sub else "-")
        k = sum(r["success"] for r in recs.values())
        n = len(recs)
        lo, hi = wilson(k, n)
        common = [key for key in recs if key in ref]
        lost = sum(ref[key]["success"] and not recs[key]["success"] for key in common)
        gained = sum(recs[key]["success"] and not ref[key]["success"] for key in common)
        emit(f"| {cond} | " + " | ".join(cells) + f" | {100 * k / max(n, 1):.1f} [{lo:.1f},{hi:.1f}] | {n} | {lost} | {gained} |")

    # ---------- T2 ----------
    emit("\n## T2 LIBERO-plus: success per category per executed variant")
    emit("| condition | " + " | ".join(CATS) + " | ALL | n |")
    emit("|" + "---|" * (len(CATS) + 3))
    for (bench, cond), recs in sorted(runs.items()):
        if bench != "libero_plus":
            continue
        cells = []
        for c in CATS:
            sub = [r["success"] for r in recs.values() if r["category"] == c]
            cells.append(f"{100 * np.mean(sub):.1f}({len(sub)})" if sub else "-")
        allv = [r["success"] for r in recs.values()]
        emit(f"| {cond} | " + " | ".join(cells) + f" | {100 * np.mean(allv):.1f} | {len(allv)} |")

    # ---------- T3 / T4 ----------
    clean = [r for r in ref.values() if r.get("dists")]
    plus = [r for r in runs.get(("libero_plus", "P0_full"), {}).values() if r.get("dists")]
    for key in ["all", "pos", "grip"]:
        emit(f"\n## T3 AUROC clean(C0_full) vs category(P0_full), disagreement[{key}] of first {a.first_k} steps")
        emit("| branch | " + " | ".join(CATS) + " |")
        emit("|" + "---|" * (len(CATS) + 1))
        for b in BRANCHES:
            neg = [x for x in (ep_feature(r, b, key, a.first_k) for r in clean) if x is not None]
            cells = []
            for c in CATS:
                pos = [x for x in (ep_feature(r, b, key, a.first_k) for r in plus if r["category"] == c) if x is not None]
                cells.append(f"{auroc(pos, neg):.2f}")
            emit(f"| {b} | " + " | ".join(cells) + " |")

    emit(f"\n## T4 AUROC disagreement -> failure inside LIBERO-plus (P0_full), first {a.first_k} steps")
    emit("| branch | " + " | ".join(CATS) + " | ALL |")
    emit("|" + "---|" * (len(CATS) + 2))
    for b in BRANCHES:
        cells = []
        for c in CATS + ["ALL"]:
            sub = [r for r in plus if c == "ALL" or r["category"] == c]
            fail = [x for x in (ep_feature(r, b, "all", a.first_k) for r in sub if not r["success"]) if x is not None]
            succ = [x for x in (ep_feature(r, b, "all", a.first_k) for r in sub if r["success"]) if x is not None]
            cells.append(f"{auroc(fail, succ):.2f}")
        emit(f"| {b} | " + " | ".join(cells) + " |")

    # ---------- T5 ----------
    emit("\n## T5 action-head cross-attention mass per context segment (first policy step)")
    for (bench, cond), recs in sorted(runs.items()):
        masses = [r["attn"] for r in recs.values() if r.get("attn")]
        if not masses:
            continue
        groups = defaultdict(list)
        for m in masses:
            tot = sum(m.values())
            for lab, w in m.items():
                groups[lab].append(w / tot)
        summary = ", ".join(f"{lab}={100 * np.mean(v):.1f}%" for lab, v in sorted(groups.items()))
        invalid = np.mean([sum(w for lab, w in m.items() if lab == "padding" or lab.startswith("masked"))
                           / sum(m.values()) for m in masses])
        emit(f"- {bench}/{cond} (n={len(masses)}): {summary}; invalid(padding+masked)={100 * invalid:.1f}%")

    # ---------- T6 / T7 ----------
    # Reference = LIBERO-plus episodes of the same suite from NON-robot categories (their robot
    # initial state is nominal). Joint-space displacement is used when logged (newer runs),
    # since LIBERO-plus perturbs joint angles; eef position alone misses orientation/elbow changes.
    plus_all = {}
    for (bench, cond), recs in runs.items():
        if bench == "libero_plus" and cond in ("P0_full", "PF_full"):
            plus_all.update(recs)
    refs = defaultdict(lambda: defaultdict(list))
    for r in plus_all.values():
        if r["category"] != "robot":
            for field in ("init_eef_pos", "init_joint_pos"):
                if r.get(field):
                    refs[field][r["suite"]].append(r[field])
    refs = {f: {s: np.mean(v, axis=0) for s, v in d.items()} for f, d in refs.items()}
    rob = [r for r in plus_all.values() if r["category"] == "robot"]
    for field, unit in (("init_eef_pos", "m"), ("init_joint_pos", "rad")):
        pairs = [(float(np.linalg.norm(np.array(r[field]) - refs[field][r["suite"]])), r["success"])
                 for r in rob if r.get(field) and r["suite"] in refs.get(field, {})]
        emit(f"\n## T6/T7 robot-init: success vs initial displacement in {field} ({unit}), "
             f"reference = non-robot LIBERO-plus episodes, P0_full+PF_full pooled")
        if len(pairs) < 8:
            emit(f"- not enough episodes with {field} (n={len(pairs)})")
            continue
        d = np.array([x[0] for x in pairs])
        sc = np.array([x[1] for x in pairs], float)
        q = np.percentile(d, [10, 50, 90])
        emit(f"- n={len(d)} p10={q[0]:.3f} median={q[1]:.3f} p90={q[2]:.3f}")
        edges = np.unique(np.percentile(d, [0, 25, 50, 75, 100]))
        emit("| displacement bin | n | success % |")
        emit("|---|---|---|")
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            last = i == len(edges) - 2
            m = (d >= lo) & ((d <= hi) if last else (d < hi))  # half-open bins, no double counting
            emit(f"| {lo:.3f}-{hi:.3f} | {int(m.sum())} | {100 * sc[m].mean():.1f} |")
        emit(f"- AUROC displacement -> failure: {auroc(d[sc == 0], d[sc == 1]):.2f}")

    out = Path(a.out_root) / "p1_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
