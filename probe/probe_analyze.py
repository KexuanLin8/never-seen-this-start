"""Cross-model tables for the diagnosis.

Reads <out-root>/<bench>/<model>/<condition>/results_shard*.jsonl (probe_client.py) and prints / saves
<out-root>/probe_report.md:

  D1  clean LIBERO success: model x condition, with the paired drop vs the model's own C0_full
  D2  LIBERO-plus success: model x category for each condition
  D3  action sensitivity on clean C0_full: how much the action changes when one input is removed
      (env action space, comparable across models; grip = fraction of flipped gripper commands)
  D4  robot-init homing test (H0_full vs H1_homing)

  python probe_analyze.py --out-root $W/outputs/probe
"""

import os
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

CONDS = ["C0_full", "C1_lang_cf", "C2_lang_empty", "C3_state_zero", "C6_mask_agent", "C7_mask_wrist"]
CATS = ["camera", "robot", "language", "light", "background", "noise", "layout"]
BRANCHES = ["no_lang", "no_state", "no_agent", "no_wrist"]


def load(root):
    runs = defaultdict(dict)  # (bench, model, cond) -> {(suite, task, init): rec}
    for f in Path(root).glob("*/*/*/results_*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                runs[(r["bench"], r["model"], r["condition"])][(r["suite"], r["task_id"], r["init_idx"])] = r
    return runs


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def sr(recs):
    v = [r["success"] for r in recs]
    return 100 * float(np.mean(v)) if v else float("nan"), len(v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", default=os.path.join(os.environ.get("W", "work"), "outputs/probe"))
    p.add_argument("--first-k", type=int, default=5)
    a = p.parse_args()
    runs = load(a.out_root)
    models = sorted({m for (_, m, _) in runs})
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("## D1 clean LIBERO success % (n) [drop vs own C0_full, paired lost/gained]")
    conds = CONDS + sorted({c for (b, _, c) in runs if b == "libero" and c not in CONDS})
    emit("| model | " + " | ".join(conds) + " |")
    emit("|" + "---|" * (len(conds) + 1))
    for m in models:
        ref = runs.get(("libero", m, "C0_full"), {})
        cells = []
        for c in conds:
            recs = runs.get(("libero", m, c))
            if not recs:
                cells.append("-")
                continue
            s, n = sr(recs.values())
            if c == "C0_full" or not ref:
                lo, hi = wilson(sum(r["success"] for r in recs.values()), n)
                cells.append(f"{s:.1f} ({n}) [{lo:.0f},{hi:.0f}]")
            else:
                common = [k for k in recs if k in ref]
                lost = sum(ref[k]["success"] and not recs[k]["success"] for k in common)
                gain = sum(recs[k]["success"] and not ref[k]["success"] for k in common)
                s0, _ = sr([ref[k] for k in common])
                s1, _ = sr([recs[k] for k in common])
                cells.append(f"{s:.1f} ({n}) [{s1 - s0:+.1f}, -{lost}/+{gain}]")
        emit(f"| {m} | " + " | ".join(cells) + " |")

    emit("\n## D2 LIBERO-plus success % (n) per category")
    for (bench, m, c), recs in sorted(runs.items()):
        if bench != "libero_plus":
            continue
        cells = []
        for cat in CATS:
            sub = [r for r in recs.values() if r["category"] == cat]
            cells.append(f"{sr(sub)[0]:.1f}({len(sub)})" if sub else "-")
        emit(f"- {m}/{c}: " + "  ".join(f"{cat}={x}" for cat, x in zip(CATS, cells))
             + f"  ALL={sr(recs.values())[0]:.1f}({len(recs)})")

    emit(f"\n## D3 action sensitivity on clean C0_full (mean over first {a.first_k} queries; env action space)")
    emit("| model | " + " | ".join(f"{b} pos / grip" for b in BRANCHES) + " |")
    emit("|" + "---|" * (len(BRANCHES) + 1))
    for m in models:
        recs = [r for r in runs.get(("libero", m, "C0_full"), {}).values() if r.get("dists")]
        if not recs:
            continue
        cells = []
        for b in BRANCHES:
            pos = [np.mean([d[b]["pos"] for d in r["dists"][: a.first_k] if b in d]) for r in recs
                   if any(b in d for d in r["dists"][: a.first_k])]
            grip = [np.mean([d[b]["grip"] for d in r["dists"][: a.first_k] if b in d]) for r in recs
                    if any(b in d for d in r["dists"][: a.first_k])]
            cells.append(f"{np.mean(pos):.3f} / {np.mean(grip):.2f}" if pos else "-")
        emit(f"| {m} | " + " | ".join(cells) + " |")

    emit("\n## D4 robot-init homing test (LIBERO-plus 'robot' category, paired)")
    for tag, c0, c1 in (("raw LIBERO-plus prompts", "H0_full", "H1_homing"),
                        ("cleaned prompts", "H0c_full", "H1c_homing")):
        for m in models:
            h0 = runs.get(("libero_plus", m, c0), {})
            h1 = runs.get(("libero_plus", m, c1), {})
            common = [k for k in h0 if k in h1]
            if common:
                s0, n = sr([h0[k] for k in common])
                s1, _ = sr([h1[k] for k in common])
                emit(f"- [{tag}] {m}: no homing {s0:.1f}%  ->  homing {s1:.1f}%  (n={n} paired)")

    emit("\n## D6 effect of the LIBERO-plus prompt suffix (raw P0_full vs cleaned P0c_full, paired per category)")
    for m in models:
        raw = runs.get(("libero_plus", m, "P0_full"), {})
        cln = runs.get(("libero_plus", m, "P0c_full"), {})
        common = [k for k in raw if k in cln]
        if not common:
            continue
        cells = []
        for cat in CATS + ["ALL"]:
            ks = [k for k in common if cat == "ALL" or raw[k]["category"] == cat]
            if ks:
                cells.append(f"{cat}={sr([raw[k] for k in ks])[0]:.1f}->{sr([cln[k] for k in ks])[0]:.1f}")
        emit(f"- {m} (n={len(common)}): " + "  ".join(cells))

    emit("\n## D5 mid-episode push (clean LIBERO, paired vs own C0_full; displacement caused by the push)")
    emit("| model | condition | success % | paired drop | n | joint disp (rad) median | eef disp (m) median |")
    emit("|---|---|---|---|---|---|---|")
    for m in models:
        ref = runs.get(("libero", m, "C0_full"), {})
        for (bench, mm, c), recs in sorted(runs.items()):
            if bench != "libero" or mm != m or not c.startswith("X"):
                continue
            common = [k for k in recs if k in ref]
            s0, _ = sr([ref[k] for k in common])
            s1, n = sr([recs[k] for k in common])
            jd = [r["disturb"]["joint_disp"] for r in recs.values() if r.get("disturb")]
            ed = [r["disturb"]["eef_disp"] for r in recs.values() if r.get("disturb")]
            emit(f"| {m} | {c} | {s1:.1f} | {s1 - s0:+.1f} | {n} | "
                 f"{np.median(jd) if jd else float('nan'):.3f} | {np.median(ed) if ed else float('nan'):.3f} |")

    # ---------- D7: start perturbation vs mid-episode push at MATCHED joint displacement ----------
    ref_path = Path(a.out_root) / "homing_ref.json"
    if ref_path.exists():
        ref_q = json.loads(ref_path.read_text())
        edges = [0.0, 0.15, 0.25, 0.35, 10.0]
        emit("\n## D7 success vs joint displacement: start perturbation (H0c/H0, no homing) vs mid-episode push (X1-X3)")
        emit("| model | source | " + " | ".join(f"{lo:.2f}-{hi:.2f} rad" if hi < 9 else f">{lo:.2f} rad"
                                                for lo, hi in zip(edges[:-1], edges[1:])) + " |")
        emit("|" + "---|" * (len(edges) + 1))
        for m in models:
            start = runs.get(("libero_plus", m, "H0c_full")) or runs.get(("libero_plus", m, "H0_full")) or {}
            s_pairs = [(float(np.linalg.norm(np.array(r["init_joint_pos"]) - np.array(ref_q[r["suite"]]))), r["success"])
                       for r in start.values() if r.get("init_joint_pos") and r["suite"] in ref_q]
            p_pairs = [(r["disturb"]["joint_disp"], r["success"]) for (b, mm, c), recs in runs.items()
                       if b == "libero" and mm == m and c.startswith("X") for r in recs.values() if r.get("disturb")]
            for name, pairs in (("start", s_pairs), ("push", p_pairs)):
                if not pairs:
                    continue
                cells = []
                for lo, hi in zip(edges[:-1], edges[1:]):
                    sel = [s for d, s in pairs if lo <= d < hi]
                    cells.append(f"{100 * np.mean(sel):.0f}% ({len(sel)})" if sel else "-")
                emit(f"| {m} | {name} | " + " | ".join(cells) + " |")

    # ---------- D8: state-bias dose response (clean LIBERO) ----------
    emit("\n## D8 constant eef-position bias in the state (clean LIBERO, paired vs own C0_full)")
    emit("| model | 0 cm (C0) | 1 cm | 2 cm | 4 cm |")
    emit("|---|---|---|---|---|")
    for m in models:
        ref = runs.get(("libero", m, "C0_full"), {})
        if not ref:
            continue
        cells = [f"{sr(ref.values())[0]:.1f}"]
        for c in ("S1_bias_1cm", "S2_bias_2cm", "S3_bias_4cm"):
            recs = runs.get(("libero", m, c))
            if not recs:
                cells.append("-")
                continue
            common = [k for k in recs if k in ref]
            cells.append(f"{sr([recs[k] for k in common])[0]:.1f} ({sr([recs[k] for k in common])[0] - sr([ref[k] for k in common])[0]:+.1f})")
        emit(f"| {m} | " + " | ".join(cells) + " |")

    # ---------- R1/R2: remedy evaluation (queue_remedy.sh), released Evo-1 vs fine-tuned variants ----------
    r_models = []
    for fam in ("evo1", "smolvla"):  # released model, its equal-compute control, then the other variants
        r_models += [m for m in (fam, f"{fam}_ft_orig", f"{fam}_ft_mid", f"{fam}_ft_start") if m in models]
        r_models += sorted(m for m in models if m.startswith(f"{fam}_ft_") and m not in r_models)
    r_conds = [("libero", c) for c in ("C0_full", "T1_start_010", "T2_start_020", "T3_start_030", "T4_start_050",
                                       "X1_push_early_mild", "X2_push_early_strong", "X3_push_late_strong",
                                       "C3_state_zero", "S1_bias_1cm", "S2_bias_2cm", "S3_bias_4cm", "K2_chain_half", "K1_chain", "K3_chain_unseen", "K4_chain_seen")]
    r_conds += [("libero_plus", "H0c_full"), ("libero_plus", "P0c_full")]
    if any("_ft_" in m for m in r_models):
        emit("\n## R1 remedy: success % (n) [95% Wilson CI]")
        emit("| model | " + " | ".join(c for _, c in r_conds) + " |")
        emit("|" + "---|" * (len(r_conds) + 1))
        for m in r_models:
            cells = []
            for b, c in r_conds:
                recs = runs.get((b, m, c))
                if not recs:
                    cells.append("-")
                    continue
                s, n = sr(recs.values())
                lo, hi = wilson(sum(r["success"] for r in recs.values()), n)
                cells.append(f"{s:.1f} ({n}) [{lo:.0f},{hi:.0f}]")
            emit(f"| {m} | " + " | ".join(cells) + " |")

        emit("\n## R2 remedy vs the family's FT-orig (equal-compute baseline), paired episodes: diff [-lost/+gained]")
        emit("| model | " + " | ".join(c for _, c in r_conds) + " |")
        emit("|" + "---|" * (len(r_conds) + 1))
        for m in r_models:
            base = m.split("_ft_")[0] + "_ft_orig"  # each family against its own equal-compute control
            if m == base:
                continue
            cells = []
            for b, c in r_conds:
                ref, recs = runs.get((b, base, c), {}), runs.get((b, m, c), {})
                common = [k for k in recs if k in ref]
                if not common:
                    cells.append("-")
                    continue
                lost = sum(ref[k]["success"] and not recs[k]["success"] for k in common)
                gain = sum(recs[k]["success"] and not ref[k]["success"] for k in common)
                cells.append(f"{sr([recs[k] for k in common])[0] - sr([ref[k] for k in common])[0]:+.1f} "
                             f"[-{lost}/+{gain}]")
            emit(f"| {m} | " + " | ".join(cells) + " |")

    out = Path(a.out_root) / "probe_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
