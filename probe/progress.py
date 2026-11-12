"""Progress bars and a rough ETA for one model's probe experiments.

  python progress.py smolvla            # or evo1 / vla_adapter
  python progress.py smolvla --parallel 4

ETA per condition = remaining episodes x mean seconds per finished episode / parallel clients.
Conditions not started yet use the mean over this model's finished conditions, so the total ETA
is only a rough guide (failure-heavy conditions run longer).
"""

import os
import argparse
import glob
import json
from pathlib import Path

ROOT = os.path.join(os.environ.get("W", "work"), "outputs/probe")
# condition -> (bench, expected episodes) in queue order (queue_<model>.sh, then queue_disturb.sh)
PLAN = [
    ("C0_full", "libero", 200), ("C1_lang_cf", "libero", 200), ("C2_lang_empty", "libero", 200),
    ("C3_state_zero", "libero", 200), ("C6_mask_agent", "libero", 200), ("C7_mask_wrist", "libero", 200),
    ("H0_full", "libero_plus", 156), ("H1_homing", "libero_plus", 156), ("P0_full", "libero_plus", 501),
    ("X1_push_early_mild", "libero", 200), ("X2_push_early_strong", "libero", 200),
    ("X3_push_late_strong", "libero", 200),
    ("H0c_full", "libero_plus", 156), ("H1c_homing", "libero_plus", 156), ("P0c_full", "libero_plus", 501),
    ("S1_bias_1cm", "libero", 200), ("S2_bias_2cm", "libero", 200), ("S3_bias_4cm", "libero", 200),
]


def fmt_h(sec):
    return f"{sec / 3600:5.1f} h" if sec >= 3600 else f"{sec / 60:5.0f} min"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--parallel", type=int, default=4, help="clients running at the same time")
    a = p.parse_args()

    rows, secs_all = [], []
    for cond, bench, total in PLAN:
        recs = {}
        for f in glob.glob(f"{ROOT}/{bench}/{a.model}/{cond}/results_*.jsonl"):
            for line in Path(f).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    recs[(r["suite"], r["task_id"], r["init_idx"])] = r
        secs = [r["seconds"] for r in recs.values()]
        secs_all += secs
        rows.append((cond, len(recs), total, sum(r["success"] for r in recs.values()), secs))

    mean_all = sum(secs_all) / len(secs_all) if secs_all else 60.0
    total_eta = 0.0
    print(f"model={a.model}   (ETA assumes {a.parallel} parallel clients)")
    print(f"{'condition':<22}{'progress':<34}{'done':>10}{'success':>9}{'s/ep':>7}{'ETA':>10}")
    for cond, done, total, succ, secs in rows:
        done_c = min(done, total)
        bar = "#" * int(30 * done_c / total) + "." * (30 - int(30 * done_c / total))
        mean = sum(secs) / len(secs) if secs else mean_all
        eta = max(total - done, 0) * mean / a.parallel
        total_eta += eta
        sr = f"{100 * succ / done:.0f}%" if done else "-"
        print(f"{cond:<22}[{bar}] {done:>4}/{total:<4}{sr:>8}{mean:>7.0f}{fmt_h(eta):>10}")
    print(f"\nremaining (rough): {fmt_h(total_eta)}  = {total_eta / 86400:.1f} days")


if __name__ == "__main__":
    main()
