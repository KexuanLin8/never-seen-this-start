"""Progress of queue_remedy.sh: finished episodes per model x condition, and how recently a result was written.

  python progress_remedy.py A        (or B, C)
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path(os.path.join(os.environ.get("W", "work"), "outputs/probe"))
MODELS = ["evo1_ft_orig", "evo1_ft_start", "evo1_ft_mid", "evo1"]  # queue order
MODELS += sorted(p.name for p in (ROOT / "libero").glob("evo1_ft_*") if p.name not in MODELS)  # later variants
MODELS += ["smolvla"] + sorted(p.name for p in (ROOT / "libero").glob("smolvla_ft_*"))
PHASES = {
    "A": [("libero", c, 200) for c in ("C0_full", "T1_start_010", "T2_start_020", "T3_start_030", "T4_start_050")]
    + [("libero_plus", "H0c_full", 156)],
    "B": [("libero", c, 200) for c in ("X1_push_early_mild", "X2_push_early_strong", "X3_push_late_strong",
                                       "C3_state_zero", "S1_bias_1cm", "S2_bias_2cm", "S3_bias_4cm")]
    + [("libero_plus", "P0c_full", 501)],
    "C": [("libero", "K2_chain_half", 200), ("libero", "K1_chain", 200)],
    "H": [("libero", "K3_chain_unseen", 200), ("libero", "K4_chain_seen", 200)],
}


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "A"
    total = done = 0
    recent = []  # modification times of result files, for the episode rate
    for m in MODELS:
        cells = []
        for bench, cond, n in PHASES[phase]:
            files = list((ROOT / bench / m / cond).glob("results_*.jsonl"))
            k = sum(1 for f in files for line in f.read_text(encoding="utf-8").splitlines() if line.strip())
            recent += [f.stat().st_mtime for f in files]
            total += n
            done += min(k, n)
            cells.append(f"{cond}={min(k, n)}/{n}")
        print(f"{m:14s} " + "  ".join(cells))
    print(f"\nphase {phase}: {done}/{total} episodes ({100 * done / total:.0f}%)")
    print("(evo1 already has C0/H0c/B-phase results from the diagnostics; those finish instantly)")
    if recent:
        print(f"last result written {(time.time() - max(recent)) / 60:.0f} min ago "
              "(more than ~10 min while running = something is stuck, check the queue log)")


if __name__ == "__main__":
    main()
