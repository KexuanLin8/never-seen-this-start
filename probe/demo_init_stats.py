"""How narrow is the initial-configuration distribution of the LIBERO training demos, and do the
LIBERO-plus robot-init perturbations leave it?

Downloads only the parquet state/action tables (no videos) of the LeRobot LIBERO datasets used to
train Evo-1 (IPEC-COMMUNITY/libero_*_no_noops_1.0.0_lerobot) and reports, per suite:
  (1) spread of the demos' initial end-effector position (distance to the suite mean);
  (2) the same for LIBERO-plus robot-init episodes logged by probe_client (init_eef_pos), with the
      nominal reference taken from non-robot LIBERO-plus episodes;
  (3) nearest-neighbour distance from each perturbed initial eef position to ANY training frame of
      the suite, compared with the same distance for nominal starts -> is the perturbed start
      outside the state region the policy has ever been trained on?

Run in the Evo1 env (pandas, pyarrow, huggingface_hub):
  python demo_init_stats.py --probe-root $W/outputs/probe --cache $W/data/libero_lerobot_parquet
"""

import os
import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

DATASETS = {
    "libero_spatial": "IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot",
    "libero_object": "IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot",
    "libero_goal": "IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot",
    "libero_10": "IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot",
}


def q(v, ps=(50, 90, 99)):
    v = np.asarray(v)
    return " ".join(f"p{p}={np.percentile(v, p):.3f}" for p in ps) + f" max={v.max():.3f}" if len(v) else "n=0"


def axisangle_to_quat(aa):
    """(N,3) axis-angle -> (N,4) quaternion (x, y, z, w), robosuite convention."""
    theta = np.linalg.norm(aa, axis=1, keepdims=True)
    axis = np.where(theta > 1e-8, aa / np.maximum(theta, 1e-8), np.array([[1.0, 0.0, 0.0]]))
    return np.concatenate([axis * np.sin(theta / 2), np.cos(theta / 2)], axis=1)


def load_demos(repo, cache):
    local = snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=["data/*/*.parquet", "meta/*"],
                              local_dir=str(Path(cache) / repo.split("/")[1]))
    frames = pd.concat([pd.read_parquet(f, columns=["observation.state", "episode_index", "frame_index"])
                        for f in sorted(glob.glob(f"{local}/data/*/*.parquet"))], ignore_index=True)
    states = np.stack(frames["observation.state"].to_numpy())
    first = frames["frame_index"].to_numpy() == 0
    # all-frame eef positions, all-frame orientations (quaternions), initial eef positions
    return states[:, :3], axisangle_to_quat(states[:, 3:6]), states[first, :3]


def nn_rot(quats, cloud, chunk=512):
    """Smallest rotation angle (rad) from each quaternion to any quaternion of the cloud."""
    out = []
    for i in range(0, len(quats), chunk):
        dot = np.abs(quats[i:i + chunk] @ cloud.T).clip(0, 1)
        out.append(2 * np.arccos(dot.max(axis=1)))
    return np.concatenate(out) if out else np.array([])


def nn_pose(pos, quats, cloud_pos, cloud_q, lam, chunk=256):
    """Nearest training frame under d = |dp| + lam * angle (m, with lam metres per radian)."""
    out = []
    for i in range(0, len(pos), chunk):
        dp = np.linalg.norm(pos[i:i + chunk, None, :] - cloud_pos[None, :, :], axis=-1)
        ang = 2 * np.arccos(np.abs(quats[i:i + chunk] @ cloud_q.T).clip(0, 1))
        out.append((dp + lam * ang).min(axis=1))
    return np.concatenate(out) if out else np.array([])


def load_p1_orient(p1_root):
    """P1 LIBERO-plus runs that logged init_eef_quat: (suite -> robot-init [(pos, quat)], nominal [(pos, quat)])."""
    pert, nominal = defaultdict(list), defaultdict(list)
    for f in glob.glob(f"{p1_root}/libero_plus/*/results_*.jsonl"):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r.get("init_eef_quat") and r.get("init_eef_pos") and r.get("condition") in ("PF_full", "P0_full"):
                    (pert if r["category"] == "robot" else nominal)[r["suite"]].append(
                        (r["init_eef_pos"], r["init_eef_quat"]))
    return pert, nominal


def load_probe(probe_root):
    """robot-init (perturbed) and nominal (non-robot categories) initial eef positions per suite."""
    pert, nominal = defaultdict(list), defaultdict(list)
    for f in glob.glob(f"{probe_root}/libero_plus/*/*/results_*.jsonl"):
        cond = Path(f).parent.name
        if cond not in ("H0_full", "P0_full"):  # no homing, no extra intervention
            continue
        for line in open(f, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r.get("init_eef_pos"):
                    (pert if r["category"] == "robot" else nominal)[r["suite"]].append(r["init_eef_pos"])
    return pert, nominal


def nn_dist(points, cloud, chunk=2048):
    out = []
    for i in range(0, len(points), chunk):
        d = np.linalg.norm(points[i:i + chunk, None, :] - cloud[None, :, :], axis=-1)
        out.append(d.min(axis=1))
    return np.concatenate(out) if out else np.array([])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe-root", default=os.path.join(os.environ.get("W", "work"), "outputs/probe"))
    p.add_argument("--p1-root", default=os.path.join(os.environ.get("W", "work"), "outputs/p1"))
    p.add_argument("--cache", default=os.path.join(os.environ.get("W", "work"), "data/libero_lerobot_parquet"))
    p.add_argument("--lam", type=float, default=0.1, help="metres per radian in the combined pose distance")
    a = p.parse_args()
    pert, nominal = load_probe(a.probe_root)
    pert_q, nominal_q = load_p1_orient(a.p1_root)

    for suite, repo in DATASETS.items():
        all_pos, all_q, init_pos = load_demos(repo, a.cache)
        print(f"\n## {suite}  ({len(init_pos)} demos, {len(all_pos)} frames)")
        P = np.asarray(pert.get(suite, []))
        N = np.asarray(nominal.get(suite, []))
        if len(P) and len(N):
            print(f"(A) position only - NN dist to any training frame (m)")
            print(f"    nominal starts:   {q(nn_dist(N, all_pos))}  n={len(N)}")
            print(f"    perturbed starts: {q(nn_dist(P, all_pos))}  n={len(P)}")
        Pq, Nq = pert_q.get(suite, []), nominal_q.get(suite, [])
        if Pq and Nq:
            for name, rows in (("nominal starts:  ", Nq), ("perturbed starts:", Pq)):
                pos = np.array([r[0] for r in rows])
                qs = np.array([r[1] for r in rows])
                qs /= np.linalg.norm(qs, axis=1, keepdims=True)
                rot = np.degrees(nn_rot(qs, all_q))
                pose = nn_pose(pos, qs, all_pos, all_q, a.lam)
                print(f"(B) {name} orientation NN (deg): {q(rot)} | pose NN (m + {a.lam} m/rad): {q(pose)}  n={len(rows)}")
        else:
            print("(B) no P1 episodes with init_eef_quat for this suite")


if __name__ == "__main__":
    main()
