"""Arm configurations for the task-chaining evaluation: where the robot is when ANOTHER task of the same
suite has just been finished (final frame of that task's demonstrations).

In deployment the next instruction starts wherever the previous task ended, so "just home first" is not
always available; these end configurations are realistic, task-shaped start states (unlike the random
joint-space directions of --start-perturb).

  python make_chain_starts.py --out $W/outputs/probe/chain_starts.json

Training pool for synth_libero.py --mode chain (disjoint from the evaluation starts):
  python make_chain_starts.py --first 10 --per-task 40 --out $W/outputs/probe/chain_pool_train.json

Output: {suite: {task_id: [[7 joint angles], ...]}} (end configurations of task_id's demos), consumed by
probe_client.py --chain-starts. Run in the `libero` env (py3.8).
"""

import os
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5-dir", default=os.path.join(os.environ.get("W", "work"), "data/libero_hdf5"))
    p.add_argument("--out", default=os.path.join(os.environ.get("W", "work"), "outputs/probe/chain_starts.json"))
    p.add_argument("--per-task", type=int, default=10, help="demos per task whose end configuration is kept")
    p.add_argument("--first", type=int, default=0,
                   help="skip the first demos (0 = evaluation pool; 10 = training pool for synth_libero --mode chain)")
    p.add_argument("--homing-ref", default=os.path.join(os.environ.get("W", "work"), "outputs/probe/homing_ref.json"))
    a = p.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    ref = json.loads(Path(a.homing_ref).read_text()) if Path(a.homing_ref).exists() else {}
    out = {}
    for suite in SUITES:
        ts = benchmark.get_benchmark_dict()[suite]()
        out[suite] = {}
        disp = []
        for tid in range(ts.n_tasks):
            task = ts.get_task(tid)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=64, camera_widths=64)
            env.seed(0)
            env.reset()
            qs = []
            with h5py.File(Path(a.hdf5_dir) / suite / f"{task.name}_demo.hdf5", "r") as h:
                keys = sorted(h["data"].keys(), key=lambda s: int(s.split("_")[1]))[a.first: a.first + a.per_task]
                for k in keys:
                    obs = env.set_init_state(h["data"][k]["states"][-1])  # simulator state at the demo's end
                    qs.append(np.asarray(obs["robot0_joint_pos"], dtype=float).round(5).tolist())
            env.close()
            out[suite][str(tid)] = qs
            if suite in ref:
                disp += [float(np.linalg.norm(np.array(q) - np.array(ref[suite]))) for q in qs]
        msg = f"{suite}: {sum(len(v) for v in out[suite].values())} end configurations"
        if disp:
            msg += (f", joint distance to the nominal start median={np.median(disp):.2f} rad "
                    f"(p10={np.percentile(disp, 10):.2f}, p90={np.percentile(disp, 90):.2f})")
        print(msg, flush=True)
    Path(a.out).write_text(json.dumps(out))
    print("saved", a.out)


if __name__ == "__main__":
    main()
