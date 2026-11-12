"""Model-agnostic probe server for the cross-model causal input-reliance study.

One websocket protocol, several VLA adapters:
  --model evo1         Evo-1 (MINT-SJTU/Evo-1)                      env: Evo1
  --model vla_adapter  VLA-Adapter (OpenHelix-Team/VLA-Adapter)      env: vla_adapter   (one checkpoint per suite)
  --model lerobot      any LeRobot policy (SmolVLA, pi0.5, ...)      env: lerobot

Request (sent by probe_client.py):
  {"image": [agentview, wrist]  uint8 HxWx3, already rotated 180 deg like every LIBERO pipeline,
   "state": [8]                 eef pos(3) + axis-angle(3) + gripper qpos(2),
   "prompt": str,
   "p1": {"seed": int, "exec": {...}, "branches": [...], "dist_horizon": int}}
  exec / branch spec keys:  mask_views: [0|1]   -> view replaced by a black image
                            state: "zero"       -> state at the normalization centre (normalized state = 0)
                            prompt: str         -> prompt override (branch no_lang uses "")
Response:
  {"actions": T x 7 env-ready actions, "env_ready": true, "default_horizon": int,
   "branch_dist": {branch: {"pos", "rot", "grip"}}}
  Distances are measured in the LIBERO env action space, so they are comparable across models:
  pos / rot = mean L2 of action dims 0:3 / 3:6 over the first dist_horizon steps;
  grip = fraction of those steps whose binarized gripper command differs.
"""

import argparse
import asyncio
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import websockets

BRANCHES = {
    "full": {},
    "no_lang": {"prompt": ""},
    "no_state": {"state": "zero"},
    "no_agent": {"mask_views": [0]},
    "no_wrist": {"mask_views": [1]},
}


def seed_all(seed):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32))


def masked_views(agent, wrist, spec):
    views = [agent, wrist]
    for v in spec.get("mask_views", []):
        views[v] = np.zeros_like(views[v])
    return views


# --------------------------------------------------------------------------------------------
class Evo1Adapter:
    """Evo-1 through its own server utilities (448 px, 3 views with a masked dummy view)."""

    default_horizon = 15

    def __init__(self, a):
        sys.path.insert(0, os.path.join(a.evo1_root, "Evo_1"))
        from scripts.Evo1_server import decode_image_from_list, load_model_and_normalizer

        self.decode = decode_image_from_list
        self.model, self.normalizer = load_model_and_normalizer(a.ckpt)
        self.arm_key, self.dataset_key = a.arm_key, a.dataset_key
        self.native_mask = a.evo1_native_mask

    @torch.no_grad()
    def predict(self, agent, wrist, state, prompt, spec, seed):
        views = masked_views(agent, wrist, spec)
        images = [self.decode(v) for v in views] + [self.decode(np.zeros((448, 448, 3), np.uint8))]
        image_mask = [1, 1, 0]
        if self.native_mask:  # P1 semantics: masked view also removed from VLM attention
            for v in spec.get("mask_views", []):
                image_mask[v] = 0
        st = torch.tensor(state, dtype=torch.float32, device="cuda").reshape(1, -1)
        st = self.normalizer.normalize_state(st, self.arm_key, self.dataset_key).float()
        if spec.get("state") == "zero":
            st = torch.zeros_like(st)
        seed_all(seed)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            act = self.model.run_inference(
                images=images, image_mask=torch.tensor(image_mask, dtype=torch.int32, device="cuda"),
                prompt=spec.get("prompt", prompt), state_input=st,
                action_mask=torch.tensor([[1] * 7 + [0] * 17], dtype=torch.int32, device="cuda"))
        act = self.normalizer.denormalize_action(act.float().reshape(-1, 24), self.arm_key, self.dataset_key)
        a7 = act[:, :7].cpu().numpy()
        cont = a7.copy()
        cont[:, 6] = 1.0 - 2.0 * a7[:, 6]  # Evo-1 gripper: >0.5 = open -> env -1
        ready = a7.copy()
        ready[:, 6] = np.where(a7[:, 6] > 0.5, -1.0, 1.0)
        return cont, ready


# --------------------------------------------------------------------------------------------
class VLAAdapterAdapter:
    """VLA-Adapter through its official LIBERO eval utilities (224 px, 2 views, 8-step chunks)."""

    default_horizon = 8

    def __init__(self, a):
        sys.path.insert(0, a.vla_adapter_root)
        os.chdir(a.vla_adapter_root)  # the repo resolves some paths relative to its root
        import experiments.robot.openvla_utils as ou
        from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model, process_action
        from experiments.robot.robot_utils import get_action, get_image_resize_size

        self.ou, self.get_action, self.process_action = ou, get_action, process_action
        self.cfg = GenerateConfig(pretrained_checkpoint=a.ckpt, task_suite_name=a.suite,
                                  use_proprio=True, num_images_in_input=2, use_film=False,
                                  use_pro_version=a.vla_pro)
        (self.model, self.action_head, self.proprio_projector,
         self.noisy_action_projector, self.processor) = initialize_model(self.cfg)
        self.resize_size = get_image_resize_size(self.cfg)
        self._orig_norm = ou.normalize_proprio

    @torch.no_grad()
    def predict(self, agent, wrist, state, prompt, spec, seed):
        views = masked_views(agent, wrist, spec)
        obs = {"full_image": self.ou.resize_image_for_policy(views[0], self.resize_size),
               "wrist_image": self.ou.resize_image_for_policy(views[1], self.resize_size),
               "state": np.asarray(state, dtype=np.float64)}
        if spec.get("state") == "zero":
            self.ou.normalize_proprio = lambda p, stats: np.zeros_like(p)
        try:
            seed_all(seed)
            acts = self.get_action(self.cfg, self.model, obs, spec.get("prompt", prompt), processor=self.processor,
                                   action_head=self.action_head, proprio_projector=self.proprio_projector,
                                   noisy_action_projector=self.noisy_action_projector,
                                   use_film=self.cfg.use_film, use_minivlm=self.cfg.use_minivlm)
        finally:
            self.ou.normalize_proprio = self._orig_norm
        raw = np.stack([np.asarray(x, dtype=np.float64) for x in acts])[:, :7]
        cont = raw.copy()
        cont[:, 6] = 1.0 - 2.0 * raw[:, 6]  # [0,1] -> [-1,1] then inverted, as in process_action
        ready = np.stack([self.process_action(x.copy(), self.cfg.model_family) for x in raw])
        return cont, ready


# --------------------------------------------------------------------------------------------
class LeRobotAdapter:
    """Any LeRobot policy with its saved pre/post-processors (SmolVLA, pi0.5, ...)."""

    def __init__(self, a):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        cfg = PreTrainedConfig.from_pretrained(a.ckpt)
        self.policy = get_policy_class(cfg.type).from_pretrained(a.ckpt).to("cuda").eval()
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=a.ckpt)
        self.agent_key, self.wrist_key = a.agent_key, a.wrist_key
        self.default_horizon = int(getattr(self.policy.config, "n_action_steps", 10))
        self.state_mean = self._find_state_mean()
        print(f"[lerobot] type={cfg.type} n_action_steps={self.default_horizon} "
              f"inputs={list(self.policy.config.input_features)} state_mean_found={self.state_mean is not None}",
              flush=True)

    def _find_state_mean(self):
        """Raw state whose normalized value is 0 (MEAN_STD: the mean; bounds and quantiles: the mid-point)."""
        mode = getattr(self.policy.config.normalization_mapping.get("STATE"), "value", "")
        bounds = {"QUANTILES": ("q01", "q99"), "QUANTILE10": ("q10", "q90")}.get(str(mode).upper())  # pi0.5
        for step in getattr(self.pre, "steps", []):
            stats = getattr(step, "stats", None)
            if stats and "observation.state" in stats:
                s = stats["observation.state"]
                if bounds and all(k in s for k in bounds):
                    return ((torch.as_tensor(s[bounds[0]]) + torch.as_tensor(s[bounds[1]])) / 2).float().reshape(1, -1)
                if "mean" in s:
                    return torch.as_tensor(s["mean"]).float().reshape(1, -1)
                if "min" in s and "max" in s:
                    return ((torch.as_tensor(s["min"]) + torch.as_tensor(s["max"])) / 2).float().reshape(1, -1)
        return None

    @staticmethod
    def _img(x):
        return torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float().div(255.0).unsqueeze(0)

    @torch.no_grad()
    def predict(self, agent, wrist, state, prompt, spec, seed):
        views = masked_views(agent, wrist, spec)
        st = torch.tensor(state, dtype=torch.float32).reshape(1, -1)
        if spec.get("state") == "zero":
            if self.state_mean is None:
                raise RuntimeError("normalizer stats for observation.state not found")
            st = self.state_mean.clone()
        batch = {self.agent_key: self._img(views[0]), self.wrist_key: self._img(views[1]),
                 "observation.state": st, "task": [spec.get("prompt", prompt)]}
        batch = self.pre(batch)
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        seed_all(seed)
        chunk = self.policy.predict_action_chunk(batch)  # (1, T, A) normalized
        steps = [self.post(chunk[:, t]) for t in range(chunk.shape[1])]
        acts = torch.stack([s.reshape(-1) for s in steps]).float().cpu().numpy()[:, :7]
        return acts.copy(), acts.copy()  # LeRobot LIBERO actions are already in env convention


ADAPTERS = {"evo1": Evo1Adapter, "vla_adapter": VLAAdapterAdapter, "lerobot": LeRobotAdapter}


# --------------------------------------------------------------------------------------------
def branch_distance(c, ref, h):
    c, ref = c[:h], ref[:h]
    n = min(len(c), len(ref))
    c, ref = c[:n], ref[:n]
    return {"pos": float(np.linalg.norm(c[:, :3] - ref[:, :3], axis=1).mean()),
            "rot": float(np.linalg.norm(c[:, 3:6] - ref[:, 3:6], axis=1).mean()),
            "grip": float((np.sign(c[:, 6]) != np.sign(ref[:, 6])).mean())}


def handle(adapter, data):
    agent = np.asarray(data["image"][0], dtype=np.uint8)
    wrist = np.asarray(data["image"][1], dtype=np.uint8)
    state, prompt = data["state"], data["prompt"]
    p1 = data.get("p1", {})
    seed, exec_spec = p1.get("seed"), p1.get("exec", {})
    h = int(p1.get("dist_horizon") or adapter.default_horizon)

    cont, ready = adapter.predict(agent, wrist, state, prompt, exec_spec, seed)
    out = {"actions": ready.tolist(), "env_ready": True, "default_horizon": adapter.default_horizon}
    branches = p1.get("branches", [])
    if branches:
        conts = {"full": cont if exec_spec == {} else adapter.predict(agent, wrist, state, prompt, {}, seed)[0]}
        for name in branches:
            conts[name] = adapter.predict(agent, wrist, state, prompt, BRANCHES[name], seed)[0]
        out["branch_dist"] = {n: branch_distance(c, conts["full"], h) for n, c in conts.items() if n != "full"}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(ADAPTERS), required=True)
    p.add_argument("--ckpt", required=True, help="checkpoint dir or HF repo id")
    p.add_argument("--port", type=int, default=9200)
    # evo1
    p.add_argument("--evo1-root", default=os.path.join(os.environ.get("W", "work"), "code/Evo-1"))
    p.add_argument("--arm-key", default="libero_robot")
    p.add_argument("--dataset-key", default="libero_4_datasets")
    p.add_argument("--evo1-native-mask", action="store_true",
                   help="also drop masked views from Evo-1's VLM attention (P1 semantics)")
    # vla_adapter
    p.add_argument("--vla-adapter-root", default=os.path.join(os.environ.get("W", "work"), "code/VLA-Adapter"))
    p.add_argument("--suite", default="libero_spatial", help="vla_adapter: the suite its checkpoint was trained on")
    p.add_argument("--vla-pro", type=lambda s: s.lower() == "true", default=True)
    # lerobot
    p.add_argument("--agent-key", default="observation.images.image")
    p.add_argument("--wrist-key", default="observation.images.image2")
    a = p.parse_args()

    adapter = ADAPTERS[a.model](a)
    print(f"[{a.model}] default_horizon={adapter.default_horizon} "
          f"GPU mem {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)

    def handle_with_retry(data):
        # A transient CUDA OOM / cuDNN failure must not close the connection (that kills the client),
        # so free the cache and retry a few times before giving up.
        for attempt in range(3):
            try:
                return handle(adapter, data)
            except RuntimeError as exc:  # torch.cuda.OutOfMemoryError is a RuntimeError subclass
                print(f"[retry {attempt + 1}/3] {type(exc).__name__}: {str(exc)[:200]}", flush=True)
                torch.cuda.empty_cache()
                if attempt == 2:
                    raise

    async def handler(ws):
        async for message in ws:
            await ws.send(json.dumps(handle_with_retry(json.loads(message))))

    async def run():
        async with websockets.serve(handler, "0.0.0.0", a.port, max_size=100_000_000,
                                    ping_interval=None, ping_timeout=None):
            print(f"probe server ({a.model}) ready on ws://0.0.0.0:{a.port}", flush=True)
            await asyncio.Future()

    asyncio.run(run())


if __name__ == "__main__":
    main()
