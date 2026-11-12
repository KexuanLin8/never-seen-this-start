"""P1 diagnostic server for Evo-1 (drop-in replacement for Evo1_server.py).

Protocol-compatible: a request without a "p1" key is answered exactly like the
official server (a plain 50x24 action list), so official clients keep working.

A request carrying "p1": {...} may additionally ask for
  seed          int   -> torch.manual_seed before every forward pass, so all
                         branches of one request share the same flow noise
  exec          dict  -> interventions on the executed policy
                         {"mask_views": [0|1], "state": "orig"|"zero",
                          "ctx_mask": ["padding", "masked_view", ...]  (key_padding_mask in the
                                       action-head cross-attention for those context segments),
                          "trim_padding": true  (no padding to 1024 tokens; also changes the VLM pass),
                          "pad_to": N  (pad the VLM sequence to N tokens instead of 1024)}
                         (language interventions are done client-side via the prompt)
  branches      list  -> extra counterfactual forward passes whose action chunks
                         are compared with the "full" branch (logging only):
                         full, no_lang, no_state, no_agent, no_wrist
  dist_horizon  int   -> how many chunk steps enter the branch distances (default 15)
  return_attn   bool  -> action-head cross-attention mass per context segment
and gets back {"actions": [...], "branch_dist": {...}, "attn": {...}}.

Evo-1 source is untouched; this file only imports from it.
Run from Evo-1/Evo_1 in the Evo1 env:
  python $W/code/pilot/p1_server.py --ckpt $W/ckpt/Evo1_LIBERO --port 9100
"""

import argparse
import asyncio
import json
import os
import sys

import torch
import websockets

sys.path.insert(0, os.path.join(os.environ.get("EVO1_ROOT", os.path.join(os.environ.get("W", "work"), "code/Evo-1")), "Evo_1"))
from scripts.Evo1_server import decode_image_from_list, load_model_and_normalizer  # noqa: E402

BRANCHES = {
    "full": {},
    "no_lang": {"prompt": ""},
    "no_state": {"state": "zero"},
    "no_agent": {"mask_views": [0]},
    "no_wrist": {"mask_views": [1]},
}


class AttnRecorder:
    """Accumulates head-averaged cross-attention weights of the action head and can
    inject a key_padding_mask into every action-head cross-attention call."""

    def __init__(self, model):
        self.active = False
        self.sum = None
        self.count = 0
        self.key_padding_mask = None  # (1, L_k) bool, True = ignore
        for block in model.action_head.transformer_blocks:
            block.attn.register_forward_hook(self._hook)
            block.attn.register_forward_pre_hook(self._pre_hook, with_kwargs=True)

    def _pre_hook(self, module, args, kwargs):
        if self.key_padding_mask is None:
            return None
        kwargs = dict(kwargs)
        kwargs["key_padding_mask"] = self.key_padding_mask
        return args, kwargs

    def _hook(self, module, inputs, output):
        if not self.active or output[1] is None:
            return
        w = output[1].float().mean(dim=(0, 1))  # (L_k,) averaged over batch and action queries
        self.sum = w if self.sum is None else self.sum + w
        self.count += 1

    def start(self):
        self.active, self.sum, self.count = True, None, 0

    def stop(self):
        self.active = False
        return None if self.sum is None else (self.sum / self.count).cpu()


def context_segments(model, prompt, image_mask):
    """Label every action-head context position: text / view_i / masked_view_i / padding / state.

    Re-tokenizes the prompt exactly like InternVL3Embedder (one 448px tile per view,
    right padding to max_text_length, state token appended last by the action head).
    """
    emb = model.embedder
    prompts = emb._build_multimodal_prompt([[1, 1, 1]], [prompt])
    tok = emb.tokenizer(prompts, padding="max_length", truncation=True, max_length=emb.max_text_length)
    ids = torch.tensor(tok["input_ids"][0])
    valid = torch.tensor(tok["attention_mask"][0]).bool()
    is_img = ids == emb.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    labels = ["text" if v else "padding" for v in valid.tolist()]
    img_pos = torch.nonzero(is_img).flatten().tolist()
    per_view = model.embedder.model.num_image_token
    for k, pos in enumerate(img_pos):
        view = k // per_view
        labels[pos] = f"view{view}" if int(image_mask[view]) else f"masked_view{view}"
    labels.append("state")
    return labels


class P1Server:
    def __init__(self, ckpt, arm_key, dataset_key):
        self.model, self.normalizer = load_model_and_normalizer(ckpt)
        self.arm_key, self.dataset_key = arm_key, dataset_key
        self.attn = AttnRecorder(self.model)

    @torch.no_grad()
    def forward(self, images, image_mask, prompt, norm_state, action_mask, spec, seed):
        """One forward pass under an intervention spec; returns normalized (50, 24) chunk."""
        images = list(images)
        image_mask = image_mask.clone()
        for v in spec.get("mask_views", []):
            images[v] = torch.zeros_like(images[v])
            image_mask[v] = 0
        state = torch.zeros_like(norm_state) if spec.get("state") == "zero" else norm_state
        prompt = spec.get("prompt", prompt)
        emb = self.model.embedder
        orig_len = emb.max_text_length
        try:
            if spec.get("trim_padding"):
                full = emb._build_multimodal_prompt([[1, 1, 1]], [prompt])
                emb.max_text_length = len(emb.tokenizer(full)["input_ids"][0])
            elif spec.get("pad_to"):  # dose-response on the number of padding tokens
                emb.max_text_length = int(spec["pad_to"])
            if spec.get("ctx_mask"):
                labels = context_segments(self.model, prompt, image_mask.tolist())
                drop = set(spec["ctx_mask"])
                kpm = [lab in drop or (lab.startswith("masked_view") and "masked_view" in drop) for lab in labels]
                self.attn.key_padding_mask = torch.tensor([kpm], dtype=torch.bool, device="cuda")
            if seed is not None:
                torch.manual_seed(seed)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                action = self.model.run_inference(images=images, image_mask=image_mask, prompt=prompt,
                                                  state_input=state, action_mask=action_mask)
        finally:
            emb.max_text_length = orig_len
            self.attn.key_padding_mask = None
        return action.float().reshape(-1, 24)

    def denorm(self, chunk):
        return self.normalizer.denormalize_action(chunk, self.arm_key, self.dataset_key)

    def handle(self, data):
        arm_key = data.get("arm_key", self.arm_key)
        dataset_key = data.get("dataset_key", self.dataset_key)
        images = [decode_image_from_list(img) for img in data["image"]]
        state = torch.tensor(data["state"], dtype=torch.float32, device="cuda").reshape(1, -1)
        norm_state = self.normalizer.normalize_state(state, arm_key, dataset_key).float()
        image_mask = torch.tensor(data["image_mask"], dtype=torch.int32, device="cuda")
        action_mask = torch.tensor([data["action_mask"]], dtype=torch.int32, device="cuda")
        prompt = data["prompt"]
        p1 = data.get("p1")

        if p1 is None:  # official protocol
            chunk = self.forward(images, image_mask, prompt, norm_state, action_mask, {}, None)
            return self.denorm(chunk).cpu().numpy().tolist()

        seed = p1.get("seed")
        exec_spec = p1.get("exec", {})
        want_attn = bool(p1.get("return_attn", False))
        h = int(p1.get("dist_horizon", 15))
        valid = action_mask[0].bool()

        if want_attn:
            self.attn.start()
        exec_chunk = self.forward(images, image_mask, prompt, norm_state, action_mask, exec_spec, seed)
        out = {"actions": self.denorm(exec_chunk).cpu().numpy().tolist()}
        if want_attn:
            weights = self.attn.stop()
            if weights is not None:
                eff_mask = image_mask.clone()
                for v in exec_spec.get("mask_views", []):
                    eff_mask[v] = 0
                emb = self.model.embedder
                orig_len = emb.max_text_length
                if exec_spec.get("trim_padding"):
                    full = emb._build_multimodal_prompt([[1, 1, 1]], [exec_spec.get("prompt", prompt)])
                    emb.max_text_length = len(emb.tokenizer(full)["input_ids"][0])
                elif exec_spec.get("pad_to"):
                    emb.max_text_length = int(exec_spec["pad_to"])
                labels = context_segments(self.model, exec_spec.get("prompt", prompt), eff_mask.tolist())
                emb.max_text_length = orig_len
                mass = {}
                for lab, w in zip(labels, weights.tolist()):
                    mass[lab] = mass.get(lab, 0.0) + w
                out["attn"] = mass

        branches = p1.get("branches", [])
        if branches:
            chunks = {}
            for name in set(branches) | {"full"}:
                spec = BRANCHES[name]
                chunks[name] = exec_chunk if (spec == exec_spec) else self.forward(
                    images, image_mask, prompt, norm_state, action_mask, spec, seed)
            ref = chunks["full"][:h][:, valid]
            dist = {}
            for name, c in chunks.items():
                if name == "full":
                    continue
                d = c[:h][:, valid] - ref  # normalized action space, valid dims only
                dist[name] = {
                    "all": d.norm(dim=-1).mean().item(),
                    "pos": d[:, :3].norm(dim=-1).mean().item(),
                    "rot": d[:, 3:6].norm(dim=-1).mean().item(),
                    "grip": d[:, 6:7].abs().mean().item() if d.shape[1] > 6 else 0.0,
                }
            out["branch_dist"] = dist
        return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--port", type=int, default=9100)
    p.add_argument("--arm_key", default="libero_robot")
    p.add_argument("--dataset_key", default="libero_4_datasets")
    a = p.parse_args()

    server = P1Server(a.ckpt, a.arm_key, a.dataset_key)
    print("GPU mem after load: %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9), flush=True)

    async def handler(ws):
        async for message in ws:
            await ws.send(json.dumps(server.handle(json.loads(message))))

    async def run():
        async with websockets.serve(handler, "0.0.0.0", a.port, max_size=100_000_000,
                                    ping_interval=None, ping_timeout=None):
            print(f"P1 server ready on ws://0.0.0.0:{a.port}", flush=True)
            await asyncio.Future()

    asyncio.run(run())


if __name__ == "__main__":
    main()
