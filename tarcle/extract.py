"""Stage 1: causal head identification and FV extraction. GPU code — imports
torch, and stage-2 geometry must never import this module.

Two conventions hold everywhere in here, and both are load-bearing on an 8GB
card (docs/decisions.md D6):

1. **Left padding with explicit position_ids.** The last real token is then
   always index -1, for every prompt in a batch regardless of length. That makes
   `logits_to_keep=1` correct on every forward. Without it, full-sequence logits
   at batch 24 cost 862MB (plus ~1.7GB more upcast for log_softmax) and OOM the
   GPU on their own. Left padding needs position_ids passed explicitly: the
   model derives them from a plain arange otherwise, which is wrong when the pad
   sits on the left.
2. **Hooks capture one position on selected modules, never full activations.**
   28 layers at one position is 4MB; all layers at all positions is 578MB, which
   is the entire headroom (CLAUDE.md hook rule).

The head sweep's batching trick: all 24 heads of a layer are patched in a single
forward by replicating the prompt 24x and writing head h into batch slot h.
Batch elements are independent, so this is exact rather than an approximation.
Layers cannot be batched together — patching at layer l changes everything
downstream of it.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import prompts as P
from .pilot import git_commit

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


@dataclass
class ExtractConfig:
    run_name: str
    model: str
    device: str = "cuda"
    dtype: str = "bfloat16"
    batch_size: int = 24

    family: str = "months"  # prompts.py variant
    ks: list[int] = field(default_factory=lambda: list(range(12)))
    shots: int = 16
    n_prompts: int = 100
    seed: int = 0

    # Causal head identification (docs/decisions.md D3)
    head_id_ks: list[int] = field(default_factory=lambda: [1, 2, 3, 6, 9, 11])
    head_id_n_prompts: int = 12
    head_id_confirm_cells: int = 30
    head_id_confirm_n_prompts: int = 40
    top_heads: int = 10

    results_dir: str = "results/fv"


def load_config(path: Path) -> ExtractConfig:
    return ExtractConfig(**json.loads(path.read_text(encoding="utf-8")))


def config_sha(config: ExtractConfig) -> str:
    return hashlib.sha256(
        json.dumps(dataclasses.asdict(config), sort_keys=True).encode()
    ).hexdigest()


# --------------------------------------------------------------------------
# architecture adapter: llama (Linear o_proj) and gpt2 (Conv1D c_proj), so the
# CI path can run the whole pipeline on gpt2/CPU per CLAUDE.md
# --------------------------------------------------------------------------


@dataclass
class Arch:
    blocks: list  # transformer blocks in order; block output[0] is the residual stream
    attn_out: list  # each block's attention output projection module
    w_out: list  # (d_model, n_heads*head_dim) per layer: columns are head slices
    n_layers: int
    n_heads: int
    head_dim: int
    d_model: int


def describe(model) -> Arch:
    cfg = model.config
    if hasattr(model, "model") and hasattr(model.model, "layers"):  # llama-like
        blocks = list(model.model.layers)
        attn_out = [b.self_attn.o_proj for b in blocks]
        w_out = [m.weight for m in attn_out]  # Linear: (out, in) = (d_model, d_attn)
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):  # gpt2
        blocks = list(model.transformer.h)
        attn_out = [b.attn.c_proj for b in blocks]
        w_out = [m.weight.T for m in attn_out]  # Conv1D stores (in, out)
    else:
        raise ValueError(f"unsupported architecture: {type(model).__name__}")
    n_heads = cfg.num_attention_heads
    d_model = cfg.hidden_size
    return Arch(
        blocks=blocks,
        attn_out=attn_out,
        w_out=w_out,
        n_layers=len(blocks),
        n_heads=n_heads,
        head_dim=getattr(cfg, "head_dim", None) or d_model // n_heads,
        d_model=d_model,
    )


def load_model(config: ExtractConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device {config.device!r} requested but CUDA is unavailable")
    dtype = DTYPES[config.dtype]
    if dtype == torch.bfloat16 and config.device.startswith("cuda"):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 unsupported on this GPU; CLAUDE.md rule 4 blocks "
                               "any lower precision for geometry-bearing runs")
    tok = AutoTokenizer.from_pretrained(config.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # see module docstring
    model = AutoModelForCausalLM.from_pretrained(
        config.model, dtype=dtype, attn_implementation="sdpa"
    )
    model.to(config.device).eval()
    model.config.use_cache = False
    if model.dtype != dtype:  # prereg §5: a geometry run below bf16 is discarded
        raise RuntimeError(f"model loaded as {model.dtype}, expected {dtype}")
    return model, tok


def encode(tok, prompts: list[str], device: str) -> dict:
    """Left-padded batch with explicit position_ids (see module docstring)."""
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    mask = enc["attention_mask"]
    enc["position_ids"] = (mask.cumsum(-1) - 1).clamp(min=0)
    return enc


def first_token_id(tok, word: str) -> int:
    """Targets are scored as continuations of '...A:', hence the leading space.
    AIE uses the first target token (Todd et al. score the answer token)."""
    return tok.encode(" " + word, add_special_tokens=False)[0]


# --------------------------------------------------------------------------
# capture and patching
# --------------------------------------------------------------------------


class LastPositionHeads:
    """Forward-pre-hooks on every attention output projection, recording that
    module's input at the final position only, reshaped to (B, n_heads, head_dim).

    The o_proj input is the concatenation of the heads' attention outputs, so
    slicing it by head and pushing the slice through the matching columns of
    W_O gives that head's additive contribution to the residual stream — which
    is what a Todd-style function vector is a sum of.
    """

    def __init__(self, arch: Arch):
        self.arch = arch
        self.acts: dict[int, torch.Tensor] = {}
        self._handles: list = []

    def __enter__(self):
        for layer, module in enumerate(self.arch.attn_out):
            self._handles.append(
                module.register_forward_pre_hook(self._make_hook(layer))
            )
        return self

    def _make_hook(self, layer: int):
        def hook(_module, args):
            x = args[0]
            self.acts[layer] = (
                x[:, -1, :].detach().view(-1, self.arch.n_heads, self.arch.head_dim)
            )

        return hook

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False

    def stacked(self) -> torch.Tensor:
        """(B, n_layers, n_heads, head_dim)"""
        return torch.stack([self.acts[i] for i in range(self.arch.n_layers)], dim=1)


class PatchHeadsDiagonal:
    """Patch head h of one layer into batch slot h, at the final position.

    Batch size must equal n_heads: slot h receives head h's mean task activation
    and nothing else, so one forward yields all of a layer's per-head causal
    effects independently.
    """

    def __init__(self, arch: Arch, layer: int, abar_layer: torch.Tensor):
        self.arch = arch
        self.layer = layer
        self.abar_layer = abar_layer  # (n_heads, head_dim)
        self._handle = None

    def __enter__(self):
        self._handle = self.arch.attn_out[self.layer].register_forward_pre_hook(
            self._hook
        )
        return self

    def _hook(self, _module, args):
        x = args[0]
        b, t, _ = x.shape
        if b != self.arch.n_heads:
            raise ValueError(f"batch {b} != n_heads {self.arch.n_heads}")
        x = x.view(b, t, self.arch.n_heads, self.arch.head_dim).clone()
        idx = torch.arange(self.arch.n_heads, device=x.device)
        x[idx, -1, idx, :] = self.abar_layer.to(x.dtype)
        return (x.view(b, t, -1),) + args[1:]

    def __exit__(self, *exc):
        self._handle.remove()
        return False


@torch.inference_mode()
def mean_head_activations(model, tok, arch: Arch, prompts: list[str], batch_size: int):
    """(n_layers, n_heads, head_dim) mean over prompts, at the last position."""
    total, count = None, 0
    for start in range(0, len(prompts), batch_size):
        enc = encode(tok, prompts[start : start + batch_size], model.device)
        with LastPositionHeads(arch) as cap:
            model(**enc, logits_to_keep=1)
        acts = cap.stacked().float()
        total = acts.sum(0) if total is None else total + acts.sum(0)
        count += acts.shape[0]
    return total / count


@torch.inference_mode()
def target_probs(model, tok, prompts: list[str], targets: list[int], batch_size: int):
    """p(first target token) at the final position, one per prompt."""
    out = []
    for start in range(0, len(prompts), batch_size):
        enc = encode(tok, prompts[start : start + batch_size], model.device)
        logits = model(**enc, logits_to_keep=1).logits[:, -1].float()
        probs = torch.softmax(logits, dim=-1)
        for row, tgt in enumerate(targets[start : start + batch_size]):
            out.append(probs[row, tgt].item())
    return np.array(out)


@torch.inference_mode()
def patched_probs_one_layer(model, tok, arch, prompt, target, abar_layer, layer):
    """(n_heads,) p(target) with head h patched, for every h in one forward."""
    enc = encode(tok, [prompt] * arch.n_heads, model.device)
    with PatchHeadsDiagonal(arch, layer, abar_layer):
        logits = model(**enc, logits_to_keep=1).logits[:, -1].float()
    return torch.softmax(logits, dim=-1)[:, target].cpu().numpy()


# --------------------------------------------------------------------------
# the sweep
# --------------------------------------------------------------------------


def head_id_samples(config: ExtractConfig):
    """(k, clean items, corrupted items) for the head-identification k subset."""
    out = []
    for k in config.head_id_ks:
        clean = P.make_prompt_set(
            config.family, k, config.head_id_n_prompts, config.shots, config.seed,
            stratum="heldout",
        )
        corrupted = [P.corrupt_labels(it, config.seed, i) for i, it in enumerate(clean)]
        out.append((k, clean, corrupted))
    return out


def sweep(model, tok, arch: Arch, config: ExtractConfig, layers=None, log=print):
    """AIE[l, h] = mean over (k, prompt) of p_patched - p_corrupted.

    Returns (aie, aie_se, n_samples, per_k_aie, baseline_mean).
    """
    layers = list(range(arch.n_layers)) if layers is None else list(layers)
    samples = head_id_samples(config)

    abar, corrupted_prompts, target_ids, baselines, ks_flat = {}, [], [], [], []
    for k, clean, corrupted in samples:
        abar[k] = mean_head_activations(
            model, tok, arch, [it.prompt for it in clean], config.batch_size
        )
        tgts = [first_token_id(tok, it.target) for it in corrupted]
        base = target_probs(
            model, tok, [it.prompt for it in corrupted], tgts, config.batch_size
        )
        corrupted_prompts += [it.prompt for it in corrupted]
        target_ids += tgts
        baselines.append(base)
        ks_flat += [k] * len(corrupted)
        log(f"  k={k:>2} mean p(target | corrupted) = {base.mean():.4f}")
    baseline = np.concatenate(baselines)

    effects = np.zeros((arch.n_layers, arch.n_heads, len(corrupted_prompts)))
    t0 = time.time()
    for li, layer in enumerate(layers):
        for si, (prompt, tgt, k) in enumerate(
            zip(corrupted_prompts, target_ids, ks_flat)
        ):
            effects[layer, :, si] = (
                patched_probs_one_layer(
                    model, tok, arch, prompt, tgt, abar[k][layer], layer
                )
                - baseline[si]
            )
        done = (li + 1) * len(corrupted_prompts)
        rate = (time.time() - t0) / done
        log(
            f"  layer {layer:>2}/{arch.n_layers - 1}  "
            f"{rate:.2f}s/forward  eta {rate * (len(layers) * len(corrupted_prompts) - done) / 60:.1f} min"
        )

    ks_arr = np.array(ks_flat)
    per_k = np.stack(
        [effects[:, :, ks_arr == k].mean(axis=2) for k in config.head_id_ks], axis=0
    )
    n = effects.shape[2]
    return (
        effects.mean(axis=2),
        effects.std(axis=2, ddof=1) / np.sqrt(n),
        n,
        per_k,
        float(baseline.mean()),
    )


@torch.inference_mode()
def confirm_cells(model, tok, arch, config, cells, log=print):
    """Re-measure the shortlisted (layer, head) cells with more prompts.

    Batches over prompts rather than heads: for a fixed cell there is only one
    head to patch, so the diagonal trick does not apply and a plain batch of
    distinct prompts is both correct and cheaper.
    """
    samples = []
    for k in config.head_id_ks:
        clean = P.make_prompt_set(
            config.family, k, config.head_id_confirm_n_prompts, config.shots,
            config.seed + 1, stratum="heldout",
        )
        corrupted = [P.corrupt_labels(it, config.seed + 1, i) for i, it in enumerate(clean)]
        abar = mean_head_activations(
            model, tok, arch, [it.prompt for it in clean], config.batch_size
        )
        prompts = [it.prompt for it in corrupted]
        tgts = [first_token_id(tok, it.target) for it in corrupted]
        # Baselines depend only on k, not on the cell being patched: computing
        # them once here rather than inside the cell loop saves 30x the work.
        base = target_probs(model, tok, prompts, tgts, config.batch_size)
        samples.append((k, prompts, tgts, base, abar))
    log(f"  confirm baselines: mean p(target | corrupted) = "
        f"{np.mean([s[3].mean() for s in samples]):.4f}")

    out = {}
    for ci, (layer, head) in enumerate(cells):
        eff = []
        for k, prompts, tgts, base, abar in samples:
            patch = abar[layer, head]
            for start in range(0, len(prompts), config.batch_size):
                chunk = prompts[start : start + config.batch_size]
                enc = encode(tok, chunk, model.device)
                handle = arch.attn_out[layer].register_forward_pre_hook(
                    _single_head_hook(arch, head, patch)
                )
                try:
                    logits = model(**enc, logits_to_keep=1).logits[:, -1].float()
                finally:
                    handle.remove()
                probs = torch.softmax(logits, dim=-1)
                for row, tgt in enumerate(tgts[start : start + len(chunk)]):
                    eff.append(probs[row, tgt].item() - base[start + row])
        eff = np.array(eff)
        out[(layer, head)] = (eff.mean(), eff.std(ddof=1) / np.sqrt(len(eff)), len(eff))
        log(f"  confirm {ci + 1}/{len(cells)}  L{layer} H{head}  "
            f"AIE {eff.mean():+.4f} ± {out[(layer, head)][1]:.4f}")
    return out


def _single_head_hook(arch: Arch, head: int, patch: torch.Tensor):
    def hook(_module, args):
        x = args[0]
        b, t, _ = x.shape
        x = x.view(b, t, arch.n_heads, arch.head_dim).clone()
        x[:, -1, head, :] = patch.to(x.dtype)
        return (x.view(b, t, -1),) + args[1:]

    return hook


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--calibrate", action="store_true",
                        help="time one layer and exit without sweeping")
    parser.add_argument("--layers", default=None, help="subset, e.g. 0,1,2 (testing)")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    out_dir = Path(config.results_dir) / config.run_name
    if not args.calibrate and (out_dir / "heads.npz").exists():
        raise SystemExit(
            f"refusing to overwrite {out_dir / 'heads.npz'}; rename the run dir or "
            "change run_name (CLAUDE.md: results are never overwritten)"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "heads.log"
    lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        lines.append(msg)
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    log(f"model {config.model}  device {config.device}  dtype {config.dtype}")
    model, tok = load_model(config)
    arch = describe(model)
    log(f"arch: {arch.n_layers} layers x {arch.n_heads} heads, head_dim "
        f"{arch.head_dim}, d_model {arch.d_model}")
    if config.device.startswith("cuda"):
        log(f"VRAM after load: {torch.cuda.memory_allocated() / 2**30:.2f} GiB "
            f"of {torch.cuda.get_device_properties(0).total_memory / 2**30:.2f} GiB")

    layers = (
        [int(x) for x in args.layers.split(",")] if args.layers
        else ([0] if args.calibrate else None)
    )
    t0 = time.time()
    aie, aie_se, n_samples, per_k, baseline = sweep(
        model, tok, arch, config, layers=layers, log=log
    )
    elapsed = time.time() - t0

    if args.calibrate:
        n_fwd = n_samples
        per_fwd = elapsed / n_fwd
        log(f"\ncalibration: {n_fwd} forwards in {elapsed:.1f}s = {per_fwd:.2f}s each")
        log(f"projected full sweep ({arch.n_layers} layers x {n_samples} samples): "
            f"{arch.n_layers * n_samples * per_fwd / 3600:.2f} h")
        if config.device.startswith("cuda"):
            log(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
        return

    flat = [
        (aie[l, h], aie_se[l, h], l, h)
        for l in range(arch.n_layers) for h in range(arch.n_heads)
    ]
    flat.sort(reverse=True)
    shortlist = [(l, h) for _, _, l, h in flat[: config.head_id_confirm_cells]]
    log(f"\nstage A done in {elapsed / 60:.1f} min; confirming {len(shortlist)} cells")
    confirmed = confirm_cells(model, tok, arch, config, shortlist, log=log)

    ranked = sorted(confirmed.items(), key=lambda kv: -kv[1][0])
    head_set = [list(cell) for cell, _ in ranked[: config.top_heads]]

    meta = {
        "config": dataclasses.asdict(config),
        "config_sha256": config_sha(config),
        "git_commit": git_commit(),
        "n_layers": arch.n_layers, "n_heads": arch.n_heads,
        "head_dim": arch.head_dim, "d_model": arch.d_model,
        "corrupted_baseline_mean": baseline,
        "stage_a_seconds": elapsed,
        "platform": platform.platform(),
        "versions": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "numpy": np.__version__,
        },
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated())
            if config.device.startswith("cuda") else 0
        ),
    }
    np.savez(
        out_dir / "heads.npz",
        aie=aie.astype(np.float32),
        aie_se=aie_se.astype(np.float32),
        aie_per_k=per_k.astype(np.float32),
        head_id_ks=np.array(config.head_id_ks, dtype=np.int32),
        n_samples=np.int32(n_samples),
        shortlist=np.array(shortlist, dtype=np.int32),
        confirm_aie=np.array([confirmed[c][0] for c in shortlist], dtype=np.float32),
        confirm_se=np.array([confirmed[c][1] for c in shortlist], dtype=np.float32),
        confirm_n=np.array([confirmed[c][2] for c in shortlist], dtype=np.int32),
        head_set=np.array(head_set, dtype=np.int32),
        meta_json=json.dumps(meta, sort_keys=True),
    )
    sha = hashlib.sha256((out_dir / "heads.npz").read_bytes()).hexdigest()
    (out_dir / "heads.sha256").write_text(sha + "\n", encoding="utf-8", newline="\n")
    log(f"\nwrote {out_dir / 'heads.npz'} (sha256 {sha[:12]})")
    log(f"head set: {head_set}")


if __name__ == "__main__":
    main()
