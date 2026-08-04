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

    # Injection strength grid, swept jointly with the layer on the head-ID k
    # subset and then frozen for every k (docs/decisions.md D12).
    injection_scales: list[float] = field(
        default_factory=lambda: [0.5, 1.0, 2.0, 3.0, 4.0]
    )

    # Head-set variation (D15/D16). `head_set_run` loads heads.npz from another
    # run so the head set can be swapped while prompts, seed and protocol stay
    # fixed; `head_set_exclude` drops cells from it, which is how the
    # intersection-8 condition is built. `injection_frozen` pins (layer, scale)
    # instead of re-sweeping, so the head set is the only variable.
    head_set_run: str = ""
    head_set_exclude: list[list[int]] = field(default_factory=list)
    injection_frozen: dict = field(default_factory=dict)
    condition: str = "primary"

    # Runs whose stage-A shortlists are unioned and persisted as per-head
    # residual-stream contributions in every FV .npz. Any head set that is a
    # subset of the union — canonical, all-k, intersection-8, or one not yet
    # invented — then becomes a stage-2 axis with no GPU and no re-extraction.
    # ~6MB per condition. This permanently removes the failure mode where an
    # alternative head set cannot be analysed from existing artifacts (D15).
    persist_head_runs: list[str] = field(default_factory=list)

    # Operand restrictions for the partition / polysemy controls: {domain: [...]}.
    # Empty means the full cycle. Targets are never restricted — shift(operand, k)
    # may leave the pool, which is what makes the transfer test meaningful.
    operand_pool: dict = field(default_factory=dict)
    query_pool: dict = field(default_factory=dict)
    # For multi-domain variants (prompts.DOMAIN_GROUPS) the demonstrations span
    # several cycles while the query is pinned to one. `family` then names the
    # prompt variant and `query_domain` names the cycle that injection, efficacy
    # and the frequency proxy are all defined over.
    query_domain: str = ""

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


def first_token_id(tok, word: str, prefix: str = " ") -> int:
    """Targets are scored as continuations of '...A:', hence the leading space.
    AIE uses the first target token (Todd et al. score the answer token).

    `prefix` exists because the leading space is not always part of the first
    content token. Llama-3 tokenizes ' 15' as [' ', '15'] — a bare space then the
    number — so for numeric candidates every choice would share the first token
    ' ' and the forced choice would be degenerate. Those families put the space
    in the prompt instead and pass prefix='' so the number token is scored.
    """
    return tok.encode(prefix + word, add_special_tokens=False)[0]


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


@torch.inference_mode()
def per_prompt_head_acts(model, tok, arch, prompts, cells, batch_size):
    """(n_prompts, len(cells), head_dim) head activations at the last position.

    Kept per prompt rather than pre-averaged so the FV, its standard error and
    the two split halves all come out of one pass. 100 prompts x 10 heads x 128
    is 0.5MB.
    """
    out = []
    for start in range(0, len(prompts), batch_size):
        enc = encode(tok, prompts[start : start + batch_size], model.device)
        with LastPositionHeads(arch) as cap:
            model(**enc, logits_to_keep=1)
        acts = cap.stacked().float()  # (B, n_layers, n_heads, head_dim)
        out.append(torch.stack([acts[:, l, h] for l, h in cells], dim=1).cpu())
    return torch.cat(out)


def project_heads(arch: Arch, cells, acts: torch.Tensor) -> torch.Tensor:
    """Head activations -> residual-stream contributions: (..., len(cells), d).

    Head (l,h)'s slice of the o_proj input, pushed through the matching columns
    of W_O, is that head's additive contribution to the residual stream. A
    Todd-style FV is the sum of these over the head set (verified against the
    attention block's actual output in tests/test_extract.py).

    no_grad, not inference_mode: w_out requires grad, and multiplying it by an
    inference tensor raises.
    """
    with torch.no_grad():
        parts = []
        for i, (layer, head) in enumerate(cells):
            w = arch.w_out[layer][
                :, head * arch.head_dim : (head + 1) * arch.head_dim
            ].float().cpu()
            parts.append(acts[..., i, :] @ w.T)
        return torch.stack(parts, dim=-2)


@torch.inference_mode()
def per_prompt_hidden(model, tok, arch, prompts, batch_size):
    """(n_prompts, n_layers, d) residual stream at the last position.

    The Hendel-style capture: whatever the prompt has compressed into the
    sentinel position, taken at every layer so the layer choice can be redone at
    stage 2 without a GPU.
    """
    out = []
    for start in range(0, len(prompts), batch_size):
        enc = encode(tok, prompts[start : start + batch_size], model.device)
        caught = {}
        handles = [
            arch.blocks[i].register_forward_hook(
                lambda _m, _a, o, i=i: caught.__setitem__(
                    i, (o[0] if isinstance(o, tuple) else o)[:, -1, :].detach()
                )
            )
            for i in range(arch.n_layers)
        ]
        try:
            model(**enc, logits_to_keep=1)
        finally:
            for h in handles:
                h.remove()
        out.append(
            torch.stack([caught[i] for i in range(arch.n_layers)], dim=1).float().cpu()
        )
    return torch.cat(out)


def dummy_query_prompts(items, dummy: str = "x") -> list[str]:
    """Hendel-style: same demonstrations, query replaced by a placeholder that
    belongs to no domain, so the captured state cannot encode a real operand."""
    return [
        P.render_prompt([tuple(d) for d in it.demos], dummy) for it in items
    ]


def summarize(per_prompt: torch.Tensor) -> dict:
    """Mean, standard error and the two split halves of a per-prompt stack."""
    x = per_prompt.numpy().astype(np.float64)
    n = len(x)
    half = n // 2
    return {
        "mean": x.mean(axis=0),
        "se": x.std(axis=0, ddof=1) / np.sqrt(n),
        "half_a": x[:half].mean(axis=0),
        "half_b": x[half : 2 * half].mean(axis=0),
        "n": n,
    }


def _single_head_hook(arch: Arch, head: int, patch: torch.Tensor):
    def hook(_module, args):
        x = args[0]
        b, t, _ = x.shape
        x = x.view(b, t, arch.n_heads, arch.head_dim).clone()
        x[:, -1, head, :] = patch.to(x.dtype)
        return (x.view(b, t, -1),) + args[1:]

    return hook


def empirical_proxy(config: ExtractConfig, proxy: dict[str, float], **kwargs):
    """Mean token-frequency proxy over the operands and targets *actually drawn*
    for each k. Returns two (n_k,) arrays.

    The pre-registration (§3 Test 1) motivates this control by saying the target
    distribution 'is a shifted copy of the operand distribution and its mean
    proxy varies with k even under uniform operand sampling'. That last clause
    is false: over the full cycle a shift is a bijection, so the mean proxy of
    the targets is *identical* for every k and the test would be vacuous. It has
    content for two reasons the prereg's phrasing misses, and both need the
    empirical draw rather than the idealised cycle:

    - finite prompt sets do not sample operands exactly uniformly, so the
      realised means do vary with k;
    - under a restricted operand pool (the partition and polysemy-leave-out
      conditions) shifting genuinely moves mass onto different target tokens.

    See docs/decisions.md D11.
    """
    operand, target = [], []
    for k in config.ks:
        items = P.make_prompt_set(
            config.family, k, config.n_prompts, config.shots, config.seed,
            stratum="heldout", **kwargs,
        )
        ops = [d[1] for it in items for d in it.demos]
        tgts = [d[2] for it in items for d in it.demos]
        operand.append(np.mean([proxy[x] for x in ops if x in proxy]))
        target.append(np.mean([proxy[x] for x in tgts if x in proxy]))
    return np.array(operand), np.array(target)


def run_fv(model, tok, arch, config: ExtractConfig, out_dir: Path, log=print) -> None:
    """Extract both FV families for every k, score them, and write the .npz pair.

    The head set is loaded from heads.npz and its SHA-256 is stamped into every
    output, so 'the same head set was used for all k' is checkable from the
    artifacts rather than asserted.
    """
    from . import causal

    heads_path = (
        Path(config.results_dir) / config.head_set_run / "heads.npz"
        if config.head_set_run else out_dir / "heads.npz"
    )
    head_sha = hashlib.sha256(heads_path.read_bytes()).hexdigest()
    hz = np.load(heads_path)
    cells = [tuple(int(x) for x in c) for c in hz["head_set"]]
    excluded = [tuple(c) for c in config.head_set_exclude]
    if excluded:
        cells = [c for c in cells if c not in excluded]
        log(f"excluded {excluded} -> {len(cells)} heads")
    log(f"head set from {heads_path} (sha256 {head_sha[:12]}): {cells}")

    union = sorted({
        tuple(int(x) for x in c)
        for run in (config.persist_head_runs or [])
        for c in np.load(Path(config.results_dir) / run / "heads.npz")["shortlist"]
    } | set(cells))
    if union:
        log(f"persisting per-head contributions for {len(union)} union cells")

    # The domain the FVs act on. The unrelated-task family has no operand cycle:
    # each task carries its own vocabulary, so causal scoring dispatches on the
    # family rather than on a domain and `cycle` is only a label there.
    cycle = config.query_domain or config.family
    cyclic = cycle in P.DOMAINS
    pools = {}
    if config.operand_pool:
        pools["operand_pool"] = config.operand_pool
    if config.query_pool:
        pools["query_pool"] = config.query_pool
    if config.query_domain:
        pools["query_domain"] = config.query_domain

    todd, hendel, prompt_shas = {}, {}, {}
    for k in config.ks:
        items = P.build_prompt_set(
            config.family, k, config.n_prompts, config.shots, config.seed,
            stratum="heldout", **pools,
        )
        prompt_shas[k] = P.write_prompt_set(items, out_dir / f"prompts_k{k}.jsonl")
        prompts = [it.prompt for it in items]

        # Capture the union once and index the active head set out of it, so the
        # extra cells cost no additional forward passes.
        capture = union or cells
        acts = per_prompt_head_acts(
            model, tok, arch, prompts, capture, config.batch_size
        )
        contrib_all = project_heads(arch, capture, acts)  # (n_prompts, U, d)
        take = [capture.index(c) for c in cells]
        contrib = contrib_all[:, take, :]
        todd[k] = {
            "fv": summarize(contrib.sum(dim=-2)),
            "contrib": contrib.numpy().astype(np.float64).mean(axis=0),
            "contrib_union": contrib_all.numpy().astype(np.float64).mean(axis=0),
        }

        hidden = per_prompt_hidden(
            model, tok, arch, dummy_query_prompts(items), config.batch_size
        )
        hendel[k] = {
            "fv": summarize(hidden),  # (n_layers, d) per statistic
            "real_query": per_prompt_hidden(
                model, tok, arch, prompts, config.batch_size
            ).numpy().astype(np.float64).mean(axis=0),
        }
        log(f"  k={k:>2}  todd |FV| {np.linalg.norm(todd[k]['fv']['mean']):8.3f}   "
            f"hendel |FV| (all layers) "
            f"{np.linalg.norm(hendel[k]['fv']['mean'], axis=-1).mean():8.3f}")

    dev = model.device
    todd_vecs = {
        k: torch.tensor(todd[k]["fv"]["mean"], device=dev, dtype=torch.float32)
        for k in config.ks
    }
    log("\ninjection-layer sweep (on the head-ID k subset, then frozen):")
    baseline = causal.baseline_accuracy(
        model, tok, arch, cycle, config.ks, config.batch_size, config.family
    )
    log(f"  zero-shot baseline acc per k: "
        f"{ {k: round(v['acc'], 3) for k, v in baseline.items()} }")

    frozen = config.injection_frozen
    if frozen:
        log(f"  injection protocol PINNED (not re-swept): {frozen}")
    if "todd" in frozen:
        todd_layer, todd_scale, todd_layers = (
            int(frozen["todd"]["layer"]), float(frozen["todd"]["scale"]), {},
        )
    else:
        todd_layer, todd_scale, todd_layers = causal.sweep_injection(
            model, tok, arch, cycle, lambda _l: todd_vecs,
            config.head_id_ks, config.batch_size, "add", config.injection_scales, log,
            config.family,
        )
    # Hendel replaces the hidden state rather than adding to it, so a scale
    # other than 1.0 would substitute a state of deliberately wrong magnitude —
    # not a stronger push but a different, invalid state. Scale is not a free
    # hyperparameter for this method (D12).
    if "hendel" in frozen:
        hendel_layer, hendel_scale, hendel_layers = (
            int(frozen["hendel"]["layer"]), float(frozen["hendel"]["scale"]), {},
        )
    else:
        hendel_layer, hendel_scale, hendel_layers = causal.sweep_injection(
            model, tok, arch, cycle,
            lambda l: {
                k: torch.tensor(
                    hendel[k]["fv"]["mean"][l], device=dev, dtype=torch.float32
                )
                for k in config.ks
            },
            config.head_id_ks, config.batch_size, "replace", [1.0], log, config.family,
        )

    todd_eff = causal.efficacy(
        model, tok, arch, cycle, todd_vecs, config.ks, todd_layer,
        "add", config.batch_size, baseline, todd_scale, config.family,
    )
    hendel_vecs = {
        k: torch.tensor(
            hendel[k]["fv"]["mean"][hendel_layer], device=dev, dtype=torch.float32
        )
        for k in config.ks
    }
    hendel_eff = causal.efficacy(
        model, tok, arch, cycle, hendel_vecs, config.ks, hendel_layer,
        "replace", config.batch_size, baseline, hendel_scale, config.family,
    )

    # The prereg §3 frequency control is about month-name token frequency and
    # polysemy. It is defined only for a family with a single operand cycle; the
    # unrelated tasks have twelve separate vocabularies and no such confound.
    if cyclic:
        proxy = causal.frequency_proxy(model, tok, cycle, config.batch_size)
        proxy_operand, proxy_target = empirical_proxy(config, proxy, **pools)
    else:
        proxy = {}
        proxy_operand = proxy_target = np.zeros(len(config.ks))

    common = {
        "ks": np.array(config.ks, dtype=np.int32),
        "freq_proxy_operand": proxy_operand.astype(np.float32),
        "freq_proxy_target": proxy_target.astype(np.float32),
        "efficacy_baseline": np.array(
            [baseline[k]["acc"] for k in config.ks], dtype=np.float32
        ),
        "efficacy_baseline_logp": np.array(
            [baseline[k]["logp"] for k in config.ks], dtype=np.float32
        ),
    }
    meta_common = {
        "config": dataclasses.asdict(config),
        "config_sha256": config_sha(config),
        "git_commit": git_commit(),
        "model": config.model, "dtype": config.dtype, "device": config.device,
        "attn_implementation": "sdpa",
        "family": config.family, "cycle_domain": cycle, "n": len(config.ks),
        "condition": config.condition,
        "head_set_cells": [list(c) for c in cells],
        "head_set_excluded": [list(c) for c in excluded],
        "operand_pool": config.operand_pool or (
            {cycle: P.DOMAINS[cycle]} if cyclic else {cycle: []}
        ),
        "query_pool": config.query_pool or (
            {cycle: P.DOMAINS[cycle]} if cyclic else {cycle: []}
        ),
        "shots": config.shots, "stratum": "heldout",
        "n_prompts_per_k": config.n_prompts, "seed": config.seed,
        "prompt_sha256": prompt_shas,
        "head_set_source": {"path": str(heads_path), "sha256": head_sha},
        "frequency_proxy": proxy,
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

    for method, eff, layer, mode, layers_acc, scale in (
        ("todd", todd_eff, todd_layer, "add", todd_layers, todd_scale),
        ("hendel", hendel_eff, hendel_layer, "replace", hendel_layers, hendel_scale),
    ):
        src = todd if method == "todd" else hendel
        if method == "todd":
            X = np.stack([src[k]["fv"]["mean"] for k in config.ks])
            extra = {
                "head_set": np.array(cells, dtype=np.int32),
                "head_aie": hz["aie"],
                "head_contrib": np.stack(
                    [src[k]["contrib"] for k in config.ks]
                ).astype(np.float32),
                "head_contrib_union": np.stack(
                    [src[k]["contrib_union"] for k in config.ks]
                ).astype(np.float32),
                "head_union_cells": np.array(union or cells, dtype=np.int32),
            }
        else:
            X = np.stack([src[k]["fv"]["mean"][layer] for k in config.ks])
            extra = {
                "X_all_layers": np.stack(
                    [src[k]["fv"]["mean"] for k in config.ks]
                ).astype(np.float32),
                "X_real_query": np.stack(
                    [src[k]["real_query"][layer] for k in config.ks]
                ).astype(np.float32),
            }
        halves = ("half_a", "half_b")
        sel = (lambda a: a) if method == "todd" else (lambda a: a[layer])
        meta = {
            **meta_common, "method": method, "injection_layer": int(layer),
            "injection_mode": mode, "injection_scale": float(scale),
            "injection_sweep": layers_acc,
        }
        path = out_dir / f"fv_{config.condition}_{method}.npz"
        np.savez(
            path,
            X=X.astype(np.float32),
            norms=np.linalg.norm(X, axis=1).astype(np.float32),
            X_half_a=np.stack(
                [sel(src[k]["fv"][halves[0]]) for k in config.ks]
            ).astype(np.float32),
            X_half_b=np.stack(
                [sel(src[k]["fv"][halves[1]]) for k in config.ks]
            ).astype(np.float32),
            X_se=np.stack([sel(src[k]["fv"]["se"]) for k in config.ks]).astype(np.float32),
            n_prompts=np.array(
                [src[k]["fv"]["n"] for k in config.ks], dtype=np.int32
            ),
            efficacy_acc=np.array([eff["acc"][k] for k in config.ks], dtype=np.float32),
            efficacy_lift=np.array(
                [eff["lift"][k] for k in config.ks], dtype=np.float32
            ),
            efficacy_logp=np.array(
                [eff["logp"][k] for k in config.ks], dtype=np.float32
            ),
            efficacy_logp_lift=np.array(
                [eff["logp_lift"][k] for k in config.ks], dtype=np.float32
            ),
            efficacy_margin=np.array(
                [eff["margin"][k] for k in config.ks], dtype=np.float32
            ),
            efficacy_logp_per_query=np.stack(
                [eff["logp_per_query"][k] for k in config.ks]
            ).astype(np.float32),
            efficacy_logp_se=np.array(
                [eff["logp_se"][k] for k in config.ks], dtype=np.float32
            ),
            efficacy_pred_shift=np.stack(
                [eff["pred_shift"][k] for k in config.ks]
            ).astype(np.int32),
            efficacy_n=np.array([eff["n"]] * len(config.ks), dtype=np.int32),
            **common, **extra,
            meta_json=json.dumps(meta, sort_keys=True),
        )
        log(f"wrote {path} "
            f"(sha256 {hashlib.sha256(path.read_bytes()).hexdigest()[:12]})")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--stage", choices=("heads", "fv"), default="heads")
    parser.add_argument("--calibrate", action="store_true",
                        help="time one layer and exit without sweeping")
    parser.add_argument("--layers", default=None, help="subset, e.g. 0,1,2 (testing)")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    out_dir = Path(config.results_dir) / config.run_name
    guard = out_dir / (
        "heads.npz" if args.stage == "heads" else f"fv_{config.condition}_todd.npz"
    )
    if not args.calibrate and guard.exists():
        raise SystemExit(
            f"refusing to overwrite {guard}; rename the run dir or change "
            "run_name (CLAUDE.md: results are never overwritten)"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{args.stage}.log"
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

    if args.stage == "fv":
        run_fv(model, tok, arch, config, out_dir, log=log)
        return

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
