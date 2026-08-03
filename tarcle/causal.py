"""Stage 1: FV injection and causal-efficacy scoring. GPU code — imports torch.

The efficacy score is the pre-declared arbiter for whether a given FV(k) is a
real task vector (docs/decisions.md D2), so what it measures matters more than
usual:

    lift(k) = accuracy(zero-shot prompt + FV(k)) - accuracy(zero-shot prompt)

The query set is the *complete* operand cycle, not a sample of it — for months
there are exactly 12 distinct zero-shot prompts `Q: <month>\\nA:`, and all 12 are
scored. So accuracy is quantised in steps of 1/12 but carries no sampling error:
it is a census of the query space, not an estimate. Widening it would mean
varying the prompt format, which docs/decisions.md D7 defers to stage 3.

Todd-style FVs are *added* to the residual stream and Hendel-style states
*replace* it, following each paper's own protocol. The difference is recorded in
`injection_mode` rather than harmonised away.

Registered caveat (D2): the zero-shot baseline is dominated by a copy prior, so
the identity task k=0 is already at ceiling before any injection and its lift is
~0 by construction. FV(0) is reported and excluded from the arbiter comparison.
"""
from __future__ import annotations

import numpy as np
import torch

from .extract import Arch, encode, first_token_id
from .prompts import DOMAINS, shift


def zero_shot_prompts(domain: str) -> list[str]:
    """The complete query space: one prompt per operand, same surface form as
    the final line of an ICL prompt (prompts.render_prompt)."""
    return [f"Q: {x}\nA:" for x in DOMAINS[domain]]


class InjectResidual:
    """Add or replace the residual stream at the final position of one layer."""

    def __init__(self, arch: Arch, layer: int, vector: torch.Tensor, mode: str):
        if mode not in ("add", "replace"):
            raise ValueError(f"unknown injection mode: {mode}")
        self.arch, self.layer, self.vector, self.mode = arch, layer, vector, mode
        self._handle = None

    def __enter__(self):
        self._handle = self.arch.blocks[self.layer].register_forward_hook(self._hook)
        return self

    def _hook(self, _module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden.clone()
        v = self.vector.to(hidden.dtype)
        if self.mode == "add":
            hidden[:, -1, :] += v
        else:
            hidden[:, -1, :] = v
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

    def __exit__(self, *exc):
        self._handle.remove()
        return False


@torch.inference_mode()
def forced_choice(
    model, tok, prompts: list[str], choice_ids: list[int], batch_size: int,
    inject: tuple[Arch, int, torch.Tensor, str] | None = None,
) -> np.ndarray:
    """(len(prompts),) index into choice_ids of the argmax choice.

    Every candidate is a single token for the months and days domains (checked:
    all 12 month names are one token as ' <Month>'), so one forward per prompt
    settles the whole forced choice at the final position.
    """
    ids = torch.tensor(choice_ids, device=model.device)
    picks = []
    for start in range(0, len(prompts), batch_size):
        enc = encode(tok, prompts[start : start + batch_size], model.device)
        if inject is None:
            logits = model(**enc, logits_to_keep=1).logits[:, -1].float()
        else:
            with InjectResidual(*inject):
                logits = model(**enc, logits_to_keep=1).logits[:, -1].float()
        picks.append(logits[:, ids].argmax(dim=-1).cpu().numpy())
    return np.concatenate(picks)


def accuracy_for_k(
    model, tok, arch, domain: str, k: int, batch_size: int,
    vector: torch.Tensor | None = None, layer: int = 0, mode: str = "add",
) -> float:
    """Fraction of the complete operand cycle mapped to operand+k."""
    items = DOMAINS[domain]
    prompts = zero_shot_prompts(domain)
    choice_ids = [first_token_id(tok, x) for x in items]
    inject = None if vector is None else (arch, layer, vector, mode)
    picks = forced_choice(model, tok, prompts, choice_ids, batch_size, inject)
    correct = [items.index(shift(domain, x, k)) for x in items]
    return float(np.mean(picks == np.array(correct)))


def baseline_accuracy(model, tok, arch, domain: str, ks, batch_size: int) -> dict:
    """Zero-shot accuracy per k with no injection. Expected: ~1.0 at k=0 (the
    copy prior answers the identity task for free) and ~0 elsewhere."""
    return {k: accuracy_for_k(model, tok, arch, domain, k, batch_size) for k in ks}


def sweep_injection(
    model, tok, arch, domain: str, vectors_at, ks,
    batch_size: int, mode: str, scales, log=print,
) -> tuple[int, float, dict]:
    """Pick injection layer AND scale once, on the head-ID k subset, then freeze.

    `vectors_at(layer) -> {k: vector}` so the two methods can differ in what
    they inject: a Todd FV is one vector added at whichever layer works best,
    while a Hendel state is layer-specific and layer L's state replaces layer
    L's.

    Both hyperparameters are chosen on the in-sweep k only and frozen for every
    k and every condition. Tuning either per k would fit the protocol to each
    task and make cross-k efficacy incomparable, which is precisely what the
    docs/decisions.md D2 arbiter must avoid.

    Scale matters more than it looks: at scale 1.0 the Todd FV moved shift-by-3
    to 0.08, and at 2.0 to 1.00. See D12.
    """
    grid = {}
    for layer in range(arch.n_layers):
        vectors = vectors_at(layer)
        for scale in scales:
            accs = [
                accuracy_for_k(
                    model, tok, arch, domain, k, batch_size,
                    vectors[k] * scale, layer, mode,
                )
                for k in ks
            ]
            grid[(layer, scale)] = float(np.mean(accs))
    best_layer, best_scale = max(grid, key=grid.get)
    log(f"  {mode}: best injection L{best_layer} scale x{best_scale} "
        f"(mean acc {grid[(best_layer, best_scale)]:.3f} over k={list(ks)})")
    top = sorted(grid.items(), key=lambda kv: -kv[1])[:5]
    log("  top: " + ", ".join(f"L{l}x{s}={a:.3f}" for (l, s), a in top))
    return best_layer, best_scale, {f"L{l}_x{s}": a for (l, s), a in grid.items()}


def efficacy(
    model, tok, arch, domain: str, vectors: dict[int, torch.Tensor], ks,
    layer: int, mode: str, batch_size: int, baseline: dict[int, float],
    scale: float = 1.0,
) -> dict:
    """Injected accuracy and lift over the no-injection baseline, per k."""
    acc = {
        k: accuracy_for_k(
            model, tok, arch, domain, k, batch_size, vectors[k] * scale, layer, mode
        )
        for k in ks
    }
    return {
        "acc": acc,
        "lift": {k: acc[k] - baseline[k] for k in ks},
        "n": len(DOMAINS[domain]),
    }


NEUTRAL_CONTEXTS = [
    "The", "It was", "In", "The month of",
    "She said", "They arrived in", "A", "Last",
]


@torch.inference_mode()
def frequency_proxy(model, tok, domain: str, batch_size: int) -> dict[str, float]:
    """Mean next-token logprob of each operand token across neutral contexts.

    The prereg §3 Test-1 artefact control: month names differ in corpus
    frequency and several carry non-month senses (May, March, August), so a
    'constant norms' claim has to be checked against this before it can be
    attributed to task structure.
    """
    ids = [first_token_id(tok, x) for x in DOMAINS[domain]]
    totals = np.zeros(len(ids))
    for start in range(0, len(NEUTRAL_CONTEXTS), batch_size):
        chunk = NEUTRAL_CONTEXTS[start : start + batch_size]
        enc = encode(tok, chunk, model.device)
        logprobs = torch.log_softmax(
            model(**enc, logits_to_keep=1).logits[:, -1].float(), dim=-1
        )
        totals += logprobs[:, ids].sum(dim=0).cpu().numpy()
    return dict(zip(DOMAINS[domain], totals / len(NEUTRAL_CONTEXTS)))
