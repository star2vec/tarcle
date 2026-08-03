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


def zero_shot_prompts(domain: str, queries: list[str] | None = None) -> list[str]:
    """The complete query space: one prompt per operand, same surface form as
    the final line of an ICL prompt (prompts.render_prompt).

    `queries` restricts to a subset of operands — the hypothesis-C transfer test
    evaluates a partition's FV on the *other* partition's operands (D18).
    """
    return [f"Q: {x}\nA:" for x in (queries or DOMAINS[domain])]


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
def choice_logprobs(
    model, tok, prompts: list[str], choice_ids: list[int], batch_size: int,
    inject: tuple[Arch, int, torch.Tensor, str] | None = None,
) -> np.ndarray:
    """(len(prompts), len(choice_ids)) log-softmax over the candidate set.

    Every candidate is a single token for the months and days domains (checked:
    all 12 month names are one token as ' <Month>'), so one forward per prompt
    settles the whole forced choice at the final position.
    """
    ids = torch.tensor(choice_ids, device=model.device)
    rows = []
    for start in range(0, len(prompts), batch_size):
        enc = encode(tok, prompts[start : start + batch_size], model.device)
        if inject is None:
            logits = model(**enc, logits_to_keep=1).logits[:, -1].float()
        else:
            with InjectResidual(*inject):
                logits = model(**enc, logits_to_keep=1).logits[:, -1].float()
        rows.append(torch.log_softmax(logits[:, ids], dim=-1).cpu().numpy())
    return np.concatenate(rows)


def forced_choice(
    model, tok, prompts: list[str], choice_ids: list[int], batch_size: int,
    inject: tuple[Arch, int, torch.Tensor, str] | None = None,
) -> np.ndarray:
    """(len(prompts),) index into choice_ids of the argmax choice."""
    return choice_logprobs(
        model, tok, prompts, choice_ids, batch_size, inject
    ).argmax(axis=-1)


def zero_shot_spec(family: str, cycle: str, k: int, queries: list[str] | None = None):
    """(prompts, choice strings, correct index per prompt, query index per prompt).

    Generalises the zero-shot evaluation over the three families. The query index
    is the position of the operand within the candidate set, and exists only for
    the cyclic families — it is what makes a *signed shift* definable. For the
    unrelated-task family there is no cycle and no shift, so it is None and the
    D18/D20 shift diagnostics do not apply.
    """
    if family == "unrelated":
        from .tasks_unrelated import TASK_NAMES, UNRELATED_TASKS, task_choices

        name = TASK_NAMES[k]
        pairs = UNRELATED_TASKS[name]
        if queries is not None:
            pairs = [p for p in pairs if p[0] in set(queries)]
        choices = task_choices(name)
        idx = {c: i for i, c in enumerate(choices)}
        return (
            [f"Q: {a}\nA:" for a, _ in pairs],
            choices,
            np.array([idx[b] for _, b in pairs]),
            None,
        )

    items = DOMAINS[cycle]  # candidate set is always the full cycle
    asked = queries or items
    idx = {x: i for i, x in enumerate(items)}
    return (
        zero_shot_prompts(cycle, asked),
        items,
        np.array([idx[shift(cycle, x, k)] for x in asked]),
        np.array([idx[x] for x in asked]),
    )


def score_for_k(
    model, tok, arch, domain: str, k: int, batch_size: int,
    vector: torch.Tensor | None = None, layer: int = 0, mode: str = "add",
    queries: list[str] | None = None, family: str = "",
) -> dict:
    """Three efficacy measures over the complete operand cycle.

    The query space for months is exhaustively 12 prompts — one per operand —
    so `acc` is a census, not a sample, and cannot be enlarged without varying
    the prompt format, which docs/decisions.md D7 defers to stage 3. Its
    resolution is therefore capped at 1/12, which is coarse for the D2 arbiter.

    `logp` and `margin` recover resolution from the same 12 queries by scoring
    the distribution rather than only its argmax:

    - `logp`   mean log P(correct | the 12 candidates) — continuous, chance is
               log(1/12) = -2.485
    - `margin` mean (logp[correct] - max logp[incorrect]) — positive iff the
               argmax is correct, and its magnitude says by how much
    """
    prompts, choices, correct, query_idx = zero_shot_spec(
        family or domain, domain, k, queries
    )
    choice_ids = [first_token_id(tok, x) for x in choices]
    inject = None if vector is None else (arch, layer, vector, mode)
    lp = choice_logprobs(model, tok, prompts, choice_ids, batch_size, inject)
    rows = np.arange(len(prompts))
    lp_correct = lp[rows, correct]
    masked = lp.copy()
    masked[rows, correct] = -np.inf
    return {
        "acc": float(np.mean(lp.argmax(axis=-1) == correct)),
        "logp": float(lp_correct.mean()),
        "margin": float((lp_correct - masked.max(axis=-1)).mean()),
        # Per-query values, so a standard error can be attached to the mean.
        # D17 forbids claiming a logp verdict for a cell whose SE is unavailable.
        "logp_per_query": lp_correct.astype(np.float64),
        "correct_per_query": (lp.argmax(axis=-1) == correct),
        # Signed prediction shift (argmax - query) mod n: the D18 wrong-region
        # signature, which needs the distribution and not just its accuracy.
        # Signed shift is defined only where the parameter is a shift on a cycle.
        # The unrelated-task family has no cycle, so this is empty rather than a
        # meaningless difference of two arbitrary vocabulary indices.
        "pred_shift": (
            ((lp.argmax(axis=-1) - query_idx) % len(choices)).astype(np.int32)
            if query_idx is not None
            else np.zeros(0, dtype=np.int32)
        ),
    }


def accuracy_for_k(
    model, tok, arch, domain: str, k: int, batch_size: int,
    vector: torch.Tensor | None = None, layer: int = 0, mode: str = "add",
) -> float:
    """Fraction of the complete operand cycle mapped to operand+k."""
    return score_for_k(
        model, tok, arch, domain, k, batch_size, vector, layer, mode
    )["acc"]


def baseline_accuracy(
    model, tok, arch, domain: str, ks, batch_size: int, family: str = ""
) -> dict:
    """Zero-shot scores per k with no injection. For the shift families: ~1.0 at
    k=0 (the copy prior answers the identity task for free) and ~0 elsewhere."""
    return {
        k: score_for_k(model, tok, arch, domain, k, batch_size, family=family)
        for k in ks
    }


def sweep_injection(
    model, tok, arch, domain: str, vectors_at, ks,
    batch_size: int, mode: str, scales, log=print, family: str = "",
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
                score_for_k(
                    model, tok, arch, domain, k, batch_size,
                    vectors[k] * scale, layer, mode, family=family,
                )["acc"]
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
    scale: float = 1.0, family: str = "",
) -> dict:
    """Injected accuracy and lift over the no-injection baseline, per k."""
    s = {
        k: score_for_k(
            model, tok, arch, domain, k, batch_size, vectors[k] * scale, layer, mode,
            family=family,
        )
        for k in ks
    }
    n = len(s[ks[0]]["logp_per_query"])  # may be a query subset, not the cycle
    return {
        "acc": {k: s[k]["acc"] for k in ks},
        "logp": {k: s[k]["logp"] for k in ks},
        "margin": {k: s[k]["margin"] for k in ks},
        "lift": {k: s[k]["acc"] - baseline[k]["acc"] for k in ks},
        "logp_lift": {k: s[k]["logp"] - baseline[k]["logp"] for k in ks},
        "logp_per_query": {k: s[k]["logp_per_query"] for k in ks},
        "logp_se": {
            k: float(s[k]["logp_per_query"].std(ddof=1) / np.sqrt(n)) for k in ks
        },
        "pred_shift": {k: s[k]["pred_shift"] for k in ks},
        "n": n,
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
