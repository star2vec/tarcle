"""Prompt generation for shift-by-k task families.

Pure Python: no torch, no numpy. Everything is seeded and serializable so that
the same config produces byte-identical prompt sets on any machine (asserted by
SHA-256 in run manifests).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

DOMAINS: dict[str, list[str]] = {
    "days": [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    ],
    "months": [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ],
    "letters": list("abcdefghijklmnopqrstuvwxyz"),
    "digits": [str(d) for d in range(10)],
}

MIXED_DOMAINS = ["days", "months", "letters", "digits"]

# Multi-domain variants. The control they serve (BRIEF §6 Control 2) needs only
# that the demonstration operands not share a single circle, which two domains of
# different cycle length already satisfy — so the D5 ladder can fall back to
# "days_months" when the four-domain set fails its behavioural gate.
DOMAIN_GROUPS: dict[str, list[str]] = {
    "mixed": MIXED_DOMAINS,
    "days_months": ["days", "months"],
}


def shift(domain: str, operand: str, k: int) -> str:
    """Apply shift-by-k within the domain's cycle, with wraparound."""
    items = DOMAINS[domain]
    return items[(items.index(operand) + k) % len(items)]


@dataclass(frozen=True)
class PromptItem:
    prompt: str
    target: str
    variant: str  # a single domain ("days", "months", ...) or "mixed"
    k: int
    domain: str  # domain of the query operand
    query: str
    query_in_demos: bool  # True: copy stratum; False: held-out stratum
    choices: list[str]  # forced-choice candidates = full cycle of the query domain
    demos: list[list[str]]  # [domain, operand, target] per demonstration


def render_prompt(demos: list[tuple[str, str, str]], query: str) -> str:
    lines = [f"Q: {operand}\nA: {target}" for _, operand, target in demos]
    lines.append(f"Q: {query}\nA:")
    return "\n\n".join(lines)


def child_seed(seed: int, variant: str, k: int) -> int:
    digest = hashlib.sha256(f"{seed}|{variant}|{k}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _sample_demo_operand(
    rng: random.Random,
    domain: str,
    query_domain: str,
    query: str,
    exclude_query: bool,
    operand_pool: dict[str, list[str]] | None = None,
) -> str:
    pool = DOMAINS[domain] if operand_pool is None else operand_pool[domain]
    if exclude_query and domain == query_domain:
        pool = [x for x in pool if x != query]
    if not pool:
        raise ValueError(
            f"empty demo-operand pool for domain {domain!r} after excluding the "
            f"query {query!r}; widen operand_pool"
        )
    return rng.choice(pool)


def make_prompt_set(
    variant: str,
    k: int,
    n: int,
    shots: int,
    seed: int,
    *,
    stratum: str = "both",
    operand_pool: dict[str, list[str]] | None = None,
    query_pool: dict[str, list[str]] | None = None,
    query_domain: str | None = None,
) -> list[PromptItem]:
    """n items for one (variant, k): first half copy stratum, second half held-out.

    Copy stratum guarantees the query pair appears verbatim among the demos (one
    demo is force-placed); held-out stratum guarantees the query operand never
    appears as a demo operand. The gate metric uses the held-out stratum only.

    Keyword-only extras, all defaulting to the legacy behaviour so prompt-set
    SHA-256s recorded by earlier runs still reproduce (asserted by
    tests/test_prompts_determinism.py):

    - stratum="heldout": every item is held-out. FV extraction uses this — the
      copy stratum lets the model copy the answer out of the demos instead of
      inferring the task, which is not the thing we want a task vector of.
    - operand_pool / query_pool: {domain: [allowed values]} restrictions on demo
      operands and on query operands respectively. The operand-partition control
      (BRIEF §6) and the prereg §3 polysemy leave-out are both this parameter.
      Targets are unrestricted: shift(operand, k) may leave the pool, which is
      what makes the transfer test meaningful.
    - query_domain: force the query's domain. Used by the mixed-domain control,
      where demos span four domains but the query is always a month so that the
      family keeps its Z/12 semantics (docs/decisions.md D5).
    """
    if variant in DOMAIN_GROUPS:
        domains = DOMAIN_GROUPS[variant]
    elif variant in DOMAINS:
        domains = [variant]  # single-domain variant: "days", "months", ...
    else:
        raise ValueError(f"unknown variant: {variant}")
    if stratum not in ("both", "heldout"):
        raise ValueError(f"unknown stratum: {stratum}")
    if query_domain is not None and query_domain not in domains:
        raise ValueError(f"query_domain {query_domain!r} not in variant {variant!r}")

    rng = random.Random(child_seed(seed, variant, k))
    items: list[PromptItem] = []
    for i in range(n):
        copy_stratum = stratum == "both" and i < n // 2
        qd = rng.choice(domains) if query_domain is None else query_domain
        qpool = DOMAINS[qd] if query_pool is None else query_pool[qd]
        query = rng.choice(qpool)

        demos: list[tuple[str, str, str]] = []
        for _ in range(shots):
            d = rng.choice(domains)
            operand = _sample_demo_operand(
                rng, d, qd, query, not copy_stratum, operand_pool
            )
            demos.append((d, operand, shift(d, operand, k)))
        if copy_stratum:
            slot = rng.randrange(shots)
            demos[slot] = (qd, query, shift(qd, query, k))

        items.append(
            PromptItem(
                prompt=render_prompt(demos, query),
                target=shift(qd, query, k),
                variant=variant,
                k=k,
                domain=qd,
                query=query,
                query_in_demos=copy_stratum,
                choices=list(DOMAINS[qd]),
                demos=[list(d) for d in demos],
            )
        )
    return items


def corrupt_labels(item: PromptItem, seed: int, index: int) -> PromptItem:
    """Todd-style shuffled-label version of a prompt: the demo targets are
    permuted among the demos, so no consistent shift-by-k is inferable, while
    the surface form, operand distribution and token count are unchanged.

    This is the baseline the causal head sweep patches into: AIE compares
    p(target | corrupted + patched head) against p(target | corrupted). The
    query, its target and the choice set are untouched — only the mapping the
    demos exhibit is destroyed.

    The permutation is a derangement whenever the targets are not all identical
    (k=0 prompts have operand == target, so shuffling is a no-op there; k=0 is
    excluded from the sweep for a separate reason, docs/decisions.md D3).
    """
    rng = random.Random(child_seed(seed, f"corrupt|{item.variant}|{index}", item.k))
    targets = [d[2] for d in item.demos]
    shuffled = targets[:]
    for _ in range(100):
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(targets, shuffled)) or len(set(targets)) == 1:
            break
    demos = [(d[0], d[1], t) for d, t in zip(item.demos, shuffled)]
    return dataclasses.replace(
        item,
        prompt=render_prompt(demos, item.query),
        demos=[list(d) for d in demos],
    )


def serialize_items(items: list[PromptItem]) -> str:
    lines = [
        json.dumps(asdict(item), sort_keys=True, ensure_ascii=False) for item in items
    ]
    return "\n".join(lines) + "\n"


def write_prompt_set(items: list[PromptItem], path: Path) -> str:
    """Write JSONL with fixed newlines (hash must match across OSes); return SHA-256."""
    text = serialize_items(items)
    path.write_text(text, encoding="utf-8", newline="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_prompt_set(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
