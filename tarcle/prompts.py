"""Prompt generation for shift-by-k task families.

Pure Python: no torch, no numpy. Everything is seeded and serializable so that
the same config produces byte-identical prompt sets on any machine (asserted by
SHA-256 in run manifests).
"""
from __future__ import annotations

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


def shift(domain: str, operand: str, k: int) -> str:
    """Apply shift-by-k within the domain's cycle, with wraparound."""
    items = DOMAINS[domain]
    return items[(items.index(operand) + k) % len(items)]


@dataclass(frozen=True)
class PromptItem:
    prompt: str
    target: str
    variant: str  # "days" | "mixed"
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
    rng: random.Random, domain: str, query_domain: str, query: str, exclude_query: bool
) -> str:
    pool = DOMAINS[domain]
    if exclude_query and domain == query_domain:
        pool = [x for x in pool if x != query]
    return rng.choice(pool)


def make_prompt_set(
    variant: str, k: int, n: int, shots: int, seed: int
) -> list[PromptItem]:
    """n items for one (variant, k): first half copy stratum, second half held-out.

    Copy stratum guarantees the query pair appears verbatim among the demos (one
    demo is force-placed); held-out stratum guarantees the query operand never
    appears as a demo operand. The gate metric uses the held-out stratum only.
    """
    if variant == "days":
        domains = ["days"]
    elif variant == "mixed":
        domains = MIXED_DOMAINS
    else:
        raise ValueError(f"unknown variant: {variant}")

    rng = random.Random(child_seed(seed, variant, k))
    items: list[PromptItem] = []
    for i in range(n):
        copy_stratum = i < n // 2
        query_domain = rng.choice(domains)
        query = rng.choice(DOMAINS[query_domain])

        demos: list[tuple[str, str, str]] = []
        for _ in range(shots):
            d = rng.choice(domains)
            operand = _sample_demo_operand(
                rng, d, query_domain, query, exclude_query=not copy_stratum
            )
            demos.append((d, operand, shift(d, operand, k)))
        if copy_stratum:
            slot = rng.randrange(shots)
            demos[slot] = (query_domain, query, shift(query_domain, query, k))

        items.append(
            PromptItem(
                prompt=render_prompt(demos, query),
                target=shift(query_domain, query, k),
                variant=variant,
                k=k,
                domain=query_domain,
                query=query,
                query_in_demos=copy_stratum,
                choices=list(DOMAINS[query_domain]),
                demos=[list(d) for d in demos],
            )
        )
    return items


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
