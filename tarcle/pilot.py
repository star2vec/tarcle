"""Behavioral pilot runner: shift-by-k ICL accuracy (the Day-1 go/no-go gate).

Usage:
    python -m tarcle.pilot experiments/pilot_gpt2_cpu.json [--n N] [--shots S]
        [--ks 0,1,2] [--run-name NAME]

Writes results/pilot/<run_name>/{prompts.jsonl, scores.jsonl, manifest.json}.
Run directories are never overwritten; pick a new --run-name to rerun.
Scoring/plots live in tarcle.pilot_report and need only the saved files.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import prompts as P


@dataclass
class PilotConfig:
    run_name: str
    model: str
    backend: str = "local_hf"  # "local_hf" | "openai_compat"
    device: str = "cpu"
    dtype: str = "float32"
    variants: list[str] = field(default_factory=lambda: ["days", "mixed"])
    ks: list[int] = field(default_factory=lambda: list(range(7)))
    n_per_k: int = 100
    shots: int = 10
    seed: int = 0
    batch_size: int = 8
    results_dir: str = "results/pilot"
    base_url: str = ""  # openai_compat only
    # Per-condition gating (docs/decisions.md D20 §3): a restricted operand pool
    # is a different effective task and needs its own behavioural gate, so the
    # pilot has to be able to reproduce the extraction condition's prompts.
    stratum: str = "both"
    operand_pool: dict = field(default_factory=dict)
    query_pool: dict = field(default_factory=dict)
    query_domain: str = ""
    list_len: int = 12  # ordinal family only


def load_config(path: Path, overrides: argparse.Namespace) -> PilotConfig:
    cfg = PilotConfig(**json.loads(path.read_text(encoding="utf-8")))
    if overrides.n is not None:
        cfg.n_per_k = overrides.n
    if overrides.shots is not None:
        cfg.shots = overrides.shots
    if overrides.ks is not None:
        cfg.ks = [int(k) for k in overrides.ks.split(",")]
    if overrides.run_name is not None:
        cfg.run_name = overrides.run_name
    return cfg


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def run(config: PilotConfig, backend=None) -> Path:
    out_dir = Path(config.results_dir) / config.run_name
    if out_dir.exists():
        sys.exit(
            f"refusing to overwrite existing run dir {out_dir}; "
            "use --run-name for a fresh run"
        )
    out_dir.mkdir(parents=True)

    extras = {"stratum": config.stratum}
    if config.operand_pool:
        extras["operand_pool"] = config.operand_pool
    if config.query_pool:
        extras["query_pool"] = config.query_pool
    if config.query_domain:
        extras["query_domain"] = config.query_domain
    if "ordinal" in config.variants:
        extras["list_len"] = config.list_len
    items = [
        item
        for variant in config.variants
        for k in config.ks
        for item in P.build_prompt_set(
            variant, k, config.n_per_k, config.shots, config.seed, **extras
        )
    ]
    prompt_sha = P.write_prompt_set(items, out_dir / "prompts.jsonl")
    print(
        f"{len(items)} prompts -> {out_dir / 'prompts.jsonl'} "
        f"(sha256 {prompt_sha[:12]})",
        flush=True,
    )

    if backend is None:
        from .backends import build_backend

        backend = build_backend(config)

    scores_path = out_dir / "scores.jsonl"
    with scores_path.open("w", encoding="utf-8", newline="\n") as f:
        done = 0
        for variant in config.variants:
            for k in config.ks:
                group = [
                    (i, it)
                    for i, it in enumerate(items)
                    if it.variant == variant and it.k == k
                ]
                scores = backend.score_choices(
                    [it.prompt for _, it in group],
                    [it.choices for _, it in group],
                )
                for (i, _), s in zip(group, scores):
                    f.write(
                        json.dumps(
                            {"idx": i, **dataclasses.asdict(s)}, sort_keys=True
                        )
                        + "\n"
                    )
                done += len(group)
                print(f"scored {variant} k={k} ({done}/{len(items)})", flush=True)

    manifest = {
        "config": dataclasses.asdict(config),
        "config_sha256": hashlib.sha256(
            json.dumps(dataclasses.asdict(config), sort_keys=True).encode()
        ).hexdigest(),
        "prompts_sha256": prompt_sha,
        "git_commit": git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": _library_versions(),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {out_dir / 'manifest.json'}")
    return out_dir


def _library_versions() -> dict[str, str]:
    versions = {}
    for name in ("torch", "transformers", "numpy"):
        try:
            versions[name] = __import__(name).__version__
        except ImportError:
            versions[name] = "not installed"
    return versions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="experiments/<run>.json")
    parser.add_argument("--n", type=int, default=None, help="override n_per_k")
    parser.add_argument("--shots", type=int, default=None, help="override shots")
    parser.add_argument("--ks", default=None, help="override ks, e.g. 0,1,2")
    parser.add_argument("--run-name", default=None, help="override run_name")
    args = parser.parse_args(argv)
    run(load_config(args.config, args))


if __name__ == "__main__":
    main()
