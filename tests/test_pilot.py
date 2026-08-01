"""Pilot pipeline tests. Whole suite must run on CPU in < 5 min (CLAUDE.md)."""
from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from tarcle import prompts as P
from tarcle.pilot import PilotConfig, run
from tarcle.pilot_report import accuracy_table, gate_verdict, load_run, main as report_main


def test_shift_wraparound():
    assert P.shift("days", "Saturday", 2) == "Monday"
    assert P.shift("days", "Sunday", 1) == "Monday"
    assert P.shift("days", "Monday", 0) == "Monday"
    assert P.shift("days", "Monday", 7) == "Monday"
    assert P.shift("months", "December", 3) == "March"
    assert P.shift("letters", "z", 1) == "a"
    assert P.shift("digits", "9", 1) == "0"


def test_prompt_set_determinism(tmp_path):
    a = P.make_prompt_set("mixed", 3, 40, 10, seed=0)
    b = P.make_prompt_set("mixed", 3, 40, 10, seed=0)
    assert P.serialize_items(a) == P.serialize_items(b)
    assert P.serialize_items(a) != P.serialize_items(
        P.make_prompt_set("mixed", 3, 40, 10, seed=1)
    )
    sha_a = P.write_prompt_set(a, tmp_path / "a.jsonl")
    sha_b = P.write_prompt_set(b, tmp_path / "b.jsonl")
    assert sha_a == sha_b


def test_stratification():
    for variant in ("days", "mixed"):
        items = P.make_prompt_set(variant, 2, 60, 10, seed=0)
        copy = [it for it in items if it.query_in_demos]
        heldout = [it for it in items if not it.query_in_demos]
        assert len(copy) == len(heldout) == 30
        for it in copy:
            assert [it.domain, it.query, it.target] in it.demos
        for it in heldout:
            assert all(
                not (d == it.domain and op == it.query) for d, op, _ in it.demos
            )


def test_targets_and_choices():
    items = P.make_prompt_set("mixed", 4, 60, 10, seed=0)
    assert {it.domain for it in items} == set(P.MIXED_DOMAINS)
    for it in items:
        assert it.target == P.shift(it.domain, it.query, it.k)
        assert it.choices == P.DOMAINS[it.domain]
        assert it.prompt.endswith(f"Q: {it.query}\nA:")
        assert it.prompt.count("Q:") == 11  # 10 demos + query
        for d, op, tgt in it.demos:
            assert tgt == P.shift(d, op, it.k)


class OracleBackend:
    """Scores the correct choice highest; lets pipeline tests run without a model."""

    def __init__(self, items):
        self.targets = {it.prompt: it.target for it in items}

    def score_choices(self, prompts, choices_per_prompt):
        from tarcle.backends import ChoiceScores

        return [
            ChoiceScores(
                choice_logprobs=[
                    0.0 if c == self.targets[p] else -10.0 for c in choices
                ],
                top_tokens=[" " + self.targets[p]],
                top_logprobs=[0.0],
            )
            for p, choices in zip(prompts, choices_per_prompt)
        ]


def _tiny_config(tmp_path, run_name, **kwargs) -> PilotConfig:
    defaults = dict(
        run_name=run_name, model="oracle", device="cpu", dtype="float32",
        variants=["days", "mixed"], ks=[0, 1, 2], n_per_k=4, shots=3,
        results_dir=str(tmp_path / "results"),
    )
    defaults.update(kwargs)
    return PilotConfig(**defaults)


def test_run_and_report_with_oracle(tmp_path, capsys):
    config = _tiny_config(tmp_path, "oracle_run")
    all_items = [
        it
        for v in config.variants
        for k in config.ks
        for it in P.make_prompt_set(v, k, config.n_per_k, config.shots, config.seed)
    ]
    out_dir = run(config, backend=OracleBackend(all_items))

    for name in ("prompts.jsonl", "scores.jsonl", "manifest.json"):
        assert (out_dir / name).exists()
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["prompts_sha256"]
    assert manifest["config"]["run_name"] == "oracle_run"

    items, _ = load_run(out_dir)
    assert all(it["correct"] for it in items)
    verdict = gate_verdict(accuracy_table(items), config.ks)
    assert verdict["verdict"] == "GO"

    report_main([str(out_dir)])
    capsys.readouterr()
    for name in ("accuracy.csv", "accuracy_vs_k.png", "confusion_days.png",
                 "report.json"):
        assert (out_dir / name).exists()


def test_never_overwrites(tmp_path):
    config = _tiny_config(tmp_path, "dup", variants=["days"], ks=[0])
    items = P.make_prompt_set("days", 0, config.n_per_k, config.shots, config.seed)
    run(config, backend=OracleBackend(items))
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        run(config, backend=OracleBackend(items))


def test_gate_verdict_bands():
    def table(accs):
        return [
            {"variant": "days", "k": k, "stratum": "heldout", "domain": "all",
             "n": 50, "accuracy": a}
            for k, a in accs.items()
        ]

    ks = [0, 1, 2]
    assert gate_verdict(table({0: 0.9, 1: 0.6, 2: 0.5}), ks)["verdict"] == "GO"
    assert gate_verdict(table({0: 0.9, 1: 0.6, 2: 0.3}), ks)["verdict"] == "NO-GO"
    assert gate_verdict(table({0: 0.9, 1: 0.6, 2: 0.4}), ks)["verdict"] == "MARGINAL"
    assert gate_verdict(table({0: 0.9}), ks)["verdict"] == "INCOMPLETE"


def test_slow_path_matches_fast_path():
    """Teacher-forced continuation scoring must agree with the next-token fast path
    on single-token candidates (guards the position indexing of the slow path)."""
    from tarcle.backends import LocalHFBackend

    backend = LocalHFBackend("gpt2", device="cpu", dtype="float32")
    prompt = "Q: Monday\nA: Tuesday\n\nQ: Friday\nA:"
    fast = backend.score_choices([prompt], [["Saturday"]])[0].choice_logprobs[0]
    slow = backend._score_continuation(prompt, backend._encode_choice("Saturday"))
    assert fast == pytest.approx(slow, abs=1e-4)

    multi_ids = backend._encode_choice("Saturdayish")
    assert len(multi_ids) > 1
    total = backend._score_continuation(prompt, multi_ids)
    assert total < backend._score_continuation(prompt, multi_ids[:1])


def test_end_to_end_gpt2(tmp_path):
    """The real-model smoke path: gpt2 on CPU, tiny prompt set."""
    from tarcle.backends import build_backend

    config = _tiny_config(
        tmp_path, "gpt2_tiny", model="gpt2", variants=["days"], ks=[0, 1],
        n_per_k=4, shots=3, batch_size=4,
    )
    out_dir = run(config, backend=build_backend(config))
    items, _ = load_run(out_dir)
    assert len(items) == 8
    for it in items:
        assert len(it["choice_logprobs"]) == 7
        assert all(lp < 0 for lp in it["choice_logprobs"])
        assert len(it["top_tokens"]) == 5
