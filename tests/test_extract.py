"""Stage-1 extraction on gpt2/CPU: the CI path CLAUDE.md requires.

These tests check mechanism, not science — gpt2 cannot do shift-by-k (pilot §5),
so no AIE value here means anything. What they assert is that the hooks read and
write the tensors we think they do, that the head->residual projection is the
identity it claims to be, and that prompt determinism survives the new keyword
arguments.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tarcle import prompts as P
from tarcle.extract import (
    Arch,
    ExtractConfig,
    LastPositionHeads,
    PatchHeadsDiagonal,
    describe,
    encode,
    load_model,
    mean_head_activations,
    sweep,
    target_probs,
)

CONFIG = ExtractConfig(
    run_name="test", model="gpt2", device="cpu", dtype="float32", batch_size=4,
    family="months", shots=2, head_id_ks=[1, 2], head_id_n_prompts=1,
)


@pytest.fixture(scope="module")
def loaded():
    model, tok = load_model(CONFIG)
    return model, tok, describe(model)


def test_arch_shapes(loaded):
    _, _, arch = loaded
    assert (arch.n_layers, arch.n_heads, arch.head_dim, arch.d_model) == (12, 12, 64, 768)
    for w in arch.w_out:
        assert w.shape == (arch.d_model, arch.n_heads * arch.head_dim)


def test_left_padding_puts_last_real_token_at_minus_one(loaded):
    model, tok, _ = loaded
    enc = encode(tok, ["Q: a\nA:", "Q: a longer prompt here\nA:"], "cpu")
    assert torch.all(enc["attention_mask"][:, -1] == 1)
    # position_ids restart at 0 on the first real token, so the padded row is
    # not silently shifted by the pad length.
    assert enc["position_ids"][0, -1] == enc["attention_mask"][0].sum() - 1


def test_captured_heads_reconstruct_the_attention_output(loaded):
    """The core claim of Todd-style extraction: the o_proj input, sliced by head
    and pushed through the matching columns of W_O, sums to the attention
    block's contribution to the residual stream. If this is off, every FV is."""
    model, tok, arch = loaded
    enc = encode(tok, ["Q: Monday\nA: Tuesday\n\nQ: Friday\nA:"], "cpu")
    captured = {}
    handle = arch.attn_out[3].register_forward_hook(
        lambda m, args, out: captured.update(inp=args[0], out=out)
    )
    with LastPositionHeads(arch) as cap, torch.inference_mode():
        model(**enc, logits_to_keep=1)
    handle.remove()

    # no_grad, not inference_mode: w_out requires grad, and multiplying it by an
    # inference tensor raises. FV assembly hits the same wall and must do its
    # head->residual projection under no_grad for the same reason.
    with torch.no_grad():
        heads = cap.acts[3][0].clone()  # (n_heads, head_dim) at the last position
        w = arch.w_out[3]
        per_head = torch.stack(
            [w[:, h * arch.head_dim : (h + 1) * arch.head_dim] @ heads[h]
             for h in range(arch.n_heads)]
        )
        reference = captured["out"][0, -1].clone() - arch.attn_out[3].bias
    assert torch.allclose(per_head.sum(0), reference, atol=1e-4)


def test_patch_writes_only_its_own_head_and_slot(loaded):
    model, tok, arch = loaded
    enc = encode(tok, ["Q: Monday\nA:"] * arch.n_heads, "cpu")
    abar = torch.arange(arch.n_heads * arch.head_dim, dtype=torch.float32).view(
        arch.n_heads, arch.head_dim
    )
    with LastPositionHeads(arch) as cap, torch.inference_mode():
        model(**enc, logits_to_keep=1)
    clean = cap.acts[5].clone()
    with PatchHeadsDiagonal(arch, 5, abar), LastPositionHeads(arch) as cap2, \
            torch.inference_mode():
        model(**enc, logits_to_keep=1)
    patched = cap2.acts[5]

    for slot in range(arch.n_heads):
        for head in range(arch.n_heads):
            if slot == head:
                assert torch.allclose(patched[slot, head], abar[head])
            else:
                assert torch.allclose(patched[slot, head], clean[slot, head])


def test_patching_a_layer_leaves_earlier_layers_untouched(loaded):
    """Guards the 'layers cannot be batched together' assumption from the other
    side: a patch at layer 5 must not perturb layer 4's captured activations."""
    model, tok, arch = loaded
    enc = encode(tok, ["Q: Monday\nA:"] * arch.n_heads, "cpu")
    abar = torch.ones(arch.n_heads, arch.head_dim)
    with LastPositionHeads(arch) as a, torch.inference_mode():
        model(**enc, logits_to_keep=1)
    with PatchHeadsDiagonal(arch, 5, abar), LastPositionHeads(arch) as b, \
            torch.inference_mode():
        model(**enc, logits_to_keep=1)
    assert torch.allclose(a.acts[4], b.acts[4])
    assert not torch.allclose(a.acts[6], b.acts[6])


def test_mean_head_activations_shape_and_batch_invariance(loaded):
    model, tok, arch = loaded
    items = P.make_prompt_set("months", 1, 4, 2, 0, stratum="heldout")
    prompts = [it.prompt for it in items]
    a = mean_head_activations(model, tok, arch, prompts, batch_size=4)
    b = mean_head_activations(model, tok, arch, prompts, batch_size=1)
    assert a.shape == (arch.n_layers, arch.n_heads, arch.head_dim)
    assert torch.allclose(a, b, atol=1e-4)


def test_target_probs_are_probabilities(loaded):
    model, tok, _ = loaded
    items = P.make_prompt_set("months", 1, 4, 2, 0, stratum="heldout")
    from tarcle.extract import first_token_id

    p = target_probs(
        model, tok, [it.prompt for it in items],
        [first_token_id(tok, it.target) for it in items], batch_size=2,
    )
    assert p.shape == (4,)
    assert np.all((p >= 0) & (p <= 1))


def test_sweep_runs_end_to_end(loaded):
    model, tok, arch = loaded
    aie, se, n, per_k, baseline = sweep(
        model, tok, arch, CONFIG, layers=[0, 1], log=lambda *_: None
    )
    assert aie.shape == (arch.n_layers, arch.n_heads)
    assert per_k.shape == (len(CONFIG.head_id_ks), arch.n_layers, arch.n_heads)
    assert n == len(CONFIG.head_id_ks) * CONFIG.head_id_n_prompts
    assert 0.0 <= baseline <= 1.0
    assert np.all(aie[2:] == 0)  # layers not swept stay at zero
    assert np.any(aie[:2] != 0)


# --------------------------------------------------------------------------
# FV assembly and injection
# --------------------------------------------------------------------------


def test_project_heads_matches_the_summed_attention_contribution(loaded):
    from tarcle.extract import per_prompt_head_acts, project_heads

    model, tok, arch = loaded
    prompts = ["Q: Monday\nA: Tuesday\n\nQ: Friday\nA:"]
    cells = [(3, h) for h in range(arch.n_heads)]
    acts = per_prompt_head_acts(model, tok, arch, prompts, cells, 4)
    contrib = project_heads(arch, cells, acts)
    assert contrib.shape == (1, arch.n_heads, arch.d_model)

    captured = {}
    handle = arch.attn_out[3].register_forward_hook(
        lambda m, a, out: captured.update(out=out)
    )
    with torch.inference_mode():
        model(**encode(tok, prompts, "cpu"), logits_to_keep=1)
    handle.remove()
    with torch.no_grad():
        reference = captured["out"][0, -1].clone() - arch.attn_out[3].bias
    assert torch.allclose(contrib[0].sum(0), reference, atol=1e-4)


def test_summarize_halves_are_disjoint_and_average_to_the_mean():
    from tarcle.extract import summarize

    x = torch.arange(40, dtype=torch.float32).view(10, 4)
    s = summarize(x)
    assert s["n"] == 10
    assert np.allclose(s["mean"], x.numpy().mean(axis=0))
    assert np.allclose((s["half_a"] + s["half_b"]) / 2, s["mean"])
    assert not np.allclose(s["half_a"], s["half_b"])


def test_dummy_query_prompts_keep_demos_and_drop_the_operand():
    from tarcle.extract import dummy_query_prompts

    items = P.make_prompt_set("months", 3, 4, 4, 0, stratum="heldout")
    dummies = dummy_query_prompts(items)
    for item, dummy in zip(items, dummies):
        assert dummy.endswith("Q: x\nA:")
        assert item.query not in dummy.rsplit("Q:", 1)[-1]
        assert dummy.count("Q:") == item.prompt.count("Q:")


def _catch_block_output(arch, layer, store, key):
    """A forward hook must return None or it *replaces* the module output. A
    lambda whose body evaluates to a tensor (dict.setdefault does) silently
    swaps a block's (hidden, ...) tuple for a bare tensor, and the next block
    then receives a 2D input. Hence a def with a bare statement."""

    def hook(_m, _a, out):
        store[key] = (out[0] if isinstance(out, tuple) else out).clone()

    return arch.blocks[layer].register_forward_hook(hook)


def test_injection_changes_the_prediction(loaded):
    from tarcle.causal import InjectResidual, zero_shot_prompts

    model, tok, arch = loaded
    prompts = zero_shot_prompts("months")[:4]
    enc = encode(tok, prompts, "cpu")
    with torch.inference_mode():
        clean = model(**enc, logits_to_keep=1).logits[:, -1].float()
    # NOT torch.ones: a constant vector lies along the all-ones direction, which
    # is exactly what the downstream LayerNorm subtracts off. Steering along it
    # is close to a no-op, and a test using it would pass on broken code too.
    v = torch.Generator().manual_seed(0)
    big = torch.randn(arch.d_model, generator=v) * 20.0
    with InjectResidual(arch, 6, big, "add"), torch.inference_mode():
        hacked = model(**enc, logits_to_keep=1).logits[:, -1].float()
    assert not torch.allclose(clean, hacked)


def test_injection_touches_only_the_last_position(loaded):
    """A steering vector that leaked into earlier positions would be editing the
    demonstrations, not the query, and every efficacy number would be wrong."""
    from tarcle.causal import InjectResidual

    model, tok, arch = loaded
    enc = encode(tok, ["Q: January\nA: April\n\nQ: July\nA:"], "cpu")
    caught = {}
    handle = _catch_block_output(arch, 8, caught, "h")
    with torch.inference_mode():
        model(**enc, logits_to_keep=1)
    before = caught.pop("h")
    v = torch.randn(arch.d_model, generator=torch.Generator().manual_seed(1)) * 20.0
    with InjectResidual(arch, 7, v, "add"), torch.inference_mode():
        model(**enc, logits_to_keep=1)
    after = caught["h"]
    handle.remove()
    assert torch.allclose(before[:, :-1], after[:, :-1], atol=1e-4)
    assert not torch.allclose(before[:, -1], after[:, -1], atol=1e-4)


def test_replace_mode_overwrites_rather_than_adds(loaded):
    from tarcle.causal import InjectResidual

    model, tok, arch = loaded
    enc = encode(tok, ["Q: January\nA:"], "cpu")
    v = torch.randn(arch.d_model, generator=torch.Generator().manual_seed(2))
    caught = {}
    # Order matters: forward hooks on the same module fire in registration
    # order, each seeing the previous one's output. The observer has to be
    # registered *after* the injector or it captures the pre-injection value.
    with InjectResidual(arch, 5, v, "replace"):
        handle = _catch_block_output(arch, 5, caught, "h")
        try:
            with torch.inference_mode():
                model(**enc, logits_to_keep=1)
        finally:
            handle.remove()
    assert torch.allclose(caught["h"][0, -1], v, atol=1e-5)


def test_zero_shot_baseline_is_the_complete_operand_cycle(loaded):
    from tarcle.causal import accuracy_for_k, zero_shot_prompts

    model, tok, arch = loaded
    assert len(zero_shot_prompts("months")) == 12
    acc = accuracy_for_k(model, tok, arch, "months", 0, 8)
    assert 0.0 <= acc <= 1.0
    # a census of the 12 operands, not a sample: accuracy lands on a 12th
    assert abs(acc * 12 - round(acc * 12)) < 1e-9


def test_frequency_proxy_covers_every_operand(loaded):
    from tarcle.causal import frequency_proxy

    model, tok, _ = loaded
    proxy = frequency_proxy(model, tok, "months", 4)
    assert set(proxy) == set(P.DOMAINS["months"])
    assert all(v < 0 for v in proxy.values())  # logprobs


# --------------------------------------------------------------------------
# prompt-set extensions
# --------------------------------------------------------------------------


def test_recorded_prompt_hashes_reproduce():
    """prereg §5 discards a run whose prompt SHA-256 fails to reproduce.

    Every recorded pilot run is regenerated from its own manifest — including
    the per-condition fields (stratum, operand_pool, query_pool, query_domain)
    that D20 §3 added, which is why the config is read rather than assumed. Runs
    predating those fields exercise the legacy defaults and must still match
    byte for byte, which is the guarantee the keyword-only extensions were
    written to preserve.
    """
    import hashlib
    import json
    from pathlib import Path

    # A run writes prompts.jsonl first and manifest.json last, so a directory
    # without a manifest belongs to a run still in progress and has nothing to
    # check against yet.
    runs = [r for r in sorted(Path("results/pilot").iterdir())
            if (r / "manifest.json").exists()]
    assert runs, "no completed pilot runs to check"
    for run in runs:
        manifest = json.loads((run / "manifest.json").read_text())
        cfg = manifest["config"]
        extras = {}
        if cfg.get("stratum"):
            extras["stratum"] = cfg["stratum"]
        for key in ("operand_pool", "query_pool", "query_domain", "list_len"):
            if cfg.get(key):
                extras[key] = cfg[key]
        items = [
            it
            for variant in cfg["variants"]
            for k in cfg["ks"]
            for it in P.build_prompt_set(
                variant, k, cfg["n_per_k"], cfg["shots"], cfg["seed"], **extras
            )
        ]
        sha = hashlib.sha256(P.serialize_items(items).encode()).hexdigest()
        assert sha == manifest["prompts_sha256"], run.name


def test_heldout_stratum_excludes_the_query_operand():
    items = P.make_prompt_set("months", 3, 20, 8, 0, stratum="heldout")
    assert len(items) == 20
    for it in items:
        assert not it.query_in_demos
        assert it.query not in [d[1] for d in it.demos]


def test_operand_pool_restricts_demos_but_not_targets():
    pool = {"months": ["January", "February", "March", "April"]}
    items = P.make_prompt_set(
        "months", 5, 20, 8, 0, stratum="heldout", operand_pool=pool, query_pool=pool
    )
    for it in items:
        assert all(d[1] in pool["months"] for d in it.demos)
        assert it.query in pool["months"]
    # shift-by-5 pushes targets out of the pool: that is the point of the control
    assert any(d[2] not in pool["months"] for it in items for d in it.demos)


def test_query_domain_forces_the_query_while_demos_stay_mixed():
    items = P.make_prompt_set(
        "mixed", 2, 30, 8, 0, stratum="heldout", query_domain="months"
    )
    assert all(it.domain == "months" for it in items)
    assert all(len(it.choices) == 12 for it in items)
    assert {d[0] for it in items for d in it.demos} == set(P.MIXED_DOMAINS)


def test_corrupt_labels_destroys_the_mapping_but_not_the_surface():
    items = P.make_prompt_set("months", 4, 10, 8, 0, stratum="heldout")
    for i, it in enumerate(items):
        bad = P.corrupt_labels(it, 0, i)
        assert [d[1] for d in bad.demos] == [d[1] for d in it.demos]  # operands kept
        assert sorted(d[2] for d in bad.demos) == sorted(d[2] for d in it.demos)
        assert bad.target == it.target and bad.query == it.query
        assert bad.prompt.count("\n") == it.prompt.count("\n")
        shifts = {
            (P.DOMAINS["months"].index(d[2]) - P.DOMAINS["months"].index(d[1])) % 12
            for d in bad.demos
        }
        assert len(shifts) > 1, "corruption left a single consistent shift"


def test_corrupt_labels_is_deterministic():
    item = P.make_prompt_set("months", 4, 1, 8, 0, stratum="heldout")[0]
    assert P.corrupt_labels(item, 0, 0) == P.corrupt_labels(item, 0, 0)
    assert P.corrupt_labels(item, 0, 0) != P.corrupt_labels(item, 1, 0)
