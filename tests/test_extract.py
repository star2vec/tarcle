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
# prompt-set extensions
# --------------------------------------------------------------------------


def test_legacy_prompt_hashes_unchanged():
    """prereg §5 discards a run whose prompt SHA-256 fails to reproduce. The new
    keyword arguments must not disturb the RNG stream of the recorded pilots."""
    import hashlib
    import json
    from pathlib import Path

    for run in Path("results/pilot").iterdir():
        manifest = json.loads((run / "manifest.json").read_text())
        cfg = manifest["config"]
        items = [
            it
            for variant in cfg["variants"]
            for k in cfg["ks"]
            for it in P.make_prompt_set(
                variant, k, cfg["n_per_k"], cfg["shots"], cfg["seed"]
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
