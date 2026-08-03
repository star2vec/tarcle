"""The two negative-control families: unrelated tasks and the ordinal family.

These are the controls that license every stage-2 diagnostic — prereg §5 voids
the whole run if the unrelated tasks show circulant structure — so what is
checked here is that they are what they claim to be: no cyclic structure, no
leakage of the answer into the prompt, enough operand diversity to clear the
pilot_findings §9 collapse threshold, and byte-identical regeneration.
"""
from __future__ import annotations

import hashlib

import pytest

from tarcle import prompts as P
from tarcle.tasks_unrelated import TASK_NAMES, UNRELATED_TASKS, task_choices, validate


def test_unrelated_task_invariants():
    validate()
    assert len(TASK_NAMES) == 12
    assert TASK_NAMES == sorted(TASK_NAMES)


def test_unrelated_tasks_do_not_share_operands():
    """Two tasks sharing operands would couple their FVs through the operand
    distribution — the very confound the months controls exist to exclude."""
    for i, a in enumerate(TASK_NAMES):
        for b in TASK_NAMES[i + 1:]:
            shared = {x for x, _ in UNRELATED_TASKS[a]} & {
                x for x, _ in UNRELATED_TASKS[b]
            }
            # some overlap is unavoidable across natural-language tasks; the
            # constraint is that no pair is near-duplicated
            assert len(shared) <= 6, (a, b, sorted(shared))


def test_unrelated_prompt_set_shape_and_diversity():
    items = P.make_unrelated_prompt_set(0, 40, 16, 0)
    assert len(items) == 40
    for it in items:
        assert it.variant == "unrelated" and it.k == 0
        assert it.domain == TASK_NAMES[0]
        assert it.target in it.choices
        assert it.query not in [d[1] for d in it.demos]  # held out
        # pilot_findings §9: FVs collapse below ~9 distinct demo operands
        assert len({d[1] for d in it.demos}) >= 9


def test_unrelated_target_never_appears_as_a_demo_target_for_the_query():
    """Guards answer leakage: the query's own pair must not be shown."""
    for task_index in range(12):
        for it in P.make_unrelated_prompt_set(task_index, 10, 16, 0):
            for _, operand, target in it.demos:
                assert not (operand == it.query and target == it.target)


def test_ordinal_prompt_set_is_ordinal_not_cyclic():
    items = P.make_ordinal_prompt_set(3, 20, 8, 0)
    assert len(items) == 20
    for it in items:
        assert it.variant == "ordinal" and it.k == 3
        assert len(it.choices) == 12
        assert it.target == it.choices[2]  # 1-indexed position 3
        for _, operand, target in it.demos:
            words = operand.split(", ")
            assert len(words) == 12
            assert target == words[2]


def test_ordinal_covers_every_position_and_chance_matches_months():
    for k in range(1, 13):
        it = P.make_ordinal_prompt_set(k, 1, 4, 0)[0]
        assert it.target == it.choices[k - 1]
        assert len(it.choices) == 12  # chance 1/12, same as the months cycle
    with pytest.raises(ValueError, match="outside"):
        P.make_ordinal_prompt_set(13, 1, 4, 0)
    with pytest.raises(ValueError, match="outside"):
        P.make_ordinal_prompt_set(0, 1, 4, 0)


def test_ordinal_query_list_is_fresh_each_item():
    """If the query list repeated across items the task would degenerate into
    recalling one list rather than indexing an arbitrary one."""
    items = P.make_ordinal_prompt_set(5, 30, 8, 0)
    assert len({it.query for it in items}) == 30


def test_both_families_regenerate_byte_identically():
    for build in (
        lambda: P.make_ordinal_prompt_set(7, 25, 16, 3),
        lambda: P.make_unrelated_prompt_set(5, 25, 16, 3),
    ):
        a = hashlib.sha256(P.serialize_items(build()).encode()).hexdigest()
        b = hashlib.sha256(P.serialize_items(build()).encode()).hexdigest()
        assert a == b


def test_corrupt_labels_works_on_both_families():
    for items in (
        P.make_ordinal_prompt_set(4, 5, 8, 0),
        P.make_unrelated_prompt_set(2, 5, 8, 0),
    ):
        for i, it in enumerate(items):
            bad = P.corrupt_labels(it, 0, i)
            assert sorted(d[2] for d in bad.demos) == sorted(d[2] for d in it.demos)
            assert bad.query == it.query and bad.target == it.target
            assert [d[1] for d in bad.demos] == [d[1] for d in it.demos]
