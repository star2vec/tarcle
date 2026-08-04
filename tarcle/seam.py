"""Seam contest (D31). Pure numpy — stage-2 code, no torch, no GPU.

A cyclic family has no seam: every point is interior. A linear family that has
been wrapped for analysis has exactly one, where the endpoints were glued. So
unrolling the parameter at each possible cut and asking which unrolling best
explains the pairwise distances tests cyclicity directly, which neither
permutation z nor the binned separation profile does (D30).

Models, each scored against the observed pairwise distances:

    cyclic      separation = min(d, n-d)          6 levels at n=12
    cut@c       separation = |pos_i - pos_j|,     11 levels
                pos(k) = (k - c) mod n

Cross-validated isotonic R2 is the verdict metric. In-sample fit is reported but
must not be compared across models: the cut models carry eleven separation
levels against the cyclic model's six and would win on degrees of freedom alone.

Usage:
    python -m tarcle.seam
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .stage2 import load

R = Path("results/fv")
N = 12  # the months modulus; add-k is unrolled against the same hypothetical n


def pair_index(ks: list[int]):
    """Upper-triangle pairs as (i, j) index arrays."""
    i, j = np.triu_indices(len(ks), k=1)
    return i, j


def separations(ks: list[int], model: str, n: int = N) -> np.ndarray:
    i, j = pair_index(ks)
    a, b = np.array(ks)[i], np.array(ks)[j]
    if model == "cyclic":
        d = np.abs(a - b)
        return np.minimum(d, n - d)
    c = int(model.split("@")[1])
    pa, pb = (a - c) % n, (b - c) % n
    return np.abs(pa - pb)


def isotonic(x: np.ndarray, y: np.ndarray) -> dict[float, float]:
    """Monotone step fit: the mean of y within each distinct x level.

    With a small number of integer levels the isotonic solution is the level
    means whenever those are already non-decreasing; pool-adjacent-violators is
    applied when they are not.
    """
    levels = np.unique(x)
    means = [y[x == v].mean() for v in levels]
    weights = [float((x == v).sum()) for v in levels]
    # pool adjacent violators
    i = 0
    while i < len(means) - 1:
        if means[i] > means[i + 1]:
            w = weights[i] + weights[i + 1]
            means[i] = (means[i] * weights[i] + means[i + 1] * weights[i + 1]) / w
            weights[i] = w
            del means[i + 1], weights[i + 1]
            levels = np.delete(levels, i + 1)
            i = max(i - 1, 0)
        else:
            i += 1
    return {float(v): float(m) for v, m in zip(levels, means)}


def predict(fit: dict[float, float], x: np.ndarray) -> np.ndarray:
    keys = np.array(sorted(fit))
    vals = np.array([fit[k] for k in keys])
    idx = np.clip(np.searchsorted(keys, x), 0, len(keys) - 1)
    return vals[idx]


def cv_r2(sep: np.ndarray, dist: np.ndarray, seed: int = 0, folds: int = 40) -> float:
    """Cross-validated isotonic R2: fit on half the pairs, score the other half."""
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(folds):
        mask = rng.random(len(sep)) < 0.5
        if mask.sum() < 3 or (~mask).sum() < 3:
            continue
        if len(np.unique(sep[mask])) < 2:
            continue
        pred = predict(isotonic(sep[mask], dist[mask]), sep[~mask])
        held = dist[~mask]
        ss_res = float(((held - pred) ** 2).sum())
        ss_tot = float(((held - held.mean()) ** 2).sum())
        if ss_tot > 0:
            scores.append(1 - ss_res / ss_tot)
    return float(np.mean(scores)) if scores else float("nan")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    def rank(x):
        order = np.argsort(x)
        r = np.empty(len(x), float)
        r[order] = np.arange(len(x), dtype=float)
        _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        s = np.zeros(len(cnt))
        np.add.at(s, inv, r)
        return (s / cnt)[inv]
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def contest(X: np.ndarray, ks: list[int], n: int = N) -> dict:
    i, j = pair_index(ks)
    dist = np.linalg.norm(X[i] - X[j], axis=1)
    # Every cut point on the circle, not only those carrying data — the seam can
    # sit in the gap left by a dropped parameter value. For months with k=0
    # dropped, cut@0 and cut@1 give identical separations (only |pos_i - pos_j|
    # matters, and the two differ by a constant shift), so they tie; that tie is
    # the D31 "seam at k=0" model and is left visible rather than deduplicated.
    models = ["cyclic"] + [f"cut@{c}" for c in range(n)]
    out = {}
    for m in models:
        sep = separations(ks, m, n)
        out[m] = {
            "cv_r2": cv_r2(sep, dist),
            "spearman": spearman(sep, dist),
            "levels": int(len(np.unique(sep))),
        }
    return out


def show(label: str, X: np.ndarray, ks: list[int]) -> dict:
    res = contest(X, ks)
    ranked = sorted(res.items(), key=lambda kv: -kv[1]["cv_r2"])
    cyc = res["cyclic"]["cv_r2"]
    best_cut, best = next((k, v) for k, v in ranked if k != "cyclic")
    print(f"\n  {label}")
    print(f"    {'model':10} {'cv R2':>8} {'spearman':>9} {'levels':>7}")
    for name, v in ranked[:4]:
        star = "  <-- cyclic" if name == "cyclic" else ""
        print(f"    {name:10} {v['cv_r2']:>8.4f} {v['spearman']:>9.4f} "
              f"{v['levels']:>7}{star}")
    print(f"    winner: {ranked[0][0]}   cyclic {cyc:.4f} vs best cut "
          f"{best_cut} {best['cv_r2']:.4f}   margin {cyc - best['cv_r2']:+.4f}")
    return {"models": res, "winner": ranked[0][0], "cyclic_cv_r2": cyc,
            "best_cut": best_cut, "best_cut_cv_r2": best["cv_r2"]}


def main() -> None:
    out = {}
    months_ks = list(range(1, 12))

    # The claim that centering is irrelevant here, checked rather than asserted.
    _t = load(R / "ctl_months_primary" / "fv_primary_todd.npz")["X"].astype(np.float64)
    _i, _j = pair_index(list(range(12)))
    assert np.allclose(
        np.linalg.norm(_t[_i] - _t[_j], axis=1),
        np.linalg.norm((_t - _t.mean(0))[_i] - (_t - _t.mean(0))[_j], axis=1),
    ), "distances are not translation-invariant"
    print("centering is a no-op for this test (distances are translation-invariant)")

    for method in ("todd", "hendel"):
        print(f"\n{'=' * 72}\n{method}\n{'=' * 72}")

        # Centering translates every point equally and pairwise distances are
        # translation-invariant, so the centered column is identical to the raw
        # one by construction. Asserted once rather than reported as a second
        # column: unlike the Gram-based diagnostics, this test is immune to the
        # shared offset that D27 exists for.
        def variants(X):
            return [("raw", X)]

        # ---- validation gate: add-k must show a seam, the null must show nothing
        print("\n  VALIDATION GATE (read before months)")
        addk = load(R / "ctl_addk" / f"fv_addk_{method}.npz")["X"].astype(np.float64)
        for tag, V in variants(addk):
            out[f"{method}/addk/{tag}"] = show(f"add-k [{tag}] — a seam is KNOWN to exist",
                                               V, months_ks)
        null = load(R / "ctl_unrelated_12tasks" / f"fv_unrelated_{method}.npz")["X"]
        null = null.astype(np.float64)
        for tag, V in variants(null):
            out[f"{method}/null/{tag}"] = show(f"unrelated [{tag}] — nothing should fit",
                                               V, list(range(12)))

        gate_ok = (
            out[f"{method}/addk/raw"]["winner"] != "cyclic"
            and abs(out[f"{method}/null/raw"]["cyclic_cv_r2"]) < 0.15
        )
        print(f"\n  gate: add-k seam detected = "
              f"{out[f'{method}/addk/raw']['winner'] != 'cyclic'}; "
              f"null fits nothing = "
              f"{abs(out[f'{method}/null/raw']['cyclic_cv_r2']) < 0.15}"
              f"  -> {'PASS' if gate_ok else 'FAIL — months not read'}")
        if not gate_ok:
            continue

        # ---- months
        print("\n  MONTHS (n=11, k=0 dropped)")
        d = load(R / "ctl_months_primary" / f"fv_primary_{method}.npz")
        keep = [i for i, k in enumerate(d["ks"]) if k != 0]
        M = d["X"].astype(np.float64)[keep]
        for tag, V in variants(M):
            out[f"{method}/months/{tag}"] = show(f"months [{tag}]", V, months_ks)

        # ---- the pairs the wraparound claim actually rests on
        i, j = pair_index(months_ks)
        a, b = np.array(months_ks)[i], np.array(months_ks)[j]
        dd = np.abs(a - b)
        dist = np.linalg.norm(M[i] - M[j], axis=1)
        scale = dist.mean()
        print(f"\n    ten fold-back pairs (linear sep > 6, cyclic sep = 12 - it)"
              f"   [mean pair distance {scale:.3f}]")
        for idx in np.argsort(dd)[::-1]:
            if dd[idx] > 6:
                print(f"      k={a[idx]:>2},{b[idx]:>2}  linear {dd[idx]:>2}  "
                      f"cyclic {min(dd[idx], N - dd[idx]):>2}  "
                      f"dist {dist[idx]:.3f}  ({dist[idx] / scale:.2f}x mean)")
        print("    five antipode-crossing pairs (cyclic sep 6)")
        for idx in range(len(dd)):
            if min(dd[idx], N - dd[idx]) == 6:
                print(f"      k={a[idx]:>2},{b[idx]:>2}  linear {dd[idx]:>2}  "
                      f"dist {dist[idx]:.3f}  ({dist[idx] / scale:.2f}x mean)")

    Path("results/stage2").mkdir(parents=True, exist_ok=True)
    Path("results/stage2/seam.json").write_text(
        json.dumps(out, indent=2, default=float) + "\n",
        encoding="utf-8", newline="\n")
    print("\nwrote results/stage2/seam.json")


if __name__ == "__main__":
    main()
