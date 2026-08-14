"""Next-item collapse diagnostic. Pure numpy, no torch — runs off saved .npz.

D19 found that under a 4-operand pool the FVs steer toward shift +/-1 regardless
of k, in the matched condition as well as the transferred one. That makes
"does this condition's FV encode shift-by-k, or has it collapsed to next-item?"
a question every condition has to answer before its criterion is read, rather
than a property assumed from the operand-pool size.

Reads `efficacy_pred_shift` (n_k, n_queries), the signed shift
(argmax - query) mod n of each injected zero-shot prediction.

Usage:
    python -m tarcle.nextitem results/fv/<run_a> [results/fv/<run_b> ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# k where +/-1 IS the correct answer, so collapse is indistinguishable from success
TRIVIAL_K = {0, 1, 11}


def analyse(run_dir: Path, method: str = "todd") -> dict | None:
    hits = sorted(run_dir.glob(f"fv_*_{method}.npz"))
    if not hits:
        return None
    z = np.load(hits[0], allow_pickle=False)
    if "efficacy_pred_shift" not in z.files:
        print(f"{run_dir.name}: no efficacy_pred_shift (run predates the schema)")
        return None
    meta = json.loads(str(z["meta_json"]))
    ks = list(int(k) for k in z["ks"])
    shifts = z["efficacy_pred_shift"]
    n = shifts.max() + 1 if shifts.size else 12
    n = 12

    # Non-cyclic families (the unrelated tasks) have no signed shift, so the
    # +/-1 attractor is undefined. The gate's purpose survives unchanged — a
    # control whose FVs encode nothing is vacuous, and for the null control that
    # vacuity would look like success — so it is read against chance instead.
    if shifts.size == 0:
        acc, base = z["efficacy_acc"], z["efficacy_baseline"]
        # Raw injected accuracy cannot carry this gate: several of these tasks
        # are answerable zero-shot ("Q: France / A:" -> Paris), so a task already
        # at ceiling scores 1.00 whatever the FV does. Only the *lift* over each
        # task's own baseline is evidence, and tasks whose baseline is already
        # high are uninformative by construction -- the D2 ceiling caveat again.
        CEILING = 0.50
        rows = [{"k": int(k), "p_correct": float(acc[i]),
                 "baseline": float(base[i]), "lift": float(acc[i] - base[i]),
                 "informative": bool(base[i] < CEILING),
                 "p_pm1": None, "top_shift": None}
                for i, k in enumerate(ks)]
        info = [r for r in rows if r["informative"]]
        pc = float(np.mean([r["lift"] for r in info])) if info else float("nan")

        print(f"\n{run_dir.name}  ({meta['condition']}, non-cyclic, {method})")
        print(f"  {'task':>4} {'zero-shot':>10} {'injected':>9} {'lift':>7}")
        for r in rows:
            mark = "" if r["informative"] else "   (at ceiling zero-shot)"
            print(f"  {r['k']:>4} {r['baseline']:>10.2f} {r['p_correct']:>9.2f} "
                  f"{r['lift']:>+7.2f}{mark}")
        verdict = "ENCODES TASK" if pc > 0.30 else "COLLAPSED / vacuous"
        print(f"  mean lift over the {len(info)} tasks with baseline < {CEILING}: "
              f"{pc:+.3f}")
        print(f"  task-encoding gate (D20, non-cyclic form, mean lift > 0.30): "
              f"{verdict}")
        return {"run": run_dir.name, "condition": meta["condition"],
                "pool_size": None, "method": method, "rows": rows,
                "p_correct_mid": pc, "p_pm1_mid": None, "margin": None,
                "n_informative": len(info), "verdict": verdict}

    rows = []
    for i, k in enumerate(ks):
        c = np.bincount(shifts[i], minlength=n).astype(float)
        c /= c.sum()
        rows.append({
            "k": k, "p_correct": float(c[k]),
            "p_pm1": float(c[1] + c[n - 1]),
            "top_shift": int(np.argmax(c)),
        })

    mid = [r for r in rows if r["k"] not in TRIVIAL_K]
    pc = float(np.mean([r["p_correct"] for r in mid]))
    pm1 = float(np.mean([r["p_pm1"] for r in mid]))
    cycle = meta.get("cycle_domain", meta["family"])
    pool = len(meta["operand_pool"].get(cycle, next(iter(meta["operand_pool"].values()))))

    print(f"\n{run_dir.name}  ({meta['condition']}, {pool}-operand pool, "
          f"{method})")
    print(f"  {'k':>3} {'P(correct)':>11} {'P(+/-1)':>9} {'top shift':>10}")
    for r in rows:
        mark = " *" if r["k"] in TRIVIAL_K else ""
        print(f"  {r['k']:>3} {r['p_correct']:>11.2f} {r['p_pm1']:>9.2f} "
              f"{r['top_shift']:>10}{mark}")
    print(f"  * k where +/-1 is the correct answer; excluded from the summary")
    print(f"  over k not in {sorted(TRIVIAL_K)}:  P(correct) {pc:.3f}   "
          f"P(+/-1) {pm1:.3f}   margin {pc - pm1:+.3f}")
    verdict = "ENCODES TASK" if pc - pm1 > 0.10 else (
        "COLLAPSED to next-item" if pm1 > pc else "AMBIGUOUS")
    print(f"  task-encoding gate (D20, margin > 0.10): {verdict}")

    return {"run": run_dir.name, "condition": meta["condition"],
            "pool_size": pool, "method": method, "rows": rows,
            "p_correct_mid": pc, "p_pm1_mid": pm1, "margin": pc - pm1,
            "verdict": verdict}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--method", default="todd")
    args = parser.parse_args(argv)
    out = [r for d in args.runs if (r := analyse(d, args.method))]
    if out:
        # The filename is stamped with a hash of the input run set: a fixed
        # name let successive invocations silently destroy each other's
        # output, which is how the six-months-conditions version of this
        # file was lost (docs/decisions.md D45). Identical regeneration is
        # allowed; differing content at the same path is refused.
        import hashlib

        stamp = hashlib.sha256(
            "|".join(sorted(d.name for d in args.runs)).encode()
        ).hexdigest()[:8]
        dest = args.runs[0].parent / f"nextitem_{args.method}_{stamp}.json"
        text = json.dumps(out, indent=2) + "\n"
        if dest.exists():
            if dest.read_text(encoding="utf-8") == text:
                print(f"\n{dest} already exists with identical content")
                return
            raise SystemExit(
                f"refusing to overwrite {dest} with differing content "
                "(CLAUDE.md: results are never overwritten)"
            )
        dest.write_text(text, encoding="utf-8", newline="\n")
        print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
