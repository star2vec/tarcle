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
    pool = len(meta["operand_pool"][meta["family"]])

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
        dest = args.runs[0].parent / f"nextitem_{args.method}.json"
        dest.write_text(json.dumps(out, indent=2) + "\n",
                        encoding="utf-8", newline="\n")
        print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
