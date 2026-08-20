"""T2: the registered injection layer x scale sweep on the saved collapsed FVs.

Registration: docs/preregistration_t7.md §2 (2026-08-14, scope note
2026-08-17). Conditions: the four gate-failed controls. Todd: full 28-layer x
{0.5, 1, 2, 3, 4} grid, additive injection of the saved FV. Hendel: layer-only
at scale 1.0, replacement — NOTE, recorded here and in the launch entry: the
ctl .npz files store only the frozen-layer (L15) dummy-query state, so the
registered "layer-only sweep on the saved FVs, no re-extraction" necessarily
means injecting that saved state at each of the 28 layers. Per-layer Hendel
states were not persisted and re-extracting them is out of T2's scope.

Scored at every grid cell: the D20 margin over the full 12-query cycle and
mid-cycle k (108 predictions/cell), plus acc / logp / per-query logp so the
registered ambiguous branch can be examined without re-running.

Chunked and resumable: one chunk = (condition, method, layer), 224 chunks
total, each written once to a hash-stamped guarded file
(results/t2/chunk_<cond>_<method>_L<layer>_<stamp>.npz). A chunk whose file
already exists is skipped, so the sweep survives interruption. The grid runs
to completion; nothing stops early on a promising cell. The verdict is NOT
read here — tarcle/t2_report.py (numpy-only) assembles chunks and applies the
registered branches.

GPU code — imports torch. Launch requires the hardware sanction in force
(D50–D52 chain: CUDA, or MPS-bf16 if the D52 gate passed).

Usage:
    python -m tarcle.t2_sweep --device mps --dtype bfloat16 [--only cond:method]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from . import causal
from .extract import describe
from .nextitem import TRIVIAL_K
from .results_io import input_stamp

MODEL = "meta-llama/Llama-3.2-3B"
BATCH = 12
KS = list(range(12))
SCALES_TODD = (0.5, 1.0, 2.0, 3.0, 4.0)
SCALES_HENDEL = (1.0,)
OUT = Path("results/t2")

CONDITIONS = [
    ("partA", "results/fv/ctl_months_partA/fv_partition_a_{m}.npz"),
    ("partB", "results/fv/ctl_months_partB/fv_partition_b_{m}.npz"),
    ("halfA", "results/fv/ctl_months_halfA/fv_half_a_{m}.npz"),
    ("halfB", "results/fv/ctl_months_halfB/fv_half_b_{m}.npz"),
]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def margin_ci(shifts: np.ndarray, ks: list[int]) -> tuple[float, float]:
    """D20 margin and its 95% CI half-width over mid-cycle k x queries."""
    rows = [i for i, k in enumerate(ks) if k not in TRIVIAL_K]
    s = shifts[rows]
    kcol = np.array([ks[i] for i in rows])[:, None]
    d = ((s == kcol).astype(float) - np.isin(s, [1, 11]).astype(float)).ravel()
    half = 1.96 * float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
    return float(d.mean()), half


def chunk_path(cond: str, method: str, layer: int, dtype: str,
               device: str) -> Path:
    stamp = input_stamp([cond, method, f"L{layer}", MODEL, dtype, device,
                         "t2_v1"])
    return OUT / f"chunk_{cond}_{method}_L{layer:02d}_{stamp}.npz"


def run_chunk(model, tok, arch, cond: str, method: str, layer: int,
              X: np.ndarray, scales: tuple, mode: str, dtype: str,
              device: str, fv_sha: str) -> None:
    path = chunk_path(cond, method, layer, dtype, device)
    if path.exists():
        return
    n_s = len(scales)
    pred = np.zeros((n_s, 12, 12), dtype=np.int32)
    acc = np.zeros((n_s, 12), dtype=np.float32)
    logp = np.zeros((n_s, 12), dtype=np.float32)
    lpq = np.zeros((n_s, 12, 12), dtype=np.float32)
    margins = np.zeros(n_s, dtype=np.float32)
    cis = np.zeros(n_s, dtype=np.float32)
    for si, scale in enumerate(scales):
        for i, k in enumerate(KS):
            v = torch.tensor(X[i], device=device, dtype=torch.float32)
            s = causal.score_for_k(model, tok, arch, "months", k, BATCH,
                                   v * scale, layer, mode)
            pred[si, i] = s["pred_shift"]
            acc[si, i] = s["acc"]
            logp[si, i] = s["logp"]
            lpq[si, i] = s["logp_per_query"]
        margins[si], cis[si] = margin_ci(pred[si], KS)
    meta = {"condition": cond, "method": method, "layer": layer,
            "scales": list(scales), "mode": mode, "model": MODEL,
            "dtype": dtype, "device": device, "fv_sha256": fv_sha,
            "git_commit": git_commit(), "registration":
            "docs/preregistration_t7.md §2"}
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, pred_shift=pred, acc=acc, logp=logp, logp_per_query=lpq,
             margin=margins, margin_ci95=cis,
             meta_json=json.dumps(meta, sort_keys=True))
    tmp.rename(path)  # atomic-ish: a chunk file, once present, is complete
    best = margins.max()
    print(f"  {cond}/{method} L{layer:02d}: best margin over scales "
          f"{best:+.3f}", flush=True)


def main(argv: list[str] | None = None) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", required=True, choices=("mps", "cuda"))
    ap.add_argument("--dtype", required=True, choices=("bfloat16", "float16"))
    ap.add_argument("--only", default=None,
                    help="restrict to cond:method (resume/debug aid)")
    args = ap.parse_args(argv)
    if args.device == "mps" and args.dtype == "float16":
        raise SystemExit("fp16 injection on MPS failed its validation (D51); "
                         "the D52 sanction, if in force, is bf16 only")

    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=getattr(torch, args.dtype),
        attn_implementation="sdpa").to(args.device).eval()
    model.config.use_cache = False
    arch = describe(model)

    t0 = time.time()
    done = total = 0
    for cond, pattern in CONDITIONS:
        for method, scales, mode in (("todd", SCALES_TODD, "add"),
                                     ("hendel", SCALES_HENDEL, "replace")):
            if args.only and args.only != f"{cond}:{method}":
                continue
            fv_file = pattern.format(m=method)
            z = np.load(fv_file, allow_pickle=False)
            X = z["X"].astype(np.float32)
            assert [int(k) for k in z["ks"]] == KS
            fv_sha = hashlib.sha256(Path(fv_file).read_bytes()).hexdigest()
            for layer in range(arch.n_layers):
                total += 1
                if chunk_path(cond, method, layer, args.dtype,
                              args.device).exists():
                    done += 1
                    continue
                run_chunk(model, tok, arch, cond, method, layer, X, scales,
                          mode, args.dtype, args.device, fv_sha)
                done += 1
    print(f"sweep complete: {done}/{total} chunks present "
          f"({time.time() - t0:.0f}s this session)")


if __name__ == "__main__":
    main()
