"""Guarded result writing (docs/decisions.md D45/D46). Pure stdlib.

Two defenses against the fixed-filename clobber class D45 documents:
`input_stamp` gives writers whose content depends on which inputs they are
pointed at a filename that cannot collide across invocations, and
`write_guarded` refuses to overwrite differing content at any path.
Identical regeneration is always allowed — a deterministic re-run that
produces byte-identical output is not an overwrite.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def input_stamp(names) -> str:
    """8-hex stamp of the sorted input names, for input-dependent writers."""
    joined = "|".join(sorted(str(n) for n in names))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:8]


def write_guarded(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") == text:
            print(f"{path} already exists with identical content")
            return
        raise SystemExit(
            f"refusing to overwrite {path} with differing content; move the "
            "existing artifact deliberately first (CLAUDE.md: results are "
            "never overwritten; docs/decisions.md D45)"
        )
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {path}")
