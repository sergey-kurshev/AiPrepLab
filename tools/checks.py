"""A very small assertion harness shared by every exercise notebook.

Notebooks are a bad place for a real test framework: you want a check to print a
result and keep going, not to abort the kernel.  So `check` records outcomes and
`report` prints a summary at the end of a section.

    from checks import check, report, close

    check("shape", out.shape == (2, 3, 4))
    check("matches reference", close(out, ref))
    report()
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_RESULTS: list[tuple[str, bool, str]] = []

PASS = "PASS"
FAIL = "FAIL"


def bootstrap() -> None:
    """Make `tools/` importable from a notebook in any module folder.

    Notebooks call this implicitly by importing this module, but a learner who
    moved the file can call it directly.
    """
    # Windows consoles default to cp1252, which cannot print the Greek letters and
    # arrows these exercises use. Notebooks are already UTF-8; this is for `python
    # solution.py` in a terminal.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    here = Path.cwd()
    for candidate in (here, *here.parents):
        tools = candidate / "tools"
        if tools.is_dir() and str(tools) not in sys.path:
            sys.path.insert(0, str(tools))
            return


def close(a, b, atol: float = 1e-5, rtol: float = 1e-4) -> bool:
    """Tolerant comparison that works for torch tensors, numpy arrays and floats."""
    try:
        import torch

        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            a_t = a if isinstance(a, torch.Tensor) else torch.as_tensor(a)
            b_t = b if isinstance(b, torch.Tensor) else torch.as_tensor(b)
            if a_t.shape != b_t.shape:
                return False
            return bool(torch.allclose(a_t.float(), b_t.float(), atol=atol, rtol=rtol))
    except ImportError:
        pass
    try:
        import numpy as np

        if hasattr(a, "shape") or hasattr(b, "shape"):
            a_n, b_n = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
            if a_n.shape != b_n.shape:
                return False
            return bool(np.allclose(a_n, b_n, atol=atol, rtol=rtol))
    except ImportError:
        pass
    return math.isclose(float(a), float(b), abs_tol=atol, rel_tol=rtol)


def check(name: str, condition, note: str = "") -> bool:
    """Record and print one assertion.  Never raises."""
    ok = bool(condition)
    _RESULTS.append((name, ok, note))
    mark = "[ok]  " if ok else "[FAIL]"
    suffix = f"  -- {note}" if note else ""
    print(f"{mark} {name}{suffix}")
    return ok


def report(reset: bool = True) -> bool:
    """Print a summary of checks recorded so far.  Returns True if all passed."""
    total = len(_RESULTS)
    failed = [n for n, ok, _ in _RESULTS if not ok]
    if not total:
        print("no checks recorded")
        return True
    if failed:
        print(f"\n{total - len(failed)}/{total} passed.  Still failing: {', '.join(failed)}")
    else:
        print(f"\nAll {total} checks passed.")
    ok = not failed
    if reset:
        _RESULTS.clear()
    return ok


bootstrap()
