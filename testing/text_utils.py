"""Lightweight text utilities shared across evaluation code.

No heavy dependencies (standard library only) so these helpers can be reused and
unit tested without importing the model stack.
"""

import difflib
import re
from typing import Dict, List


def diff_sentences(old: str, new: str) -> Dict[str, List[str]]:
    """Sentence-level diff between two texts.

    Returns the sentences removed from ``old`` and the sentences added in
    ``new`` as ``{"removed": [...], "added": [...]}``.
    """
    split = lambda t: [s.strip() for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s]
    o, n = split(old), split(new)
    sm = difflib.SequenceMatcher(a=o, b=n)
    added, removed = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "delete":
            removed.extend(o[i1:i2])
        elif tag == "insert":
            added.extend(n[j1:j2])
        elif tag == "replace":
            removed.extend(o[i1:i2])
            added.extend(n[j1:j2])
    return {"removed": removed, "added": added}
