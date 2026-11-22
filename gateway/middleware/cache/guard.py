"""Lexical guard for semantic hits.

Embeddings place "convert 100 USD" and "convert 200 USD" almost on top of each other.
A semantic hit is accepted only if the *signature* (numbers + negation) of the query equals
the signature of the cached text. Cheap, language-agnostic for digits, RU/KK/EN for negation.
"""

from __future__ import annotations

import re

_NUM = re.compile(r"\d+(?:[.,]\d+)?")
_WORD = re.compile(r"[\w'’-]+", re.UNICODE)
_NEG = frozenset(
    {
        # ru
        "не", "нет", "без", "ни", "нельзя", "невозможно", "никогда",
        # kk
        "емес", "жоқ", "жок", "болмайды", "ешқашан", "сыз", "сіз",
        # en
        "not", "no", "never", "without", "cannot", "can't", "cant", "don't", "dont", "isn't",
    }
)  # fmt: skip


def signature(text: str) -> str:
    """Stable string: sorted numbers + negation flag."""
    nums = sorted(n.replace(",", ".") for n in _NUM.findall(text))
    words = {w.lower() for w in _WORD.findall(text)}
    neg = bool(words & _NEG) or any(w.endswith("n't") for w in words)
    return f"{'|'.join(nums)}#{int(neg)}"
