"""Rule-based PII detectors (no LLM). Checksums are validated, not just regexes."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
from typing import Protocol, runtime_checkable

# --------------------------------------------------------------------------- checksums


def luhn_valid(digits: str) -> bool:
    """Luhn checksum over a string of digits."""
    if not digits.isdigit() or len(digits) < 2:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def iin_check_digit(first11: str) -> int | None:
    """Check digit for a KZ IIN/BIN body (11 digits); None if no valid digit exists."""
    if len(first11) != 11 or not first11.isdigit():
        return None
    digits = [int(c) for c in first11]
    s = sum(d * w for d, w in zip(digits, range(1, 12), strict=True)) % 11
    if s == 10:
        weights = [3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2]
        s = sum(d * w for d, w in zip(digits, weights, strict=True)) % 11
        if s == 10:
            return None
    return s


def iin_checksum_valid(value: str) -> bool:
    """Checksum of a 12-digit KZ IIN or BIN."""
    if len(value) != 12 or not value.isdigit():
        return False
    return iin_check_digit(value[:11]) == int(value[11])


def _valid_birth_date(yy: int, mm: int, dd: int, century_digit: int) -> bool:
    century = {1: 1800, 2: 1800, 3: 1900, 4: 1900, 5: 2000, 6: 2000}.get(century_digit)
    if century is None:
        return False
    try:
        date(century + yy, mm, dd)
    except ValueError:
        return False
    return True


def is_valid_iin(value: str) -> bool:
    """Individual IIN: YYMMDD + century/sex digit (1-6) + 4 digits + check digit."""
    if not iin_checksum_valid(value):
        return False
    return _valid_birth_date(int(value[0:2]), int(value[2:4]), int(value[4:6]), int(value[6]))


def is_valid_bin(value: str) -> bool:
    """Legal-entity BIN: YYMM + type digit (4-6) + subtype digit (0-3) + 5 digits + check digit."""
    if not iin_checksum_valid(value):
        return False
    mm = int(value[2:4])
    return 1 <= mm <= 12 and value[4] in "456" and value[5] in "0123"


def iban_valid(value: str) -> bool:
    """ISO 13616 mod-97 check for a KZ IBAN (20 chars)."""
    s = value.replace(" ", "").upper()
    if len(s) != 20 or not s.startswith("KZ") or not s.isalnum():
        return False
    rearranged = s[4:] + s[:4]
    num = "".join(str(int(c, 36)) for c in rearranged)
    return int(num) % 97 == 1


# --------------------------------------------------------------------------- interface


@dataclass(frozen=True, slots=True)
class Match:
    start: int
    end: int
    entity: str  # e.g. "IIN"; used in placeholders like <IIN_1>
    value: str


@runtime_checkable
class Detector(Protocol):
    """Pluggable detector. `priority`: lower wins when matches overlap."""

    entity: str
    priority: int

    def find(self, text: str) -> Iterable[Match]: ...


class RegexDetector:
    """Regex + optional validator. The match span is `group` (default: whole match)."""

    def __init__(
        self,
        entity: str,
        pattern: str,
        *,
        priority: int = 50,
        validator: Callable[[str], bool] | None = None,
        group: int | str = 0,
        flags: int = 0,
        normalize: Callable[[str], str] | None = None,
    ) -> None:
        self.entity = entity
        self.priority = priority
        self._re = re.compile(pattern, flags)
        self._validator = validator
        self._group = group
        self._normalize = normalize

    def find(self, text: str) -> Iterable[Match]:
        for m in self._re.finditer(text):
            raw = m.group(self._group)
            if raw is None:
                continue
            candidate = self._normalize(raw) if self._normalize else raw
            if self._validator is not None and not self._validator(candidate):
                continue
            start, end = m.span(self._group)
            yield Match(start, end, self.entity, raw)


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def _card_ok(s: str) -> bool:
    d = _digits(s)
    return 13 <= len(d) <= 19 and d[0] in "23456" and luhn_valid(d)


def _phone_ok(s: str) -> bool:
    d = _digits(s)
    if len(d) != 11 or d[0] not in "78":
        return False
    return d[1] in "3456789"  # KZ/RU area codes never start with 0, 1 or 2


# --------------------------------------------------------------------------- built-ins

_KW = (
    r"(?:паспорт\w*|удостоверени\w*(?:\s+личности)?|жеке\s+куәлік\w*|төлқұжат\w*"
    r"|passport|id\s*card|\bID\b)"
)
_SEP = r"[\s:№#.,\-]{0,6}"


def default_detectors() -> list[Detector]:
    return [
        RegexDetector(
            "IBAN",
            r"(?<![0-9A-Za-z])KZ\d{2}(?: ?[0-9A-Z]{4}){4}(?![0-9A-Za-z])",
            priority=0,
            validator=iban_valid,
            flags=re.IGNORECASE,
            normalize=lambda s: s.upper(),
        ),
        RegexDetector(
            "CARD",
            r"(?<![\d])(?:\d[ \-]?){12,18}\d(?![\d])",
            priority=1,
            validator=_card_ok,
        ),
        RegexDetector(
            "IIN",
            r"(?<!\d)\d{12}(?!\d)",
            priority=2,
            validator=is_valid_iin,
        ),
        RegexDetector(
            "BIN",
            r"(?<!\d)\d{12}(?!\d)",
            priority=3,
            validator=is_valid_bin,
        ),
        RegexDetector(
            "EMAIL",
            r"(?<![\w.+-])[A-Za-z0-9][A-Za-z0-9._%+-]*@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}",
            priority=4,
        ),
        RegexDetector(
            "PHONE",
            r"(?<![\w+])(?:\+7|8|7)[\s\-]*\(?\d{3}\)?[\s\-]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}(?!\d)",
            priority=5,
            validator=_phone_ok,
        ),
        # KZ passport: N + 8 digits
        RegexDetector(
            "PASSPORT",
            r"(?<![\w])N\d{8}(?!\d)",
            priority=6,
        ),
        # Keyworded numbers: KZ ID card (9 digits), RU passport (4 + 6 digits)
        RegexDetector(
            "PASSPORT",
            _KW + _SEP + r"(?:серия\s*)?(?P<num>\d{9}(?!\d)|\d{2}\s?\d{2}[\s№#-]{0,3}\d{6}(?!\d))",
            priority=7,
            group="num",
            flags=re.IGNORECASE,
        ),
    ]


# --------------------------------------------------------------------------- engine


def find_all(text: str, detectors: Iterable[Detector]) -> list[Match]:
    """Run detectors, resolve overlaps (lower priority wins, then longer, then earlier)."""
    found: list[tuple[int, Match]] = []
    for det in detectors:
        for m in det.find(text):
            found.append((det.priority, m))
    found.sort(key=lambda pm: (pm[0], -(pm[1].end - pm[1].start), pm[1].start))
    taken: list[Match] = []
    for _, m in found:
        if all(m.end <= t.start or m.start >= t.end for t in taken):
            taken.append(m)
    taken.sort(key=lambda m: m.start)
    return taken
