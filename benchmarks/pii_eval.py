"""Per-entity precision/recall of the PII detectors on the synthetic labeled dataset.

Run: uv run python benchmarks/pii_eval.py [--markdown]
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests" / "pii"))

from pii_dataset import Sample, generate  # noqa: E402

from gateway.middleware.pii import default_detectors, find_all  # noqa: E402

ENTITIES = ["IIN", "BIN", "PHONE", "CARD", "IBAN", "EMAIL", "PASSPORT"]


def evaluate(samples: list[Sample]) -> dict[str, dict[str, float]]:
    dets = default_detectors()
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    for s in samples:
        pred = {(m.start, m.end, m.entity) for m in find_all(s.text, dets)}
        gold = set(s.spans)
        for x in pred & gold:
            tp[x[2]] += 1
        for x in pred - gold:
            fp[x[2]] += 1
        for x in gold - pred:
            fn[x[2]] += 1
    out: dict[str, dict[str, float]] = {}
    for e in ENTITIES:
        p = tp[e] / (tp[e] + fp[e]) if tp[e] + fp[e] else 1.0
        r = tp[e] / (tp[e] + fn[e]) if tp[e] + fn[e] else 1.0
        out[e] = {"precision": p, "recall": r, "support": tp[e] + fn[e]}
    return out


def main() -> None:
    samples = generate()
    res = evaluate(samples)
    print(f"samples: {len(samples)}")
    if "--markdown" in sys.argv:
        print("| Entity | Precision | Recall | Support |\n|---|---|---|---|")
        for e, m in res.items():
            print(f"| {e} | {m['precision']:.3f} | {m['recall']:.3f} | {int(m['support'])} |")
    else:
        for e, m in res.items():
            print(f"{e:9s} P={m['precision']:.3f} R={m['recall']:.3f} n={int(m['support'])}")


if __name__ == "__main__":
    main()
