# ruff: noqa: E501
"""Semantic-cache threshold sweep + embedding latency on CPU.

Protocol
- Index = first text of every paraphrase group (what is "already cached").
- Each other text of a group is a query: a *correct hit* if the top-1 index entry is its own
  group and sim >= t; a *cross-group false hit* if top-1 is another group and sim >= t.
- Each hard-negative pair (a, b): querying b against a is a *negative false hit* if sim >= t.
- Default threshold = lowest t whose cross-group false-hit rate (+guard) <= 0.5%
  (hit rate only falls as t grows, so the lowest such t maximizes hits). Hard-negative
  pairs are adversarial minimal pairs; their false-hit rate is reported, not optimized.

Run: uv run python benchmarks/cache_eval.py   (downloads the model on first run)
Writes docs/cache_eval_results.md, docs/cache_eval.json, docs/cache_threshold_curve.svg
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from gateway.middleware.cache.embedder import E5OnnxEmbedder
from gateway.middleware.cache.guard import signature

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "benchmarks" / "cache"
DOCS = ROOT / "docs"
TARGET_NEG_FALSE_HIT = 0.02  # reported only: no threshold reaches it with a usable hit rate
TARGET_CROSS_FALSE_HIT = 0.005  # selection criterion
THRESHOLDS = [round(0.80 + 0.005 * i, 3) for i in range(0, 40)]  # 0.80 .. 0.995


def load(name: str) -> list[dict]:  # type: ignore[type-arg]
    return [json.loads(line) for line in (DATA / name).read_text().splitlines() if line.strip()]


def lang_of(g: dict) -> str:  # type: ignore[type-arg]
    return "mixed" if len(g["langs"]) > 1 else g["langs"][0]


def main() -> None:
    emb = E5OnnxEmbedder()
    groups, negs = load("paraphrases.jsonl"), load("hard_negatives.jsonl")

    # --- embeddings
    first = emb.embed([g["texts"][0] for g in groups])
    q_texts = [(gi, t) for gi, g in enumerate(groups) for t in g["texts"][1:]]
    q_vecs = emb.embed([t for _, t in q_texts])
    na = emb.embed([n["a"] for n in negs])
    nb = emb.embed([n["b"] for n in negs])

    sims_raw = q_vecs @ first.T  # (Q, G)
    first_sig = np.array([signature(g["texts"][0]) for g in groups])
    q_sig = np.array([signature(t) for _, t in q_texts])
    sims_guard = np.where(q_sig[:, None] == first_sig[None, :], sims_raw, -1.0)
    neg_guard_ok = np.array([signature(n["a"]) == signature(n["b"]) for n in negs])
    sims = sims_raw
    top, top_sim = sims.argmax(axis=1), sims.max(axis=1)
    gtop, gtop_sim = sims_guard.argmax(axis=1), sims_guard.max(axis=1)
    own = np.array([gi for gi, _ in q_texts])
    q_lang = [lang_of(groups[gi]) for gi, _ in q_texts]
    neg_sim = (na * nb).sum(axis=1)
    para_pos = np.array([float(sims[i, own[i]]) for i in range(len(q_texts))])

    rows = []
    for t in THRESHOLDS:
        fired = top_sim >= t
        correct = fired & (top == own)
        cross = fired & (top != own)
        rows.append(
            {
                "threshold": t,
                "hit_rate": float(correct.mean()),
                "cross_group_false_hit": float(cross.mean()),
                "negative_false_hit": float((neg_sim >= t).mean()),
                "guard_hit_rate": float(((gtop_sim >= t) & (gtop == own)).mean()),
                "guard_cross_group_false_hit": float(((gtop_sim >= t) & (gtop != own)).mean()),
                "guard_negative_false_hit": float(((neg_sim >= t) & neg_guard_ok).mean()),
            }
        )
    ok = [r for r in rows if r["guard_cross_group_false_hit"] <= TARGET_CROSS_FALSE_HIT]
    chosen = ok[0] if ok else rows[-1]
    t_star = chosen["threshold"]

    per_lang = {}
    for lg in sorted(set(q_lang)):
        idx = [i for i, x in enumerate(q_lang) if x == lg]
        fired = gtop_sim[idx] >= t_star
        per_lang[lg] = float((fired & (gtop[idx] == own[idx])).mean())
    neg_lang = {}
    for lg in sorted({n["lang"] for n in negs}):
        idx = [i for i, n in enumerate(negs) if n["lang"] == lg]
        neg_lang[lg] = float(((neg_sim[idx] >= t_star) & neg_guard_ok[idx]).mean())

    # --- latency (single text, as on the request path) and small batch
    texts = [g["texts"][1] for g in groups]
    for t in texts[:5]:
        emb.embed([t])  # warm-up
    single = []
    for t in texts[:100]:
        s = time.perf_counter()
        emb.embed([t])
        single.append((time.perf_counter() - s) * 1000)
    batch8 = []
    for i in range(0, 80, 8):
        s = time.perf_counter()
        emb.embed(texts[i : i + 8])
        batch8.append((time.perf_counter() - s) * 1000)
    q = statistics.quantiles(single, n=100)
    latency = {
        "single_ms_p50": statistics.median(single),
        "single_ms_p95": q[94],
        "batch8_ms_mean": statistics.mean(batch8),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "onnx_threads": 2,
        "model": "intfloat/multilingual-e5-small (fp32 ONNX)",
    }

    summary = {
        "n_groups": len(groups),
        "n_paraphrase_queries": len(q_texts),
        "n_negative_pairs": len(negs),
        "target_cross_group_false_hit": TARGET_CROSS_FALSE_HIT,
        "target_negative_false_hit_unmet": TARGET_NEG_FALSE_HIT,
        "chosen_threshold": t_star,
        "chosen": chosen,
        "hit_rate_by_lang_at_chosen": per_lang,
        "negative_false_hit_by_lang_at_chosen": neg_lang,
        "paraphrase_sim_mean": float(para_pos.mean()),
        "negative_sim_mean": float(neg_sim.mean()),
        "latency": latency,
        "curve": rows,
    }
    DOCS.mkdir(exist_ok=True)
    (DOCS / "cache_eval.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (DOCS / "cache_threshold_curve.svg").write_text(svg(rows, t_star))
    md = [
        "| threshold | hit (emb only) | hard-neg false hit (emb only) | hit (+guard) "
        "| other-group false hit (+guard) | hard-neg false hit (+guard) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        if round(r["threshold"] * 200) % 2 == 0:  # every 0.01
            mark = " **←default**" if r["threshold"] == t_star else ""
            md.append(
                f"| {r['threshold']:.3f}{mark} | {r['hit_rate']:.3f} | {r['negative_false_hit']:.3f} "
                f"| {r['guard_hit_rate']:.3f} | {r['guard_cross_group_false_hit']:.3f} "
                f"| {r['guard_negative_false_hit']:.3f} |"
            )
    (DOCS / "cache_eval_results.md").write_text("\n".join(md) + "\n")
    print(
        json.dumps({k: v for k, v in summary.items() if k != "curve"}, indent=2, ensure_ascii=False)
    )


def svg(rows: list[dict], t_star: float) -> str:  # type: ignore[type-arg]
    w, h, ml, mb, mt, mr = 640, 380, 56, 52, 24, 20
    pw, ph = w - ml - mr, h - mt - mb
    t0, t1 = rows[0]["threshold"], rows[-1]["threshold"]

    def x(t: float) -> float:
        return ml + (t - t0) / (t1 - t0) * pw

    def y(v: float) -> float:
        return mt + (1 - v) * ph

    series = [
        ("hit_rate", "#2a7ae2", "correct hit (embedding only)"),
        ("guard_hit_rate", "#1c4f9c", "correct hit (+ lexical guard)"),
        ("negative_false_hit", "#d6336c", "hard-negative false hit (embedding only)"),
        ("guard_negative_false_hit", "#8a1c4a", "hard-negative false hit (+ guard)"),
    ]
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="sans-serif" font-size="12">',
        f'<rect width="{w}" height="{h}" fill="#fff"/>',
    ]
    for i in range(6):
        v = i / 5
        out.append(
            f'<line x1="{ml}" x2="{w - mr}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#e5e5e5"/>'
        )
        out.append(
            f'<text x="{ml - 8}" y="{y(v) + 4:.1f}" text-anchor="end" fill="#444">{v:.1f}</text>'
        )
    for i in range(0, 40, 5):
        t = rows[i]["threshold"]
        out.append(
            f'<text x="{x(t):.1f}" y="{h - mb + 18}" text-anchor="middle" fill="#444">{t:.2f}</text>'
        )
    for key, color, _ in series:
        pts = " ".join(f"{x(r['threshold']):.1f},{y(r[key]):.1f}" for r in rows)
        out.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{pts}"/>')
    out.append(
        f'<line x1="{x(t_star):.1f}" x2="{x(t_star):.1f}" y1="{mt}" y2="{mt + ph}" stroke="#333" stroke-dasharray="5 4"/>'
    )
    out.append(
        f'<text x="{x(t_star) + 6:.1f}" y="{mt + 14}" fill="#333">default {t_star:.3f}</text>'
    )
    out.append(
        f'<text x="{ml + pw / 2}" y="{h - 10}" text-anchor="middle" fill="#222">cosine similarity threshold</text>'
    )
    for i, (_, color, label) in enumerate(series):
        yy = mt + 40 + i * 18
        out.append(f'<rect x="{ml + 12}" y="{yy - 9}" width="12" height="4" fill="{color}"/>')
        out.append(f'<text x="{ml + 30}" y="{yy - 3}" fill="#222">{label}</text>')
    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    main()
