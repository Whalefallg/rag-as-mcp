"""Dependency-free, graded retrieval metrics used by OpsBench."""
from __future__ import annotations

import math
from statistics import mean


def _relevant(qrels: dict[str, int]) -> set[str]:
    return {key for key, grade in qrels.items() if grade >= 2}


def hit_at(results: list[str], qrels: dict[str, int], k: int) -> float:
    return float(bool(set(results[:k]) & _relevant(qrels)))


def recall_at(results: list[str], qrels: dict[str, int], k: int) -> float:
    relevant = _relevant(qrels)
    return len(set(results[:k]) & relevant) / len(relevant) if relevant else 0.0


def mrr_at(results: list[str], qrels: dict[str, int], k: int = 10) -> float:
    relevant = _relevant(qrels)
    for rank, key in enumerate(results[:k], 1):
        if key in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at(results: list[str], qrels: dict[str, int], k: int = 10) -> float:
    def dcg(grades: list[int]) -> float:
        return sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, 1))
    actual = dcg([qrels.get(key, 0) for key in results[:k]])
    ideal = dcg(sorted(qrels.values(), reverse=True)[:k])
    return actual / ideal if ideal else 0.0


def percentile(values: list[float], quantile: float) -> float:
    """Linear interpolation (R-7 / numpy default) over sorted observations."""
    if not values:
        return 0.0
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def aggregate(rows: list[dict]) -> dict[str, float]:
    keys = ("hit_at_1", "hit_at_5", "hit_at_10", "recall_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10")
    return {key: mean(row[key] for row in rows) if rows else 0.0 for key in keys}
