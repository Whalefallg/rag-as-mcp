"""OpsBench dataset loading, validation, stable-ID extraction, and scoring."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from .retrieval_metrics import hit_at, recall_at, mrr_at, ndcg_at

KB_ID_RE = re.compile(r"\[KB_ID:\s*([^\]\s]+)\s*\]")
EXPECTED = {"lexical": 16, "semantic": 16, "hard_negative": 16, "multi_hop": 12, "noisy": 12, "no_answer": 8}


def extract_kb_ids(text: str) -> list[str]:
    return list(dict.fromkeys(KB_ID_RE.findall(text or "")))


def load_dataset(root: Path) -> tuple[dict, list[dict], dict]:
    manifest = json.loads((root / "corpus_manifest.json").read_text())
    queries = [json.loads(line) for line in (root / "queries.jsonl").read_text().splitlines() if line.strip()]
    qrels = json.loads((root / "qrels.json").read_text())
    return manifest, queries, qrels


def validate_dataset(root: Path) -> dict:
    manifest, queries, qrels = load_dataset(root)
    errors: list[str] = []
    docs = manifest.get("documents", [])
    if len(docs) != 24: errors.append(f"expected 24 documents, got {len(docs)}")
    if len(queries) != 80: errors.append(f"expected 80 queries, got {len(queries)}")
    counts = Counter(q.get("category") for q in queries)
    if dict(counts) != EXPECTED: errors.append(f"category distribution {dict(counts)} != {EXPECTED}")
    for label, values in (("document", [d.get("document_id") for d in docs]), ("query", [q.get("id") for q in queries])):
        if len(values) != len(set(values)): errors.append(f"duplicate {label} IDs")
    kb_ids = [kid for d in docs for kid in d.get("kb_ids", [])]
    if len(kb_ids) != len(set(kb_ids)): errors.append("duplicate KB IDs")
    known, query_ids = set(kb_ids), {q["id"] for q in queries}
    unknown_queries = set(qrels) - query_ids
    if unknown_queries: errors.append(f"qrels contain unknown queries: {sorted(unknown_queries)}")
    for query in queries:
        rels = qrels.get(query["id"], {})
        unknown = set(rels) - known
        if unknown: errors.append(f"{query['id']} refers to unknown KB IDs: {sorted(unknown)}")
        positives = [grade for grade in rels.values() if grade >= 2]
        if query["answerable"] and not positives: errors.append(f"{query['id']} has no positive qrel")
        if not query["answerable"] and positives: errors.append(f"{query['id']} no-answer has positive qrel")
    if errors: raise ValueError("OpsBench validation failed:\n- " + "\n- ".join(errors))
    return {"documents": len(docs), "queries": len(queries), "categories": dict(counts), "kb_ids": len(kb_ids)}


def corpus_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "corpus").glob("*.yaml")):
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def score_result(kb_ids: list[str], qrels: dict[str, int]) -> int:
    return max((qrels.get(kid, 0) for kid in kb_ids), default=0)


def score_query(ranked_chunks: list[list[str]], qrels: dict[str, int]) -> dict[str, float]:
    """Score passage ranks while crediting every stable ID contained by a chunk."""
    relevant = {kid for kid, grade in qrels.items() if grade >= 2}
    grades = [max((qrels.get(kid, 0) for kid in kids), default=0) for kids in ranked_chunks]
    def hit(k): return float(any(grade >= 2 for grade in grades[:k]))
    def recall(k):
        found = {kid for kids in ranked_chunks[:k] for kid in kids} & relevant
        return len(found) / len(relevant) if relevant else 0.0
    reciprocal = next((1.0/rank for rank,grade in enumerate(grades[:10],1) if grade>=2),0.0)
    # Synthetic keys retain passage ranks while reusing the tested graded nDCG implementation.
    keys=[f"rank-{i}" for i in range(len(grades))]; ranked_qrels={key:grade for key,grade in zip(keys,grades)}
    ideal_qrels={f"ideal-{i}":grade for i,grade in enumerate(qrels.values())}
    import math
    dcg=sum((2**grade-1)/math.log2(rank+1) for rank,grade in enumerate(grades[:10],1))
    ideal=sum((2**grade-1)/math.log2(rank+1) for rank,grade in enumerate(sorted(qrels.values(),reverse=True)[:10],1))
    return {"hit_at_1":hit(1),"hit_at_5":hit(5),"hit_at_10":hit(10),"recall_at_5":recall(5),"recall_at_10":recall(10),"mrr_at_10":reciprocal,"ndcg_at_10":dcg/ideal if ideal else 0.0}
