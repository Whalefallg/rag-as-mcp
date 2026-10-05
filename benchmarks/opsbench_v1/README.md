# OpsBench v1

OpsBench v1 is a deterministic operational-runbook retrieval benchmark layered on the production retrieval pipeline in this repository. It contains 24 deliberately overlapping runbooks, 80 fixed queries, graded passage-level qrels, stable `[KB_ID: ...]` markers, slice metrics, latency, and agentic cost telemetry.

Generate and validate the committed sources and ingestion PDFs:

```bash
python scripts/generate_opsbench_corpus.py
```

Ingest through the production `IngestionPipeline` and verify Chroma, BM25, and smoke queries:

```bash
python scripts/ingest_opsbench.py --collection opsbench_v1 --config config/opsbench.yaml
```

Run every configured variant:

```bash
python scripts/benchmark_opsbench.py --collection opsbench_v1 --variant all --top-k 10 --config config/opsbench.yaml
```

`hybrid_rerank` and `agentic_rerank` are reported as `SKIPPED` when `rerank.backend` is `none`; they are never labeled as reranking experiments without a real backend. No-answer queries are excluded from ordinary recall/MRR/nDCG aggregates and reported separately.

The current real execution report is in [reports/latest.md](reports/latest.md), with per-query evidence in `reports/latest.json`. No quality threshold is asserted: the report is an empirical baseline, including weak results.
