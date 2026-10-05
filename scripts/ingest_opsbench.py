#!/usr/bin/env python3
"""Generate and ingest OpsBench through the production IngestionPipeline."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(); p.add_argument("--collection",default="opsbench_v1"); p.add_argument("--config",default="config/settings.yaml"); p.add_argument("--force",action="store_true"); args=p.parse_args()
    subprocess.run([sys.executable,str(ROOT/"scripts/generate_opsbench_corpus.py")],check=True,cwd=ROOT)
    from src.core.settings import load_settings
    from src.ingestion.pipeline import IngestionPipeline
    from src.libs.vector_store.vector_store_factory import create_vector_store
    from src.ingestion.storage.bm25_indexer import BM25Indexer
    from src.core.query_engine.hybrid_search import HybridSearch
    settings=load_settings(args.config); pipeline=IngestionPipeline(settings,collection=args.collection)
    results=[]
    for pdf in sorted((ROOT/"data/documents/opsbench_v1").glob("*.pdf")):
        print(f"ingest {pdf.name}",flush=True); results.append(pipeline.run(str(pdf),force=args.force))
    store=create_vector_store(settings); collections=store.list_collections()
    if args.collection not in collections: raise SystemExit(f"collection missing: {args.collection}")
    dense_count=store.get_or_create_collection(args.collection).count()
    bm25=BM25Indexer(); bm25_hits=bm25.query({"redis":1.0},top_k=1,collection=args.collection)
    search=HybridSearch(settings)
    smoke={q:len(search.search(q,top_k=3,collection=args.collection)) for q in ("Redis OOM maxmemory rejected writes","Postgres connection pool acquisition timeout","Kubernetes OOMKilled exit code 137")}
    if dense_count < 24 or not bm25_hits or not all(smoke.values()): raise SystemExit(f"ingest verification failed dense={dense_count} bm25={len(bm25_hits)} smoke={smoke}")
    print(json.dumps({"collection":args.collection,"collections":collections,"documents":24,"dense_chunks":dense_count,"bm25_present":True,"smoke":smoke,"ingested":sum(not r.get('skipped') for r in results),"skipped":sum(bool(r.get('skipped')) for r in results)},indent=2))
if __name__=="__main__": main()
