#!/usr/bin/env python3
"""Run OpsBench v1 against real production retrievers and serialize evidence."""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); BENCH=ROOT/"benchmarks/opsbench_v1"
from src.observability.evaluation.opsbench import corpus_hash, extract_kb_ids, load_dataset, score_query, score_result, validate_dataset
from src.observability.evaluation.retrieval_metrics import aggregate, percentile

NAMES=["bm25","dense","hybrid","hybrid_rerank","agentic","agentic_rerank"]
def git_commit():
    try:return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    except Exception:return None
def metadata(settings,args,query_count):
    return {"dataset_version":"1.0.0","corpus_hash":corpus_hash(BENCH),"query_count":query_count,"git_commit":git_commit(),"timestamp":datetime.now(timezone.utc).isoformat(),"embedding_provider":settings.embedding.provider,"embedding_model":settings.embedding.model,"reranker_backend":settings.rerank.backend,"reranker_model":settings.rerank.model or None,"agentic_configuration":{"max_iterations":settings.agentic.max_iterations,"planner_use_llm":settings.agentic.planner.use_llm,"grader_use_llm":settings.agentic.grader.use_llm,"rewriter_use_llm":settings.agentic.rewriter.use_llm},"collection":args.collection,"top_k":args.top_k}
def run_variant(name,settings,queries,qrels,args):
    from src.observability.evaluation.retrieval_variant import create_variant
    adapter=create_variant(name,settings)
    if adapter.skip_reason:return {"status":"SKIPPED","reason":adapter.skip_reason}
    for q in queries[:min(5,len(queries))]: adapter.search(q["query"],args.top_k,args.collection)
    rows=[]; agents=[]
    for q in queries:
        start=time.perf_counter(); output=adapter.search(q["query"],args.top_k,args.collection); latency=(time.perf_counter()-start)*1000
        if name.startswith("agentic"): results,agent=output; agents.append(agent)
        else: results=output
        details=[]; ranked=[]; rels=qrels.get(q["id"],{})
        for rank,result in enumerate(results,1):
            kids=extract_kb_ids(result.text); grade=score_result(kids,rels)
            details.append({"rank":rank,"chunk_id":result.chunk_id,"source":result.source,"score":result.score,"kb_ids":kids,"relevance":grade})
            ranked.append(kids)
        metrics=score_query(ranked,rels) if q["answerable"] else {}
        rows.append({"id":q["id"],"category":q["category"],"answerable":q["answerable"],"latency_ms":latency,"metrics":metrics,"results":details})
    answerable=[{**r["metrics"]} for r in rows if r["answerable"]]
    overall=aggregate(answerable); lat=[r["latency_ms"] for r in rows]
    summary={**overall,"latency":{"mean_ms":mean(lat),"p50_ms":percentile(lat,.5),"p95_ms":percentile(lat,.95)}}
    summary["slices"]={cat:aggregate([r["metrics"] for r in rows if r["answerable"] and r["category"]==cat]) for cat in ("lexical","semantic","hard_negative","multi_hop","noisy")}
    summary["no_answer"]={"query_count":sum(not r["answerable"] for r in rows),"abstention":"N/A"}
    if agents:
        term=Counter(a.termination_reason.value for a in agents); rejected=sum(a.termination_reason.value in {"no_results","max_iterations"} for a,q in zip(agents,queries) if not q["answerable"]); no_count=sum(not q["answerable"] for q in queries); false=sum(a.termination_reason.value in {"no_results","max_iterations"} for a,q in zip(agents,queries) if q["answerable"])
        summary["agentic"]={"average_iterations":mean(a.iteration_count for a in agents),"average_retrieval_calls":mean(a.retrieval_calls for a in agents),"rewrite_rate":mean(a.rewrite_count>0 for a in agents),"degradation_rate":mean(a.degraded for a in agents),"average_llm_calls":mean(a.llm_calls for a in agents),"termination_reason_distribution":dict(term),"no_answer_rejection_rate":rejected/no_count if no_count else None,"false_abstention_rate":false/sum(q["answerable"] for q in queries)}
    return {"status":"COMPLETED","summary":summary,"queries":rows}
def markdown(report):
    lines=["# OpsBench v1 Results","",f"Generated: {report['metadata']['timestamp']}","", "| Variant | Hit@5 | Recall@5 | MRR@10 | nDCG@10 | p50 ms | p95 ms | Calls/query |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name,data in report["variants"].items():
        if data["status"]!="COMPLETED": lines.append(f"| {name} | SKIPPED | — | — | — | — | — | — |"); continue
        s=data["summary"]; calls=s.get("agentic",{}).get("average_retrieval_calls",1)
        lines.append(f"| {name} | {s['hit_at_5']:.3f} | {s['recall_at_5']:.3f} | {s['mrr_at_10']:.3f} | {s['ndcg_at_10']:.3f} | {s['latency']['p50_ms']:.1f} | {s['latency']['p95_ms']:.1f} | {calls:.2f} |")
    lines += ["","## nDCG@10 slices","","| Variant | Overall | Lexical | Semantic | Hard Negative | Multi-hop | Noisy |","|---|---:|---:|---:|---:|---:|---:|"]
    for name,data in report["variants"].items():
        if data["status"]!="COMPLETED":continue
        s=data["summary"]; sl=s["slices"]; lines.append(f"| {name} | {s['ndcg_at_10']:.3f} | {sl['lexical']['ndcg_at_10']:.3f} | {sl['semantic']['ndcg_at_10']:.3f} | {sl['hard_negative']['ndcg_at_10']:.3f} | {sl['multi_hop']['ndcg_at_10']:.3f} | {sl['noisy']['ndcg_at_10']:.3f} |")
    return "\n".join(lines)+"\n"
def main():
    p=argparse.ArgumentParser();p.add_argument("--collection",default="opsbench_v1");p.add_argument("--variant",choices=NAMES+["all"],default="all");p.add_argument("--category",choices=["lexical","semantic","hard_negative","multi_hop","noisy","no_answer"]);p.add_argument("--top-k",type=int,default=10);p.add_argument("--config",default="config/settings.yaml");args=p.parse_args()
    validate_dataset(BENCH); _,queries,qrels=load_dataset(BENCH)
    if args.category:queries=[q for q in queries if q["category"]==args.category]
    from src.core.settings import load_settings
    settings=load_settings(args.config); names=NAMES if args.variant=="all" else [args.variant]
    report={"metadata":metadata(settings,args,len(queries)),"variants":{}}
    for name in names: print(f"benchmark {name}",flush=True); report["variants"][name]=run_variant(name,settings,queries,qrels,args)
    out=BENCH/"reports";out.mkdir(exist_ok=True);(out/"latest.json").write_text(json.dumps(report,indent=2,ensure_ascii=False,default=str)+"\n");(out/"latest.md").write_text(markdown(report));print(markdown(report))
if __name__=="__main__":main()
