#!/usr/bin/env python3
"""Deterministically materialize the committed OpsBench v1 corpus and PDFs."""
from __future__ import annotations

import json, sys
from pathlib import Path
import fitz, yaml

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "opsbench_v1"
OUT = ROOT / "data" / "documents" / "opsbench_v1"
SECTIONS = ("symptoms", "signals", "diagnosis", "mitigation", "escalation", "common_confusions")
DOCUMENT_ID_ALIASES = {
    "postgres-pool-exhaustion": "postgres-connection-pool-runbook",
    "lambda-timeout": "lambda-execution-timeout-runbook",
}

# slug, title, discriminating signal, diagnosis, mitigation, common confusion
CASES = [
("redis-oom","Redis OOM and maxmemory exhaustion","OOM command errors; used_memory reaches maxmemory; writes rejected","Compare used_memory, maxmemory, allocator fragmentation, and largest keys; inspect OOM errors","Remove or expire safe large keys, raise maxmemory only with headroom, then correct retention","Evictions may coexist, but OOM is identified by rejected allocations and oom error counters"),
("redis-latency","Redis latency degradation","p99 command latency rises without sustained memory rejection","Run LATENCY DOCTOR and SLOWLOG; correlate fork, blocked clients, hot keys, and network RTT","Remove blocking commands, shard hot keys, tune persistence, and cap client concurrency","High memory and evictions can add latency, but latency alone does not prove exhaustion"),
("redis-eviction","Redis eviction pressure","evicted_keys rises continuously as memory approaches maxmemory","Inspect maxmemory-policy, evicted_keys rate, TTL coverage, hit ratio, and volatile key population","Fix TTLs, choose an intentional eviction policy, reduce cache footprint, or add capacity","OOM and latency share high memory; a rising eviction counter is the decisive signal"),
("postgres-pool-exhaustion","PostgreSQL connection pool exhaustion","pool waiters and acquisition timeouts rise while database connections approach the pool cap","Compare active, idle, waiting, max pool size, long transactions, and connection leaks","Release leaked connections, bound request concurrency, tune pool sizing, and shorten transactions","Database latency can create waiters but pool acquisition timeout is distinct from lock deadlock"),
("postgres-deadlock","PostgreSQL transaction deadlock","SQLSTATE 40P01 and deadlock detected errors appear","Inspect PostgreSQL deadlock logs, pg_locks wait graph, transaction ordering, and lock scope","Retry aborted transactions with jitter and enforce consistent lock ordering","Pool exhaustion has waiting sessions but does not emit SQLSTATE 40P01"),
("postgres-replica-lag","PostgreSQL replica lag","replica replay delay grows and reads are stale","Compare pg_last_wal_receive_lsn and replay LSN; inspect apply rate, IO, long queries, and WAL volume","Route consistency-sensitive reads to primary, remove replica bottlenecks, and add capacity","Primary query latency is not replica lag unless WAL receive or replay distance grows"),
("kubernetes-oomkilled","Kubernetes OOMKilled containers","pod terminates with reason OOMKilled and exit code 137","Inspect last termination state, memory limit, working set, events, and heap growth","Raise a justified memory limit, fix leaks, reduce concurrency, and set realistic requests","CrashLoopBackOff describes restart behavior; exit code 137 identifies memory termination"),
("kubernetes-crashloop","Kubernetes CrashLoopBackOff","restart backoff increases after repeated application exits","Inspect previous container logs, exit code, probes, command, configuration, and mounted secrets","Fix the failing startup dependency or configuration; adjust probes only when justified","OOMKilled may cause a crash loop, but many crash loops have non-memory exit codes"),
("kubernetes-cpu-throttling","Kubernetes CPU throttling","latency rises with container_cpu_cfs_throttled_seconds_total","Compare CPU quota, throttled periods, runnable load, limits, and node contention","Remove an overly tight CPU limit, optimize hot work, or scale replicas","High CPU utilization without CFS throttling requires a different capacity diagnosis"),
("payment-gateway-timeout","Payment gateway timeout","gateway calls exceed deadline and return timeout or HTTP 504","Trace upstream connect, TLS, first-byte and total time; compare provider status","Use bounded retries only for idempotent requests, circuit break, and fail safely","Rate limiting emits 429 and Retry-After; timeout is deadline exhaustion"),
("payment-rate-limit","Payment provider rate limiting","HTTP 429 and Retry-After responses increase","Inspect provider quota, Retry-After, tenant burst, retry amplification, and idempotency keys","Honor Retry-After with jitter, queue work, reduce bursts, and request quota","Timeout retries can amplify load, but 429 is the discriminating signal"),
("payment-webhook-failure","Payment webhook delivery failure","signed webhook deliveries fail or are not acknowledged with 2xx","Verify signature timestamp, endpoint logs, delivery attempts, DNS, TLS, and processing deadline","Acknowledge quickly, enqueue idempotently, repair signature validation, and replay events","Checkout 5xx is a synchronous customer path; webhooks are asynchronous callbacks"),
("checkout-5xx","Checkout HTTP 5xx errors","checkout endpoint error rate rises with HTTP 500 or 503","Slice errors by route, exception, dependency, instance, and release; inspect traces","Rollback a bad release, shed load, or isolate the failing dependency","Latency alone is not a 5xx incident; confirm server error status"),
("checkout-latency","Checkout latency regression","checkout p95 and p99 rise while success rate may remain normal","Break down traces by database, payment, inventory, cache, and queue time","Remove the slow dependency, reduce fanout, cache safe reads, or scale bottlenecks","A deployment can cause latency, but timing correlation must be supported by traces"),
("checkout-deployment-regression","Checkout deployment regression","errors or latency begin immediately after a checkout release or flag change","Compare canary and prior version, release timestamp, config diff, and feature flags","Rollback or disable the implicated flag, then bisect the change","Coincidental dependency failure can overlap a deploy; require version-specific evidence"),
("lambda-timeout","Lambda execution timeout","invocations reach configured duration and end Task timed out","Inspect duration near timeout, downstream spans, remaining time, memory, and retries","Shorten blocking work, tune client deadlines, increase timeout deliberately, or use async workflow","Cold starts add init duration but do not always consume the full timeout"),
("lambda-cold-start","Lambda cold start latency","Init Duration and first-invocation latency rise after idle periods or deployments","Compare Init Duration, runtime initialization, package size, VPC attach, and warm invocations","Reduce package and initialization work, provision concurrency, or avoid unnecessary VPC setup","Steady-state timeouts are not cold starts when Init Duration is absent"),
("lambda-concurrency-throttling","Lambda concurrency throttling","Throttles rise as concurrent executions reach reserved or account concurrency","Inspect ConcurrentExecutions, Throttles, reserved concurrency, event source backlog, and account quota","Raise or rebalance concurrency, smooth producers, and configure event-source backpressure","Provider API rate limits may also say throttled; confirm Lambda concurrency metrics"),
("kafka-consumer-lag","Kafka consumer lag","consumer group lag grows while production continues","Inspect per-partition lag, consume rate, processing latency, assignments, and poison messages","Scale consumers up to partition count, fix slow processing, and handle poison messages","A rebalance can cause transient lag; persistent growth after stability is consumer lag"),
("kafka-rebalance-storm","Kafka rebalance storm","consumer group repeatedly revokes and reassigns partitions","Inspect group generation changes, heartbeats, session timeout, poll interval, and member churn","Fix slow polls or unstable members, use cooperative assignment, and tune timeouts cautiously","Lag is a symptom of rebalances but does not prove member churn"),
("kafka-under-replicated","Kafka under-replicated partitions","UnderReplicatedPartitions rises and ISR shrinks","Inspect ISR, offline replicas, broker disk/network, controller logs, and replica fetch lag","Restore broker capacity or connectivity and throttle recovery to protect availability","Consumer lag concerns application progress; ISR shrink concerns broker replication"),
("gateway-502","Gateway HTTP 502 bad gateway","proxy returns 502 after invalid or reset upstream response","Inspect upstream status, connect resets, response headers, target health, and proxy error logs","Remove unhealthy targets, repair upstream protocol/TLS, and use safe retries","A 504 is an upstream deadline; a 502 is an invalid or failed upstream response"),
("gateway-504","Gateway HTTP 504 timeout","proxy returns 504 when upstream exceeds gateway deadline","Compare gateway timeout, upstream duration, queueing, connect time, and dependency traces","Reduce upstream latency, align deadlines, shed load, and retry only idempotent requests","Payment timeout can surface as 504; identify the timing boundary and owning upstream"),
("gateway-connection-reset","Gateway upstream connection reset","connection reset by peer or premature close appears in proxy logs","Inspect reset timing, keepalive mismatch, upstream restarts, load balancer idle timeout, and TCP errors","Align keepalive and idle timeouts, drain restarts, and repair unstable upstreams","A reset may be rendered as 502, but transport reset evidence determines this runbook"),
]

def sections(slug, title, signal, diagnosis, mitigation, confusion):
    domain = slug.split("-")[0]
    return {
      "symptoms": f"Operators observe {signal}. Related {domain} incidents may also show latency, errors, saturation, retries, or reduced throughput, so symptoms alone are not sufficient.",
      "signals": f"Primary discriminator: {signal}. Correlate rates over the same time window and compare affected and healthy instances.",
      "diagnosis": diagnosis + ". Establish a timeline before changing capacity or retry policy.",
      "mitigation": mitigation + ". Verify recovery using the primary discriminator and customer impact metrics.",
      "escalation": f"Escalate to the {domain} service owner when impact persists after safe mitigation, data integrity is at risk, or privileged provider action is required. Include timestamps, graphs, traces, and changes.",
      "common_confusions": confusion + ". Avoid selecting a runbook from a shared keyword without the discriminator.",
    }

def write_pdf(path, title, body):
    doc = fitz.open(); page = doc.new_page(width=595, height=842)
    y=50; page.insert_text((50,y), title, fontsize=16); y += 30
    for line in body.splitlines():
        if y > 790: page=doc.new_page(width=595,height=842); y=50
        page.insert_textbox((50,y,545,y+55), line, fontsize=9); y += 55 if line else 12
    doc.set_metadata({"title": title, "author": "OpsBench v1", "creationDate": "D:20260101000000Z", "modDate": "D:20260101000000Z"})
    doc.save(path, garbage=4, deflate=True, no_new_id=True)

def main():
    corpus=BENCH/"corpus"; corpus.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True); (BENCH/"reports").mkdir(exist_ok=True)
    manifest={"dataset_version":"1.0.0","documents":[]}
    query_rows=[]; qrels={}
    for idx,(slug,title,signal,diagnosis,mitigation,confusion) in enumerate(CASES):
        secs=sections(slug,title,signal,diagnosis,mitigation,confusion)
        document_id=DOCUMENT_ID_ALIASES.get(slug,f"{slug}-runbook")
        data={"document_id":document_id,"title":title,"domain":slug.split('-')[0],"sections":{}}
        lines=[]; ids=[]
        for name in SECTIONS:
            kid=f"{slug}#{name.replace('_','-')}"; ids.append(kid); data["sections"][name]={"kb_id":kid,"text":secs[name]}
            lines += [f"[KB_ID: {kid}]", name.replace('_',' ').title(), secs[name], ""]
        yaml_path=corpus/f"{slug.replace('-','_')}.yaml"; yaml_path.write_text(yaml.safe_dump(data,sort_keys=False,allow_unicode=True),encoding="utf-8")
        pdf_path=OUT/f"{document_id}.pdf"; write_pdf(pdf_path,title,"\n".join(lines))
        manifest["documents"].append({"document_id":data["document_id"],"title":title,"domain":data["domain"],"source":str(yaml_path.relative_to(ROOT)),"generated":str(pdf_path.relative_to(ROOT)),"kb_ids":ids})
    expected_pdfs={f"{DOCUMENT_ID_ALIASES.get(slug,slug+'-runbook')}.pdf" for slug,*_ in CASES}
    for stale in OUT.glob("*.pdf"):
        if stale.name not in expected_pdfs: stale.unlink()
    # Four answerable query styles per case = 96 candidates; select exact fixed distribution below.
    specs={"lexical":16,"semantic":16,"hard_negative":16,"multi_hop":12,"noisy":12}
    offsets={"lexical":0,"semantic":5,"hard_negative":11,"multi_hop":2,"noisy":7}
    for category,count in specs.items():
        for i in range(count):
            case=CASES[(i+offsets[category])%len(CASES)]; slug,title,signal,diagnosis,mitigation,confusion=case
            if category=="lexical": query=f"{signal}; exact diagnostic code or counter and first checks?"
            elif category=="semantic": query=f"The service behaves like this: {signal.lower()}. What should an operator investigate and do safely?"
            elif category=="hard_negative": query=f"Several related failures show latency, errors and saturation. {signal}. Which discriminator and mitigation identify the real cause?"
            elif category=="multi_hop": query=f"Given {signal.lower()}, how do I confirm the cause, mitigate customer impact, verify recovery, and know when to escalate?"
            else: query=f"uh seeing {signal.lower()} maybe after chnage; whats the likely issue + safe nxt step?"
            qid=f"{category.replace('_','-')}-{i+1:03d}"; query_rows.append({"id":qid,"query":query,"category":category,"difficulty":"hard" if category in {"hard_negative","multi_hop","noisy"} else "medium","answerable":True,"tags":[slug.split('-')[0],slug]})
            rel={f"{slug}#diagnosis":3,f"{slug}#signals":3,f"{slug}#mitigation":2}
            if category=="multi_hop": rel[f"{slug}#escalation"]=2
            qrels[qid]=rel
    no_answers=["rotate Kubernetes cluster CA certificates","restore a deleted S3 object version","renew an expired Vault root token","debug DNSSEC key rollover","resize an Elasticsearch shard","rotate an Oracle wallet","repair Ceph placement groups","recover a Consul quorum"]
    for i,text in enumerate(no_answers,1):
        qid=f"no-answer-{i:03d}"; query_rows.append({"id":qid,"query":f"How do I {text}?","category":"no_answer","difficulty":"hard","answerable":False,"tags":[text.split()[1].lower()]}); qrels[qid]={}
    (BENCH/"corpus_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
    (BENCH/"queries.jsonl").write_text("".join(json.dumps(q,ensure_ascii=False)+"\n" for q in query_rows))
    (BENCH/"qrels.json").write_text(json.dumps(qrels,indent=2,ensure_ascii=False)+"\n")
    from src.observability.evaluation.opsbench import validate_dataset
    print(json.dumps(validate_dataset(BENCH),indent=2))
if __name__=="__main__": sys.path.insert(0,str(ROOT)); main()
