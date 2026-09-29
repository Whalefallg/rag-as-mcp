# D11 — Agentic RAG / Adaptive Retrieval

## 为什么固定 RAG 流程不总是足够

固定单次检索对简单事实问题高效可靠，但复杂比较、多跳问题、精确标识符和含糊表达需要不同策略。统一强制多轮调用又会增加延迟和成本。因此项目保留成熟的 Dense + BM25 + RRF 原语，在其上增加轻量、显式、可测试的自适应层。

## Adaptive RAG 的定义

这里的 Agentic RAG 是 **bounded adaptive retrieval loop**，不是无界自主 Agent。`auto` router 让简单问题直接走 classic HybridSearch；只有复杂查询才进入分析、规划、证据评分和纠正检索。

```text
MCP Query -> route -> classic HybridSearch ------------------------+
                  \-> analyze -> plan -> decompose/multi-query ----+-> grade
                                                                     | enough
                                                                     +-> rerank -> response
                                                                     | insufficient
                                                                     +-> rewrite -> retry (bounded)
```

## 组件职责

- `QueryAnalyzer`：利用长度、连接词、比较结构和 identifier regex 分类；可通过项目 `BaseLLM` abstraction 做结构化增强，失败退回 deterministic。
- `RetrievalPlanner`：选择 hybrid、sparse-focused、multi-query 或 decomposed 策略，只制定计划，不实现检索器。
- Multi-query / decomposition：每个子查询调用既有 `HybridSearch.search()`，结果由 RRF 合并并按 `chunk_id` 去重。
- `EvidenceGrader`：结合结果数量、词法覆盖、子查询覆盖和来源多样性给出 confidence、coverage 与 reason codes；LLM 仅是可选增强。
- `QueryRewriter`：去除会话噪音、保留错误码和标识符，把低覆盖查询改成更明确的检索表达。
- `AgenticRAGOrchestrator`：只负责状态转换、预算、fallback、termination 和最终 rerank。

## Agent state 与终止语义

状态显式跟踪 iteration、retrieval calls、rewrite count、subquery count、LLM calls、degraded 和 exhausted。终止原因包括 `SUFFICIENT_EVIDENCE`、`MAX_ITERATIONS`、`NO_RESULTS`、`CLASSIC_ROUTE`、`ERROR_FALLBACK`。默认最多两轮；预算耗尽时返回评分最好的可用候选，而不是继续调用模型。

## Fallback 语义

LLM 分析或评分失败时退回 deterministic；规划失败退回单查询 hybrid plan；multi-query 异常退回原始查询 HybridSearch；rewrite 失败停止重试并返回最佳证据；MCP 层捕获 orchestrator 未预期异常并执行完整 classic pipeline。所有降级写入同一个 `TraceContext`。

## 为什么不是每次查询都跑 Agent loop

“What is BM25?” 这类单意图问题多轮规划几乎不增加召回质量，却会增加延迟、模型成本和故障面。Adaptive routing 把额外计算留给比较、多跳和覆盖不足的问题，兼顾质量与吞吐。

## 为什么建立在 deterministic retrieval primitive 上

Dense、BM25、RRF 与 reranker 已有稳定 contract、离线测试和可预测降级。Agent 层组合这些原语，而不重新实现它们，因而智能决策失败时仍可提供可靠检索服务，并能在无 API key 的 CI 环境完整测试。

## 成本、延迟与评估

预算限制 `max_iterations`、`max_subqueries`、`max_candidate_results`；昂贵 reranker 只在最终候选池运行一次。LLM-assisted 模式会增加网络延迟和费用，默认关闭。

`python scripts/evaluate.py --mode classic|agentic|compare` 保留 Hit Rate、MRR、Precision@K 和 latency，并为 agentic 增加 average iterations、average retrieval calls、rewrite rate 与 degradation rate。Golden set 包含 simple、multi-hop、comparison、corrective、identifier 和 insufficient-evidence 案例。结果必须来自真实 corpus 执行，不允许针对 fixture 硬编码查询或改写。
