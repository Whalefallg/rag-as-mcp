# 项目知识总结索引

> 项目：RAG AS MCP  
> 生成时间：2026-07-07  
> 总计：10 个知识域，45 个知识点

---

## 快速导航

| 文件 | 知识域 | 知识点数 | 核心关键词 |
|------|--------|---------|-----------|
| [D1](./D1-RAG-Pipeline-整体架构.md) | RAG Pipeline 整体架构 | 5 | 端到端数据流、三层架构、配置驱动、类型系统 |
| [D2](./D2-Ingestion-Pipeline.md) | Ingestion Pipeline | 5 | Chunking、Transform 链、双路编码、存储协同 |
| [D3](./D3-Hybrid-Search-Retrieval.md) | Hybrid Search & Retrieval | 5 | Dense/Sparse 召回、RRF 融合、QueryProcessor |
| [D4](./D4-Rerank-机制.md) | Rerank 机制 | 4 | Cross-Encoder、LLM Rerank、Fallback、两段式架构 |
| [D5](./D5-MCP-Server-协议.md) | MCP Server 协议 | 4 | JSON-RPC 2.0、Stdio Transport、Tool 注册、生命周期 |
| [D6](./D6-可插拔架构-配置系统.md) | 可插拔架构 & 配置系统 | 5 | 工厂模式、装饰器注册、Settings、多 Provider |
| [D7](./D7-多模态处理.md) | 多模态处理 | 4 | PDF 解析、Vision LLM、ImageCaptioner、图转文 |
| [D8](./D8-可观测性-评估体系.md) | 可观测性 & 评估体系 | 5 | TraceContext、Dashboard、Ragas、JSON Lines |
| [D9](./D9-测试策略-工程化.md) | 测试策略 & 工程化 | 4 | 测试金字塔、Mock 策略、pyproject.toml、CLI 脚本 |
| [D10](./D10-Document-Manager-幂等性.md) | Document Manager & 幂等性 | 4 | SHA256 去重、Upsert、状态追踪、跨存储协调 |

---

## 知识点全览（45 个）

### D1 RAG Pipeline 整体架构

- **D1.1** 端到端数据流：Ingestion 离线链路 + Query 在线链路
- **D1.2** 三层架构：core（业务逻辑）/ ingestion（Pipeline 编排）/ libs（可插拔抽象）
- **D1.3** Pipeline 组装：配置驱动 + 工厂模式，改 YAML = 换组件
- **D1.4** 核心数据类型：Document / Chunk / ChunkRecord / RetrievalResult / ProcessedQuery
- **D1.5** 入口脚本设计：ingest.py / query.py / evaluate.py / main.py 职责划分

### D2 Ingestion Pipeline

- **D2.1** Pipeline 整体流程：integrity → load → split → transform → encode → upsert → store_images
- **D2.2** Chunking 策略：RecursiveSplitter + DocumentChunker 适配器（6 个增值职责）
- **D2.3** Transform 链：ChunkRefiner → MetadataEnricher → ImageCaptioner（顺序有依赖）
- **D2.4** Embedding 编码：Dense + Sparse 双路，BatchProcessor 分批处理
- **D2.5** 存储层：Chroma（向量）/ BM25（倒排索引）/ ImageStorage / FileIntegrity 四类存储

### D3 Hybrid Search & Retrieval

- **D3.1** Dense Retrieval：query embedding → 余弦相似度检索，捕捉语义关联
- **D3.2** Sparse Retrieval：BM25 公式（k1=1.5, b=0.75），倒排索引，精确关键词匹配
- **D3.3** Hybrid Search 融合：RRF 算法（`1/(k+rank)`，k=60），两路超配额召回再融合
- **D3.4** QueryProcessor：一次解析生成 original + keywords + filters，三路消费
- **D3.5** Response 构建：CitationGenerator + MultimodalAssembler → MCP content 数组

### D4 Rerank 机制

- **D4.1** Reranker 抽象：BaseReranker + NoneReranker（Null Object Pattern）+ Factory
- **D4.2** CrossEncoder Reranker：联合编码 (query, chunk)，懒加载，批量推理，失败降级
- **D4.3** LLM Reranker：结构化 Prompt 输出 ranked ids，适合无本地 GPU 场景
- **D4.4** 集成位置：Fusion Top-30 → Reranker → Top-10；超时/异常 fallback=True

### D5 MCP Server 协议

- **D5.1** MCP 概述：JSON-RPC 2.0，Stdio Transport，子进程通信，stdout 只写 MCP 消息
- **D5.2** Tool 注册：TOOL_NAME / DESCRIPTION / INPUT_SCHEMA / execute 四元组，延迟导入
- **D5.3** ProtocolHandler：initialize（能力协商）/ tools/list / tools/call 路由分发
- **D5.4** 生命周期：for raw_line in stdin 主循环，异常隔离，stdin 关闭自然退出

### D6 可插拔架构 & 配置系统

- **D6.1** 工厂模式全景：5 大工厂（LLM/Embedding/Reranker/VectorStore/Evaluator）装饰器注册
- **D6.2** Settings 加载：YAML → dataclass → validate（fail-fast），raw_config 保留扩展字段
- **D6.3** LLM Provider 切换：统一 `BaseLLM.chat(messages)` 接口，OpenAI 兼容协议适配多厂商
- **D6.4** Embedding Provider 选型：BGE-M3（中文/多语言）/ text-embedding-3（英文）/ Ollama（本地）
- **D6.5** Base 类设计哲学：接口隔离，最小化抽象，继承层次，4 步扩展新 Provider

### D7 多模态处理

- **D7.1** PDF 解析：MarkItDown → Markdown，`[IMAGE: id]` 占位符，FileIntegrity SHA256 跳过
- **D7.2** Vision LLM：BaseVisionLLM 抽象，chat_with_image，图片 Base64 编码，自动压缩
- **D7.3** ImageCaptioner：读 image_refs → 调 Vision LLM → caption 注入正文，降级不阻塞
- **D7.4** 多模态存储：ChromaDB（含 caption 的文本向量）+ 文件系统（原始图片）+ SQLite 映射

### D8 可观测性 & 评估体系

- **D8.1** Trace 系统：TraceContext（trace_id / trace_type / stages[]）+ TraceCollector（→ .jsonl）
- **D8.2** Dashboard 架构：6 页面 Streamlit + Services 层（TraceService / DataService / ConfigService）
- **D8.3** 评估指标：Hit Rate@K / MRR / NDCG（检索）+ Faithfulness / Answer Relevancy（生成）
- **D8.4** 评估框架：BaseEvaluator + CompositeEvaluator 并行 + RagasEvaluator + CustomEvaluator
- **D8.5** 日志系统：JSONFormatter → stderr（MCP 约束），JSON Lines 追加写入，jq 友好

### D9 测试策略 & 工程化

- **D9.1** 测试金字塔：Unit（大量/mock）/ Integration（中量/临时 DB）/ E2E（少量/真实 Server）
- **D9.2** Fixtures 与 Mock：deterministic Fake 优先于 MagicMock，tmp_path 隔离，conftest.py
- **D9.3** pyproject.toml：requires-python≥3.10，pytest markers，覆盖率目标 80%/100%/100%
- **D9.4** 脚本设计：脚本≠业务逻辑，只做参数解析+初始化+调用，Core 层可被多入口复用

### D10 Document Manager & 幂等性

- **D10.1** 文档去重：文件级 SHA256 + Chunk 级 content_hash，两层互补，零成本增量
- **D10.2** 增量 Ingestion：force=True 强制重摄，Upsert 语义保证向量库无重复
- **D10.3** Collection 管理：DocumentManager 跨 4 存储协调删除（Chroma/BM25/Images/Integrity）
- **D10.4** 状态追踪：未入库/已入库/处理失败/已删除，WAL 模式 SQLite 并发安全

---

## 常见设计问题

| 问题 | 答案所在知识点 |
|------|-------------|
| RAG 系统的完整链路是什么？ | D1.1 |
| 为什么用 Markdown 而不是纯文本作为 Chunk 的格式？ | D2.2, D7.1 |
| RRF 比直接加权平均好在哪里？ | D3.3 |
| Bi-Encoder 和 Cross-Encoder 有什么区别？ | D4.2 |
| 为什么 BM25 召回和向量召回各取 20 个，最终只返回 10 个？ | D3.3 |
| MCP 协议的 stdout 约束是什么？为什么？ | D5.1, D5.4 |
| 怎么做到改配置文件就能换 LLM？ | D6.1, D6.2 |
| 图片怎么被检索到（架构上）？ | D7.3, D7.4 |
| 如何保证同一文档重复摄取不产生重复向量？ | D10.1, D10.2 |
| BM25 中 k1 和 b 参数分别控制什么？ | D3.2 |
| Cross-Encoder 为什么只对 Top-30 做精排而不对全量？ | D4.4 |
| Trace 系统是怎么设计的？ | D8.1 |

---

# Knowledge Base Index (English)

> Project: RAG AS MCP  
> Total: 10 knowledge domains, 45 knowledge points

---

## Quick Navigation

| File | Domain | Points | Keywords |
|------|--------|--------|----------|
| [D1](./D1-RAG-Pipeline-整体架构.md) | RAG Pipeline: Overall Architecture | 5 | end-to-end flow, 3-layer arch, config-driven, type system |
| [D2](./D2-Ingestion-Pipeline.md) | Ingestion Pipeline | 5 | chunking, transform chain, dual encoding, storage coordination |
| [D3](./D3-Hybrid-Search-Retrieval.md) | Hybrid Search & Retrieval | 5 | Dense/Sparse recall, RRF fusion, QueryProcessor |
| [D4](./D4-Rerank-机制.md) | Rerank Mechanism | 4 | Cross-Encoder, LLM Rerank, fallback, two-stage architecture |
| [D5](./D5-MCP-Server-协议.md) | MCP Server Protocol | 4 | JSON-RPC 2.0, Stdio Transport, tool registration, lifecycle |
| [D6](./D6-可插拔架构-配置系统.md) | Pluggable Architecture & Config | 5 | factory pattern, decorator registration, Settings, multi-provider |
| [D7](./D7-多模态处理.md) | Multimodal Processing | 4 | PDF parsing, Vision LLM, ImageCaptioner, image-to-text |
| [D8](./D8-可观测性-评估体系.md) | Observability & Evaluation | 5 | TraceContext, Dashboard, Ragas, JSON Lines |
| [D9](./D9-测试策略-工程化.md) | Testing Strategy & Engineering | 4 | test pyramid, mock strategy, pyproject.toml, CLI scripts |
| [D10](./D10-Document-Manager-幂等性.md) | Document Manager & Idempotency | 4 | SHA256 dedup, upsert, state tracking, cross-storage coordination |

---

## All 45 Knowledge Points

### D1 — RAG Pipeline: Overall Architecture
- **D1.1** End-to-end data flow: offline Ingestion pipeline + online Query pipeline
- **D1.2** Three-layer architecture: core (business logic) / ingestion (pipeline orchestration) / libs (pluggable abstractions)
- **D1.3** Pipeline assembly: configuration-driven + factory pattern — change YAML = swap component
- **D1.4** Core data types: Document / Chunk / ChunkRecord / RetrievalResult / ProcessedQuery
- **D1.5** Entry script design: responsibility boundaries across ingest.py / query.py / evaluate.py / main.py

### D2 — Ingestion Pipeline
- **D2.1** Pipeline overall flow: integrity → load → split → transform → encode → upsert → store_images
- **D2.2** Chunking strategy: RecursiveSplitter + DocumentChunker adapter (6 value-adds)
- **D2.3** Transform chain: ChunkRefiner → MetadataEnricher → ImageCaptioner (order is dependency-driven)
- **D2.4** Embedding: Dense + Sparse dual-path, BatchProcessor batch processing
- **D2.5** Storage layer: Chroma (vectors) / BM25 (inverted index) / ImageStorage / FileIntegrity — four stores

### D3 — Hybrid Search & Retrieval
- **D3.1** Dense Retrieval: query embedding → cosine similarity search; captures semantic associations
- **D3.2** Sparse Retrieval: BM25 formula (k1=1.5, b=0.75); inverted index; exact keyword matching
- **D3.3** Hybrid Search fusion: RRF algorithm (`1/(k+rank)`, k=60); over-quota recall before fusion
- **D3.4** QueryProcessor: one parse → `original` + `keywords` + `filters`; consumed by all three downstream paths
- **D3.5** Response building: CitationGenerator + MultimodalAssembler → MCP content array

### D4 — Rerank Mechanism
- **D4.1** Reranker abstraction: BaseReranker + NoneReranker (Null Object Pattern) + Factory
- **D4.2** CrossEncoder Reranker: joint encoding of (query, chunk); lazy load; batch inference; failure fallback
- **D4.3** LLM Reranker: structured prompt outputs ranked IDs; suitable when no local GPU is available
- **D4.4** Integration position: Fusion Top-30 → Reranker → Top-10; timeout/exception triggers fallback=True

### D5 — MCP Server Protocol
- **D5.1** MCP overview: JSON-RPC 2.0, Stdio Transport, subprocess communication; stdout must only contain MCP messages
- **D5.2** Tool registration: TOOL_NAME / DESCRIPTION / INPUT_SCHEMA / execute four-tuple; delayed import
- **D5.3** ProtocolHandler: initialize (capability negotiation) / tools/list / tools/call routing
- **D5.4** Lifecycle: `for raw_line in stdin` main loop; exception isolation; natural exit on stdin close

### D6 — Pluggable Architecture & Configuration System
- **D6.1** Factory pattern overview: 5 factories (LLM/Embedding/Reranker/VectorStore/Evaluator); decorator registration
- **D6.2** Settings loading: YAML → dataclass → validate (fail-fast); `raw_config` preserves extension fields
- **D6.3** LLM provider switching: unified `BaseLLM.chat(messages)` interface; OpenAI-compatible protocol adapts many vendors
- **D6.4** Embedding provider selection: BGE-M3 (Chinese/multilingual) / text-embedding-3 (English) / Ollama (local)
- **D6.5** Base class design: interface segregation, minimal abstraction, inheritance hierarchy, 4-step extension pattern

### D7 — Multimodal Processing
- **D7.1** PDF parsing: MarkItDown → Markdown; `[IMAGE: id]` placeholders; FileIntegrity SHA256 skip
- **D7.2** Vision LLM: BaseVisionLLM abstraction; chat_with_image; image Base64 encoding; auto-resize
- **D7.3** ImageCaptioner: reads image_refs → calls Vision LLM → injects caption into chunk.text; graceful degradation
- **D7.4** Multimodal storage: ChromaDB (text+caption vectors) + filesystem (raw images) + SQLite (id→path mapping)

### D8 — Observability & Evaluation System
- **D8.1** Trace system: TraceContext (trace_id / trace_type / stages[]) + TraceCollector (→ .jsonl)
- **D8.2** Dashboard architecture: 6-page Streamlit + Services layer (TraceService / DataService / ConfigService)
- **D8.3** Evaluation metrics: Hit Rate@K / MRR / NDCG (retrieval) + Faithfulness / Answer Relevancy (generation)
- **D8.4** Evaluation framework: BaseEvaluator + CompositeEvaluator parallel execution + RagasEvaluator + CustomEvaluator
- **D8.5** Logging system: JSONFormatter → stderr (MCP constraint); JSON Lines append-write; jq-friendly

### D9 — Testing Strategy & Engineering
- **D9.1** Test pyramid: Unit (many/mock) / Integration (some/temp DB) / E2E (few/real Server)
- **D9.2** Fixtures and mock: deterministic Fake preferred over MagicMock; `tmp_path` isolation; `conftest.py`
- **D9.3** pyproject.toml: requires-python≥3.10; pytest markers; coverage targets 80%/100%/100%
- **D9.4** Script design: scripts ≠ business logic; only arg parsing + init + Core layer call; reusable across entry points

### D10 — Document Manager & Idempotency
- **D10.1** Document deduplication: file-level SHA256 + chunk-level content hash — two complementary layers
- **D10.2** Incremental ingestion: `force=True` override; Upsert semantics ensure no duplicates in vector store
- **D10.3** Collection management: DocumentManager coordinates deletion across 4 stores (Chroma/BM25/Images/Integrity)
- **D10.4** State tracking: never ingested / ingested / failed / deleted; WAL-mode SQLite for concurrency safety

---

## High-Frequency Interview Questions (English)

| Question | Knowledge Point |
|----------|----------------|
| Describe the complete RAG pipeline end-to-end. | D1.1 |
| Why use Markdown as the intermediate format for chunking? | D2.2, D7.1 |
| Why is RRF better than a direct weighted average of scores? | D3.3 |
| What is the difference between a Bi-Encoder and a Cross-Encoder? | D4.2 |
| Why does each retrieval path over-retrieve (Top-20 → final Top-10)? | D3.3 |
| What is the stdout constraint for an MCP Server and why does it exist? | D5.1, D5.4 |
| How does changing one config key swap the entire LLM provider? | D6.1, D6.2 |
| How are images made searchable via text retrieval? | D7.3, D7.4 |
| How do you guarantee ingesting the same document twice produces no duplicates? | D10.1, D10.2 |
| What do the k1 and b parameters in BM25 control? | D3.2 |
| Why does Cross-Encoder reranking only run on Top-30 candidates, not all results? | D4.4 |
| How is the Trace system designed and what problem does it solve? | D8.1 |
