# D1 — RAG Pipeline 整体架构

> 对应源码：`DEV_SPEC.md`, `main.py`, `config/settings.yaml`, `src/core/settings.py`, `src/core/types.py`, `scripts/`

---

## D1.1 端到端数据流：从文档上传到生成回答的完整链路

### 两条主链路

```
离线（Ingestion）
原始 PDF
  → [Loader]      解析为统一 Document（text + metadata）
  → [Splitter]    按语义边界切分为 Chunk[]
  → [Transform]   ChunkRefiner + MetadataEnricher + ImageCaptioner
  → [Embedding]   DenseEncoder（语义向量）+ SparseEncoder（BM25 权重）
  → [Storage]     ChromaDB Upsert + BM25Index + ImageStorage + FileIntegrity

在线（Query）
用户问题（via MCP Client）
  → [MCP Server]      Stdio Transport / JSON-RPC 解析
  → [QueryProcessor]  关键词提取 + filters 构建
  → [HybridSearch]    Dense 召回 ‖ Sparse 召回（并行）
  → [RRF Fusion]      倒数排名融合
  → [Reranker]        Cross-Encoder 精排（可选）
  → [ResponseBuilder] 引用生成 + 多模态组装
  → MCP 标准响应（TextContent + ImageContent）
```

### 关键代码路径

| 步骤 | 文件 |
|------|------|
| 服务启动 | `main.py` → `src/mcp_server/server.py` |
| 离线摄取入口 | `scripts/ingest.py` → `src/ingestion/pipeline.py` |
| 在线查询入口 | `scripts/query.py` → `src/core/query_engine/hybrid_search.py` |
| MCP 工具入口 | `src/mcp_server/tools/query_knowledge_hub.py` |

### 设计要点

- 为什么 RAG 要把 Ingestion 和 Query 分成两条独立链路？
  → 离线处理可以做慢速、高质量的 LLM 增强（如 Image Captioning）；在线路径只做快速检索，不用 LLM。
- 端到端延迟的瓶颈在哪里？
  → 在线路径：Embedding（网络 RTT）+ Reranker（模型推理）；离线路径：Transform 阶段的 LLM 调用。

---

## D1.2 三层架构设计：core / ingestion / libs 各层职责与依赖方向

```
MCP Server 层（接口层）
    ↓ 调用
Core 层（核心业务逻辑：QueryEngine / Response / Trace）
    ↓ 调用
Ingestion 层（Pipeline 编排：Loader/Chunker/Transform/Embed/Store）
    ↓ 调用
Libs 层（可插拔抽象：LLM / Embedding / VectorStore / Reranker / Evaluator）
```

### 各层职责

| 层 | 目录 | 职责 |
|----|------|------|
| MCP Server 层 | `src/mcp_server/` | JSON-RPC 协议处理，tool 注册与分发 |
| Core 层 | `src/core/` | QueryEngine、ResponseBuilder、Trace — 与协议/存储无关的纯业务逻辑 |
| Ingestion 层 | `src/ingestion/` | Pipeline 编排，串联 Loader → Splitter → Transform → Embed → Store |
| Libs 层 | `src/libs/` | 每个组件的抽象基类 + 工厂 + 具体实现；其他层只依赖抽象接口 |
| Observability 层 | `src/observability/` | 结构化日志、Trace、Dashboard、评估 |

### 依赖规则（重要！）

- 上层可以依赖下层，**严禁下层依赖上层**
- Libs 层不依赖任何业务模块，只定义抽象接口
- 所有数据类型（Document/Chunk/RetrievalResult）集中定义在 `src/core/types.py`，各层共用

---

## D1.3 Pipeline 组装：配置驱动的组件组合机制

### 核心思想

`config/settings.yaml` 是唯一的"组件选择器"，改配置 = 换组件，零代码修改。

```yaml
# 改这一行，整个 LLM Provider 就换了
llm:
  provider: deepseek   # azure | openai | ollama | deepseek
  model: deepseek-chat

# 改这一行，向量数据库就换了
vector_store:
  backend: chroma      # chroma | qdrant | pinecone

# 改这一行，精排策略就换了
rerank:
  backend: none        # none | cross_encoder | llm
```

### 组装机制（工厂模式）

```
settings.yaml
    → load_settings() → Settings dataclass
    → LLMFactory.create(settings) → 对应的 BaseLLM 实现
    → EmbeddingFactory.create(settings) → 对应的 BaseEmbedding 实现
    → RerankerFactory.create(settings) → 对应的 BaseReranker 实现
    → VectorStoreFactory.create(settings) → 对应的 BaseVectorStore 实现
```

### Settings 结构

```python
@dataclass
class Settings:
    llm: LLMConfig
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    splitter: SplitterConfig
    retrieval: RetrievalConfig
    rerank: RerankConfig
    raw_config: Dict[str, Any]   # 原始 YAML，供读取自定义扩展字段
```

`raw_config` 的存在允许各模块读取 YAML 中的 schema 之外的字段（如 `ingestion.chunk_refiner.use_llm`），而不必在 Settings dataclass 里提前声明每一个字段。

---

## D1.4 核心数据类型：Document / Chunk / ChunkRecord / RetrievalResult / ProcessedQuery

> 文件：`src/core/types.py`

### 类型系统全景

```
Loader 产出          Splitter 产出          存储单元               检索输出
Document         →   Chunk[]           →   ChunkRecord         →  RetrievalResult
(id,source,text,    (id,doc_id,text,       (id,text,metadata,     (chunk_id,score,
 metadata)           index,metadata)        dense_vector,           text,metadata,
                                            sparse_vector)          source)
```

### Document 核心字段

```python
@dataclass
class Document:
    id: str          # sha256(source_path)
    source: str      # 原始文件路径
    text: str        # 标准化 Markdown 全文，图片位置用 [IMAGE: {image_id}] 占位
    metadata: dict   # source_path, doc_type, images: List[ImageRef], ...
```

### Chunk 核心字段

```python
@dataclass
class Chunk:
    id: str          # {doc_id}_{index:04d}_{hash_8chars}，稳定可重复
    doc_id: str      # 来源 Document.id（溯源链接）
    text: str        # 片段文本（含图片描述 caption 注入后的最终文本）
    index: int       # chunk 在文档中的序号
    metadata: dict   # 继承 Document.metadata + chunk_index + image_refs[]
```

### RetrievalResult：跨层统一输出

```python
@dataclass
class RetrievalResult:
    chunk_id: str
    score: float
    text: str
    metadata: dict
    source: Optional[str]  # "dense" / "sparse" / "fused" / "reranked"
```

`source` 字段的价值：调试时可以看出某条结果来自哪条召回路径，方便定位问题。

### ProcessedQuery：一次解析，多处消费

```python
@dataclass
class ProcessedQuery:
    original: str      # 原始 query 文本（给 Dense 检索用）
    keywords: List[str]  # 关键词列表（给 BM25 Sparse 检索用）
    filters: dict        # metadata 过滤条件（给 VectorStore 前置过滤用）
```

QueryProcessor 只解析一次，HybridSearch 把结果传给 Dense/Sparse 两条路，避免重复解析。

---

## D1.5 入口脚本设计：CLI 脚本的职责划分与参数传递

| 脚本 | 职责 | 核心参数 |
|------|------|---------|
| `scripts/ingest.py` | 离线文档摄取 | `--path`, `--collection`, `--force` |
| `scripts/query.py` | 在线查询调试 | `--query`, `--top-k`, `--collection`, `--verbose`, `--no-rerank` |
| `scripts/evaluate.py` | 评估运行 | 读取 `golden_test_set.json` |
| `scripts/start_dashboard.py` | 启动 Streamlit Dashboard | — |
| `main.py` | MCP Server 启动入口 | 读取 `MCP_SETTINGS_PATH` 环境变量 |

### 设计原则

- CLI 脚本**不包含业务逻辑**，只做参数解析 + 组件初始化 + 调用 Core/Ingestion 层
- 脚本可以被 Dashboard、测试、CI 复用同一套 Core 层逻辑
- `main.py` 是 MCP 协议入口，遵守"stdout 只写 JSON-RPC"约束；调试信息全走 `scripts/query.py`

### main.py 的特殊性

```python
# stdout 只能有 JSON-RPC 消息，任何 print() 都会破坏 MCP Client
# 所有日志必须走 stderr（logger 已配置）
from src.mcp_server.server import main
if __name__ == "__main__":
    main()
```

---

# D1 — RAG Pipeline: Overall Architecture (English)

> Source files: `DEV_SPEC.md`, `main.py`, `config/settings.yaml`, `src/core/settings.py`, `src/core/types.py`, `scripts/`

---

## D1.1 End-to-End Data Flow: From Document Upload to Answer Generation

### Two Main Pipelines

```
Offline (Ingestion)
Raw PDF
  → [Loader]      Parse into unified Document (text + metadata)
  → [Splitter]    Split into Chunk[] along semantic boundaries
  → [Transform]   ChunkRefiner + MetadataEnricher + ImageCaptioner
  → [Embedding]   DenseEncoder (semantic vectors) + SparseEncoder (BM25 weights)
  → [Storage]     ChromaDB Upsert + BM25Index + ImageStorage + FileIntegrity

Online (Query)
User question (via MCP Client)
  → [MCP Server]      Stdio Transport / JSON-RPC parsing
  → [QueryProcessor]  Keyword extraction + filter building
  → [HybridSearch]    Dense retrieval ‖ Sparse retrieval (parallel)
  → [RRF Fusion]      Reciprocal rank fusion
  → [Reranker]        Cross-Encoder reranking (optional)
  → [ResponseBuilder] Citation generation + multimodal assembly
  → MCP standard response (TextContent + ImageContent)
```

### Key Interview Points

- Why separate Ingestion and Query into two independent pipelines?
  → Offline processing allows slow, high-quality LLM augmentation (e.g. Image Captioning); the online path only does fast retrieval without LLM calls.
- Where is the latency bottleneck?
  → Online: Embedding (network RTT) + Reranker (model inference). Offline: LLM calls in the Transform stage.

---

## D1.2 Three-Layer Architecture: Responsibilities and Dependency Direction

```
MCP Server Layer  (interface — JSON-RPC, tool dispatch)
      ↓
Core Layer        (business logic — QueryEngine / Response / Trace)
      ↓
Ingestion Layer   (pipeline orchestration — Loader/Chunker/Transform/Embed/Store)
      ↓
Libs Layer        (pluggable abstractions — LLM / Embedding / VectorStore / Reranker)
```

**Dependency rule**: upper layers may call lower layers; lower layers must never import upper layers. All shared data types live in `src/core/types.py`.

---

## D1.3 Pipeline Assembly: Configuration-Driven Component Composition

`config/settings.yaml` is the single "component selector" — changing a config key swaps an entire component with zero code changes:

```yaml
llm:      provider: deepseek   # azure | openai | ollama | deepseek
rerank:   backend: none        # none | cross_encoder | llm
vector_store: backend: chroma  # chroma | qdrant | pinecone
```

The `raw_config` field in `Settings` retains the full YAML dict so modules can read custom extension fields without pre-declaring them in the dataclass.

---

## D1.4 Core Data Types

| Type | Created by | Key fields |
|------|-----------|------------|
| `Document` | Loader | `id`, `text` (Markdown + `[IMAGE: id]` placeholders), `metadata` |
| `Chunk` | Splitter | `id` (`{doc_id}_{index:04d}_{hash[:8]}`), `doc_id`, `text`, `index`, `metadata` |
| `ChunkRecord` | Embedding | adds `dense_vector`, `sparse_vector` |
| `RetrievalResult` | Retrieval | `chunk_id`, `score`, `text`, `metadata`, `source` ("dense"/"sparse"/"fused") |
| `ProcessedQuery` | QueryProcessor | `original`, `keywords`, `filters` — parsed once, consumed by all downstream paths |

---

## D1.5 Entry Script Design

| Script | Responsibility | Notable flags |
|--------|---------------|---------------|
| `scripts/ingest.py` | Offline ingestion | `--path`, `--collection`, `--force` |
| `scripts/query.py` | Online query debug | `--query`, `--top-k`, `--verbose`, `--no-rerank` |
| `scripts/evaluate.py` | Evaluation runner | reads `golden_test_set.json` |
| `main.py` | MCP Server entry | `MCP_SETTINGS_PATH` env var; **stdout must only contain JSON-RPC messages** |
