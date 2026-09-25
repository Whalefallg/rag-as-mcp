# D3 — Hybrid Search & Retrieval

> 对应源码：`src/core/query_engine/dense_retriever.py`, `sparse_retriever.py`, `hybrid_search.py`, `fusion.py`, `query_processor.py`, `src/core/response/`

---

## D3.1 Dense Retrieval：向量检索原理与 DenseRetriever 实现

### 原理

把 query 文本编码成向量，在向量库中找余弦相似度最高的 Top-N 个 chunk。捕捉**语义关联**，能处理同义词、模糊表达、跨语言查询。

```
用户 query
  → EmbeddingClient.embed([query])  → query_vector (1536 维等)
  → VectorStore.query(query_vector, top_k=20, filters)
  → [{chunk_id, score, text, metadata}]
  → List[RetrievalResult]
```

### DenseRetriever 核心设计

```python
class DenseRetriever:
    def retrieve(self, query: str, top_k: int, filters: dict = None,
                 collection: str = "default", trace = None) -> List[RetrievalResult]:
        vector = self._embedding.embed([query])[0]
        raw = self._vector_store.query(vector, top_k, filters, collection)
        return [RetrievalResult(chunk_id=r["id"], score=r["score"],
                                text=r["text"], metadata=r["metadata"],
                                source="dense") for r in raw]
```

### 支持的 Embedding Provider

| Provider | 模型示例 | 特点 |
|----------|---------|------|
| OpenAI / SiliconFlow | `BAAI/bge-m3` | 多语言旗舰，当前项目使用 |
| Azure OpenAI | `text-embedding-ada-002` | 企业合规 |
| Ollama | `nomic-embed-text` | 完全本地，无 API 费用 |

---

## D3.2 Sparse Retrieval：BM25 稀疏检索与 SparseRetriever 实现

### 原理

对 query 分词，在 BM25 倒排索引中查找含有这些词的 chunk，按 BM25 打分排序。捕捉**精确关键词匹配**，擅长查代码函数名、型号、专有名词。

```
用户 query
  → SparseEncoder → query_terms: {word: tf} 字典
  → BM25Indexer.query(query_terms, top_k=20)
  → [{chunk_id, score}]
  → VectorStore.get_by_ids(chunk_ids)   # 获取 text 和 metadata
  → List[RetrievalResult]
```

### BM25 公式

```
IDF(t) = log((N - df + 0.5) / (df + 0.5) + 1)

score(q, d) = Σ_t IDF(t) * tf(t,d)*(k1+1)
                            / (tf(t,d) + k1*(1-b + b*|d|/avgdl))
k1=1.5  b=0.75
```

- **IDF**（逆文档频率）：出现在越少文档里的词，权重越高（"的"权重趋近 0，"ChromaDB"权重很高）
- **TF 饱和**：k1 控制同一篇文档中词出现 10 次 vs 100 次的差距，避免超长文档垄断结果
- **长度归一化**：b=0.75 对长文档做适度惩罚，防止长文档因包含更多词而占优

### 持久化结构

```
data/db/bm25/
  postings.json   # {term: {idf, postings: [{chunk_id, tf, doc_length}]}}
  meta.json       # {chunk_id: {doc_length, source}}
```

索引序列化为 JSON，重启后 `_load_if_needed()` 懒加载，不重建。

---

## D3.3 Hybrid Search 融合：RRF 算法与 Fusion 模块设计

### 为什么需要融合

Dense 分数（余弦相似度，0~1）和 Sparse 分数（BM25，无上界）量纲不同，无法直接相加。

**RRF（Reciprocal Rank Fusion）** 只看排名，不看分数，天然规避量纲问题：

```
RRF_score(chunk) = Σ_i  1 / (k + rank_i)
```

其中 `k=60` 是平滑参数（经验值，在多个信息检索 benchmark 上最优）。

### RRFusion 实现

```python
class RRFusion:
    def __init__(self, k: int = 60):
        self._k = k

    def fuse(self, result_lists: List[List[RetrievalResult]], top_k: int) -> List[RetrievalResult]:
        rrf_scores = {}
        for result_list in result_lists:
            for rank, result in enumerate(result_list, start=1):
                cid = result.chunk_id
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self._k + rank)
        ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])[:top_k]
        return [RetrievalResult(chunk_id=cid, score=score, ..., source="fused")
                for cid, score in ranked]
```

### HybridSearch 完整流程

```python
def search(self, query, top_k, filters, collection, trace):
    # 1. 预处理
    processed = self._query_proc.process(query, filters)

    # 2. 并行召回（任一路失败自动降级为空列表）
    dense_results  = self._dense.retrieve(...)   # Top-20
    sparse_results = self._sparse.retrieve(...)  # Top-20

    # 3. RRF 融合
    fused = self._fusion.fuse([dense_results, sparse_results], top_k)

    # 4. 后置 metadata 过滤（兜底）
    if processed.filters:
        fused = self._apply_metadata_filters(fused, processed.filters)

    return fused[:final_top_k]  # 默认 Top-10
```

### 为什么 Dense/Sparse 各召回 20 个，最终只返回 10 个？

因为两路各自独立召回 Top-20，融合时才有真正意义的"超配额候选"——
若每路只召回 10 个，RRF 融合后最多也是 10 个，没有提升效果。

---

## D3.4 QueryProcessor：查询预处理与查询扩展机制

### ProcessedQuery 结构

```python
@dataclass
class ProcessedQuery:
    original: str          # 原始 query（给 Dense）
    keywords: List[str]    # 去停用词后的关键词列表（给 BM25 Sparse）
    filters: dict          # metadata 过滤条件（给 VectorStore 前置过滤）
```

一次解析，三路消费，避免重复处理。

### 关键词提取策略

- 中文：jieba 分词 + 停用词过滤
- 英文：按空格/标点切分 + 停用词过滤
- 保留专有名词、大写词（型号、品牌名、代码名）

### Metadata 过滤

从 query 中解析结构化约束，例如：

```
"查 2024 年的 PDF 文件" → filters = {"doc_type": "pdf"}
"在 default 集合里搜" → filters = {"collection": "default"}
```

过滤有两个时机：
- **前置过滤（Pre-filter）**：VectorStore.query() 时传入 filters，缩小候选集（成本低）
- **后置过滤（Post-filter）**：fusion 后再过滤，作为前置过滤遗漏的兜底

---

## D3.5 Response 构建：ResponseBuilder / CitationGenerator / MultimodalAssembler

### MCP 响应格式

```json
{
  "content": [
    {"type": "text", "text": "根据文档...\n\n[1] config_guide.pdf 第5页"},
    {"type": "image", "data": "<base64>", "mimeType": "image/png"}
  ],
  "isError": false
}
```

### ResponseBuilder 职责

把 `List[RetrievalResult]` 组装成 MCP 标准响应：

```
RetrievalResult[]
  → CitationGenerator.generate()  → [Citation(source, page, chunk_id, score)]
  → 格式化为 Markdown（含 [1][2] 引用标注）
  → MultimodalAssembler.assemble()  → 读取图片 → Base64 编码
  → [TextContent, ImageContent, ...]
```

### CitationGenerator

为每个 chunk 生成引用信息，包含：

| 字段 | 来源 |
|------|------|
| `source` | `chunk.metadata["source_path"]` |
| `page` | `chunk.metadata.get("page")` |
| `chunk_id` | `result.chunk_id` |
| `score` | `result.score` |

### MultimodalAssembler

当 chunk 含有 `image_refs` 时：
1. 查询 `ImageStorage` 获取图片文件路径
2. 读取图片文件 bytes
3. Base64 编码
4. 包装为 `{"type": "image", "data": "<base64>", "mimeType": "image/png"}`
5. 追加到 content 数组

Client 端（Claude Desktop 等）按需渲染图片；不支持图片的 Client 只展示文本内容。

---

# D3 — Hybrid Search & Retrieval (English)

> Source files: `src/core/query_engine/dense_retriever.py`, `sparse_retriever.py`, `hybrid_search.py`, `fusion.py`, `query_processor.py`, `src/core/response/`

---

## D3.1 Dense Retrieval: Vector Search Principles and DenseRetriever Implementation

Encode the query as a vector, then find the top-N chunks in the vector store by cosine similarity. Captures **semantic associations** — handles synonyms, paraphrasing, and cross-lingual queries.

```
query → EmbeddingClient.embed([query]) → query_vector
      → VectorStore.query(vector, top_k, filters)
      → List[RetrievalResult]  (source="dense")
```

---

## D3.2 Sparse Retrieval: BM25 and SparseRetriever Implementation

Tokenize the query, look up chunks containing those terms in the BM25 inverted index, and rank by BM25 score. Captures **exact keyword matches** — function names, model numbers, proper nouns.

### BM25 Formula

```
IDF(t) = log((N − df + 0.5) / (df + 0.5) + 1)

score(q, d) = Σ_t IDF(t) · tf(t,d)·(k1+1)
                            / (tf(t,d) + k1·(1 − b + b·|d|/avgdl))
k1 = 1.5   b = 0.75
```

- **k1 = 1.5**: controls TF saturation speed — higher k1 means diminishing returns on high-frequency terms come more slowly.
- **b = 0.75**: length normalization strength — 1.0 = full normalization, 0 = no normalization.

---

## D3.3 Hybrid Search Fusion: RRF Algorithm and Fusion Module Design

Dense scores (cosine, 0–1) and Sparse scores (BM25, unbounded) are **not comparable** — a weighted sum would be distorted by the different scales.

**Reciprocal Rank Fusion (RRF)** looks only at rank, not raw score:

```
RRF_score(chunk) = Σ_i  1 / (k + rank_i)     k = 60  (empirical optimum)
```

```python
class RRFusion:
    def fuse(self, result_lists, top_k):
        for result_list in result_lists:
            for rank, result in enumerate(result_list, start=1):
                rrf_scores[result.chunk_id] += 1.0 / (self._k + rank)
        ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])
        return top_k results with source="fused"
```

### Why Each Path Over-retrieves (Top-20 → Final Top-10)

If each path only returned 10 results, the fusion would have nothing extra to promote — "over-quota candidates" are what give RRF its lift.

---

## D3.4 QueryProcessor: Query Pre-processing and Expansion

`ProcessedQuery` is parsed **once** and consumed by all three downstream paths:

| Field | Consumer |
|-------|---------|
| `original` | Dense retrieval (embedding) |
| `keywords` | BM25 sparse retrieval |
| `filters` | VectorStore pre-filter + post-filter fallback |

Filtering has two moments:
- **Pre-filter**: passed to `VectorStore.query()` to narrow the candidate set cheaply.
- **Post-filter**: applied after fusion as a safety net for any results that slipped through.

---

## D3.5 Response Building: ResponseBuilder / CitationGenerator / MultimodalAssembler

```
List[RetrievalResult]
  → CitationGenerator  → [Citation(source, page, chunk_id, score)]
  → Markdown text with [1][2] inline reference markers
  → MultimodalAssembler → read image files → Base64 encode
  → [TextContent, ImageContent, ...]   (MCP content array)
```

Each `ImageContent` is `{"type": "image", "data": "<base64>", "mimeType": "image/png"}`. Clients that don't support images (e.g. some Copilot versions) simply ignore these entries and display the text content only.
