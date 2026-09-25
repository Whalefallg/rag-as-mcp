# D4 — Rerank 机制

> 对应源码：`src/libs/reranker/`, `src/core/query_engine/reranker.py`, `config/prompts/rerank.txt`

---

## D4.1 Reranker 抽象与工厂模式：BaseReranker 与 RerankerFactory 设计

### 设计动机

精排是 RAG 系统中成本最高的步骤（Cross-Encoder 每次需过模型），但也是提升 Top-K 质量最有效的手段。需要满足：
1. 可以随时关闭（`backend=none`）而不改任何调用代码
2. 后端可以在 Cross-Encoder / LLM 之间切换
3. 失败时自动降级，不影响系统可用性

### BaseReranker 抽象接口

```python
class BaseReranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        trace=None,
    ) -> List[Dict[str, Any]]:
        """候选列表从最相关到最不相关排序，返回 top_k 条"""
        pass
```

### NoneReranker（Null Object Pattern）

```python
class NoneReranker(BaseReranker):
    def rerank(self, query, candidates, top_k=None, trace=None):
        return candidates[:top_k] if top_k else candidates
```

**空对象模式**的价值：调用方不需要 `if reranker is not None` 判断，代码路径始终一致。`backend=none` 时直接返回 RRF 融合排序，零额外开销。

### RerankerFactory

```python
# 注册机制：装饰器自动注册到工厂
@register_reranker("cross_encoder")
class CrossEncoderReranker(BaseReranker): ...

@register_reranker("llm")
class LLMReranker(BaseReranker): ...

# 工厂创建
def create_reranker(settings: Settings) -> BaseReranker:
    backend = settings.rerank.backend   # none | cross_encoder | llm
    ...
```

---

## D4.2 CrossEncoder Reranker：模型原理与实现细节

### Bi-Encoder vs Cross-Encoder 对比

| 维度 | Bi-Encoder（Dense 召回） | Cross-Encoder（精排） |
|------|------------------------|---------------------|
| 输入 | query 和 doc 分别编码 | query + doc 联合编码 |
| 速度 | 快（向量预计算） | 慢（每次过模型） |
| 质量 | 中等 | 高（能捕捉 query-doc 交互） |
| 用途 | 大规模召回（Top-20~100） | 精排小候选集（Top-10~30） |

### 实现细节

```python
@register_reranker("cross_encoder")
class CrossEncoderReranker(BaseReranker):
    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def rerank(self, query, candidates, top_k=None, trace=None):
        scorer = self._get_scorer()   # 懒加载 sentence-transformers CrossEncoder
        pairs = [(query, c.get("text", "")) for c in candidates]
        scores = scorer.predict(pairs).tolist()   # 批量推理

        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = score

        ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:top_k] if top_k else ranked
```

### 关键工程细节

- **懒加载**：`_get_scorer()` 在首次调用时才加载模型（避免启动时间过长）
- **失败降级**：`try/except` 捕获所有异常，回退原始候选列表，确保系统不崩溃
- **批量推理**：`scorer.predict(pairs)` 一次性对所有 (query, chunk) 对打分，比逐条调用快 10x
- **默认模型**：`ms-marco-MiniLM-L-6-v2` 是轻量高效的 Cross-Encoder，在 CPU 上也可运行

### 性能优化策略

- 只对 Top-M（30 个）候选做精排，不对全量做（否则慢到不可用）
- 设置超时机制，超时自动降级到 RRF 排序
- 建议 GPU 环境下可扩大 top_m，CPU 建议 top_m ≤ 20

---

## D4.3 LLM Reranker：基于大语言模型的重排序方案与 Prompt 设计

### 使用场景

- 没有本地 GPU，无法运行 Cross-Encoder
- 需要理解复杂指令（如"优先返回包含代码示例的文档"）
- 候选集较小（≤ 20 个）时可以接受 LLM 的延迟

### Prompt 设计原则（`config/prompts/rerank.txt`）

```
要求 LLM 输出严格的结构化格式（JSON 的 ranked ids），
不满足 schema 时抛出可读错误（而不是静默失败）。

候选数应更小（M ≤ 20），控制 Token 消耗。
原始排名作为兜底：LLM 输出解析失败时回退 RRF 结果。
```

### LLMReranker 实现要点

```python
@register_reranker("llm")
class LLMReranker(BaseReranker):
    def rerank(self, query, candidates, top_k=None, trace=None):
        prompt = self._build_prompt(query, candidates)   # 读 rerank.txt 模板
        response = self._llm.chat([{"role": "user", "content": prompt}])
        ranked_ids = self._parse_response(response)      # 解析 JSON 格式的 ranked ids
        # 按 ranked_ids 重排候选列表
        id_to_candidate = {c["chunk_id"]: c for c in candidates}
        ranked = [id_to_candidate[cid] for cid in ranked_ids if cid in id_to_candidate]
        return ranked[:top_k] if top_k else ranked
```

---

## D4.4 Rerank 在检索 Pipeline 中的集成位置与效果分析

### 集成位置（Core 层编排器）

```
HybridSearch.search()
    → fused = RRFusion.fuse(dense_results + sparse_results)  # Top-30
    ↓
Core Reranker（src/core/query_engine/reranker.py）
    → reranker.rerank(query, fused, top_k=10)
    ↓
Top-10 精排结果 → ResponseBuilder
```

Core 层的 `reranker.py` 是**编排层**，负责：
1. 接收 Fusion 结果（candidates as `List[RetrievalResult]`）
2. 转换为 Reranker 接口格式（`List[Dict]`）
3. 调用 `libs.reranker` 后端执行精排
4. 失败/超时时捕获异常，标记 `fallback=True`，回退 Fusion 排序

### Fallback 机制

```python
try:
    ranked = self._reranker.rerank(query, candidates, top_k)
except Exception:
    # 精排失败 → 使用 RRF 排序作为最终结果，保证系统可用
    ranked = candidates[:top_k]
    # trace 中标记 fallback=True
```

### 效果分析

| 场景 | 不开 Rerank | 开 Cross-Encoder Rerank |
|------|------------|------------------------|
| 精确关键词查询 | 好（BM25 擅长） | 相近 |
| 语义模糊查询 | 中等 | 明显提升（理解 query 意图） |
| 专业领域查询 | 弱（无领域知识） | 取决于模型训练数据 |
| 延迟（CPU） | ~100ms | ~500ms~2s（取决于 top_m） |

**两段式架构**的价值：粗排（低成本泛召回）→ 精排（高成本精过滤），在不牺牲整体响应速度的前提下大幅提升 Top-K 精度。

---

# D4 — Rerank Mechanism (English)

> Source files: `src/libs/reranker/`, `src/core/query_engine/reranker.py`, `config/prompts/rerank.txt`

---

## D4.1 Reranker Abstraction and Factory Pattern

Reranking is the most expensive step in a RAG pipeline (Cross-Encoder runs the full model per candidate), but also the most effective quality boost. Three requirements:
1. Can be turned off (`backend=none`) without changing any calling code.
2. Backend can switch between Cross-Encoder and LLM.
3. Failures must auto-degrade — never crash the system.

**Null Object Pattern** (`NoneReranker`): callers never need an `if reranker is not None` guard — the code path is identical whether reranking is enabled or not.

**Decorator registration**: each backend registers itself with `@register_reranker("cross_encoder")`, so adding a new backend only requires a new file + decorator — no factory code changes.

---

## D4.2 CrossEncoder Reranker: Model Principles and Implementation

### Bi-Encoder vs Cross-Encoder

| Dimension | Bi-Encoder (Dense retrieval) | Cross-Encoder (Reranking) |
|-----------|-----------------------------|-----------------------------|
| Input | Query and doc encoded independently | Query + doc encoded jointly |
| Speed | Fast (pre-computed vectors) | Slow (full model pass per pair) |
| Quality | Moderate | High (captures query-doc interaction) |
| Use case | Large-scale recall (Top-20–100) | Precision reranking of small set (Top-10–30) |

### Implementation Highlights

```python
pairs = [(query, c.get("text", "")) for c in candidates]
scores = scorer.predict(pairs).tolist()   # batch inference — 10× faster than one-by-one
ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
```

- **Lazy loading**: model is loaded only on first call.
- **Failure fallback**: any exception silently returns the original candidate list.
- **Default model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` — lightweight, runs on CPU.

---

## D4.3 LLM Reranker: LLM-Based Reranking and Prompt Design

**Use cases**: no local GPU; need to understand complex instructions (e.g. "prefer results with code examples"); small candidate set (≤ 20).

**Prompt design** (`config/prompts/rerank.txt`): require the LLM to output strictly structured JSON (`ranked_ids`). If the output does not match the schema, raise a readable error and fall back to the RRF order.

---

## D4.4 Integration Position and Effect Analysis

```
HybridSearch  →  Fusion Top-30  →  Core Reranker  →  Top-10  →  ResponseBuilder
```

The **Core layer `reranker.py`** is the orchestration wrapper:
1. Receives `List[RetrievalResult]` from Fusion.
2. Converts to `List[Dict]` for the Libs reranker interface.
3. Calls the backend.
4. On timeout/exception: catches it, marks `fallback=True`, returns the Fusion ordering.

**Two-stage architecture value**: coarse recall (low-cost, high-coverage) → fine reranking (high-cost, high-precision). Top-K precision improves significantly without hurting overall latency because the expensive step only runs on a small candidate set.
