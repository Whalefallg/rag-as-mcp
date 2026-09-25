# Correctness Hardening Notes

> 当前分支：`fix/rag-correctness-hardening`  
> 目的：记录 2026-09 correctness hardening 后的**实际运行契约**。  
> 本文优先描述当前源码行为；`D1`–`D10` 仍可作为架构背景材料，但若细节冲突，以源码和本文为准。

## 1. Collection isolation

Collection 是物理命名空间，不只是 metadata filter。

查询链路：

```text
MCP collection
  -> HybridSearch
     -> DenseRetriever -> Chroma collection
     -> SparseRetriever -> BM25 collection -> same Chroma collection hydrate
  -> Fusion
  -> Rerank
```

BM25 使用 collection-aware key，默认 collection 保留旧索引兼容逻辑。图片索引同样以
`(image_id, collection)` 为唯一身份，避免两个 collection 中同名图片互相覆盖。

## 2. Document replacement semantics

文档更新不是“先删旧数据再写新数据”，而是可重试收敛的 replacement：

```text
load/split/transform/encode
  -> enumerate old chunk ids for source + collection
  -> upsert new vector chunks
  -> update BM25
  -> delete stale BM25 chunk ids
  -> delete stale vector chunk ids
  -> save new images
  -> delete stale image ids
  -> mark integrity success
```

设计目标是：写新版本失败时尽量保留旧版本，而不是为了伪装事务性先破坏已有数据。

`FileIntegrity` 的成功记录按 `(file_hash, collection)` 隔离；同一路径在同一 collection
成功写入新版本后，会清理该路径的旧成功 hash，避免 `v1 -> v2 -> v1` 被错误跳过。

### 仍然存在的边界

这不是跨 Chroma/BM25/filesystem/SQLite 的真正 ACID transaction。异常时可能暂时存在
“新旧记录同时存在”的中间状态，设计依赖重试收敛。对于已经发生的、只存在于 BM25
而 Chroma 中没有对应 chunk id 的历史孤儿记录，replacement 不承诺自动修复。

## 3. Runtime capability contract

配置校验不再维护一份与实现分离的“想象中的支持列表”。

可插拔能力直接读取 runtime registry：

- LLM provider
- Embedding provider
- VectorStore backend
- Splitter method
- Reranker backend
- Fusion algorithm

因此未注册的 `qdrant`、`pinecone`、`semantic`、`fixed` 等值不会通过启动校验。
当前 sparse backend 的真实实现只有 `bm25`，fusion 的真实实现只有 `rrf`。

`rerank.top_m` 已进入运行路径：启用 rerank 时，HybridSearch 至少取 `top_m`
候选，再由 reranker 截断到用户请求的 `top_k`；关闭 rerank 时不会额外扩大候选池。

## 4. Deletion semantics

`DocumentManager.delete_document()` 是 **best-effort coordinated deletion**，不是原子事务。

它依次尝试清理：

1. VectorStore
2. BM25
3. ImageStorage
4. FileIntegrity

返回值中的 `details` 明确包含：

```text
mode = best_effort
successful_stores = [...]
failed_stores = [...]
```

部分失败时 `success=False`，并保留已成功删除步骤的事实，调用方可以据此决定是否重试。

## 5. Query trace ownership

Query trace 的 owner 是 `query_knowledge_hub`。

一次 MCP tool call 创建一个 `TraceContext(trace_type="query")`，同一个对象贯穿：

```text
MCP tool
  -> HybridSearch
     -> Dense retrieval
     -> Sparse retrieval
     -> Fusion
  -> Rerank
  -> Multimodal assembly
  -> Response build
```

tool 使用 `finally` 收集 trace，因此组件初始化失败、检索失败、rerank 降级等路径也会留下记录。

常见 metadata：

- `user_query`
- `collection`
- `top_k`
- `candidate_k`
- `rerank_backend`
- `result_count`
- `status`

## 6. Test and CI contract

测试目录就是 CI 分层边界：

```text
tests/unit/         fast logic/regression tests
tests/integration/  cross-module contracts
tests/e2e/          process/UI/evaluation flow smoke tests
```

CI 策略：

```text
unit         Python 3.10 + 3.12
integration  Python 3.12
e2e          Python 3.12
```

本地命令：

```bash
python -m pytest tests/unit -q
python -m pytest tests/integration -q
python -m pytest tests/e2e -q
python -m pytest -q
```

测试 helper 不使用 `Test*` 类名，避免被 pytest 当成测试类收集。
`PytestCollectionWarning` 被升级为错误，防止同类 warning 再次悄悄进入 CI。
