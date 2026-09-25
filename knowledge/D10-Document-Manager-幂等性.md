# D10 — Document Manager & 幂等性

> 对应源码：`src/ingestion/document_manager.py`, `src/libs/loader/file_integrity.py`, `src/ingestion/pipeline.py`, `src/ingestion/storage/`

---

## D10.1 文档去重：Hash 计算与重复检测机制

### SHA256 文件级去重

**触发时机**：每次摄取前，Pipeline 第一步即执行完整性检查。

```python
class SQLiteIntegrityChecker:
    def compute_sha256(self, path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def should_skip(self, file_hash: str) -> bool:
        # 查 ingestion_history.db，hash 存在且 status='success' → True
        return row is not None and row["status"] == "success"
```

### 去重决策树

```
调用 Pipeline.run(file_path)
    ↓
compute_sha256(file_path)
    ↓
should_skip(hash)?
    YES → 直接返回 {skipped: True}（零成本，不调任何 LLM）
    NO  → 继续后续所有步骤
    ↓
摄取完成后 mark_success(hash, path)
```

### 存储结构（SQLite WAL 模式）

```sql
-- data/db/ingestion_history.db
CREATE TABLE ingestion_history (
    file_hash     TEXT PRIMARY KEY,
    file_path     TEXT NOT NULL,
    status        TEXT NOT NULL,   -- 'success' | 'failed'
    processed_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    error_msg     TEXT
);
```

**WAL 模式**（Write-Ahead Logging）的意义：支持多进程并发安全读写，Dashboard 和 CLI 同时操作不会产生锁冲突。

### 文件级 vs Chunk 级去重

| 层次 | 机制 | 触发时机 |
|------|------|---------|
| 文件级 | SHA256 哈希 | Pipeline 第 1 步（最粗粒度，最高效） |
| Chunk 级 | `chunk_id = hash(source + index + content)` | Upsert 时（Chroma 幂等写入） |

两层去重互补：文件未变时文件级去重跳过全部处理；文件有小改动时 Chunk 级去重复用未变 chunk 的向量。

---

## D10.2 增量 Ingestion：幂等性保证与文档更新策略

### 什么是幂等性

同一文档无论被摄取多少次，向量库中都只有一份最新数据，不会堆积重复向量。

### 三层幂等性保证

**第 1 层：文件 SHA256**
```
文件内容未变 → SHA256 相同 → should_skip=True → 直接返回，零副作用
```

**第 2 层：Chunk ID 确定性**
```python
chunk_id = f"{doc_id}_{index:04d}_{content_hash[:8]}"
# 相同 Document + 相同切分参数 → 相同的 chunk_id 序列
```

**第 3 层：Upsert 语义**
```python
# ChromaDB.upsert()：已存在 ID → 更新，不存在 → 插入
# 重复摄取不增加记录数，只更新内容
chroma.upsert(ids=[chunk.id], documents=[chunk.text], metadatas=[chunk.metadata], ...)
```

### 文档更新策略

当文档内容变更时（SHA256 不同），`force=True` 触发完整重新摄取：

```bash
python scripts/ingest.py --path guide.pdf --force  # 强制重新摄取
```

流程：
1. `force=True` 跳过 `should_skip` 检查
2. 重新切分、重新编码
3. 新的 chunk_id（内容哈希变了）Upsert 到 Chroma
4. 旧 chunk（已被新内容替代的 chunk）需手动清理（通过 DocumentManager.delete 再重新 ingest）

---

## D10.3 Collection 管理：集合元数据关联与文档生命周期

### Collection 的概念

Collection 是知识库中的逻辑分组（类似数据库中的 Table），例如：

```
default/         通用文档
  ├── config_guide.pdf
  ├── setup_guide.pdf
  └── rag_concepts.pdf
company-internal/  公司内部文档
  └── handbook.pdf
```

每个 chunk 的 `metadata["collection"]` 记录其所属集合，查询时可通过 `collection` 参数限定范围。

### DocumentManager：跨存储协调层

```python
class DocumentManager:
    def __init__(self, chroma_store, bm25_indexer, image_storage, file_integrity):
        ...

    def list_documents(self, collection=None) -> List[DocumentInfo]:
        """列出已摄入文档及统计信息（chunk 数、图片数）"""

    def delete_document(self, source_path: str, collection: str) -> DeleteResult:
        """协调删除跨四个存储的关联数据"""

    def get_collection_stats(self, collection=None) -> CollectionStats:
        """返回集合级统计（文档数、chunk 数、图片数）"""
```

### delete_document 四步协调删除

```python
def delete_document(self, source_path, collection) -> DeleteResult:
    deleted_chunks = 0
    deleted_images = 0

    # 1. ChromaDB：按 metadata.source 批量删除所有 chunk 向量
    deleted_chunks = self._chroma.delete_by_metadata({"source": source_path})

    # 2. BM25 索引：移除该文档的倒排索引条目
    self._bm25.remove_document(source_path)

    # 3. ImageStorage：删除关联图片文件
    deleted_images = self._images.delete_by_source(source_path)

    # 4. FileIntegrity：移除 hash 记录（使文件可重新摄取）
    self._integrity.remove_record_by_path(source_path)

    return DeleteResult(success=True, deleted_chunks=deleted_chunks,
                        deleted_images=deleted_images, ...)
```

**删除后**，文件的 SHA256 记录也被清除，下次 `ingest.py` 时会被视为新文件重新处理。

---

## D10.4 文档状态追踪：已入库 / 待更新 / 已删除的状态流转

### 状态定义

| 状态 | 含义 | 在 ingestion_history.db 的记录 |
|------|------|-------------------------------|
| **未入库** | 文件从未被摄取 | 无记录 |
| **已入库** | 摄取成功，内容在向量库中 | status='success' |
| **处理失败** | 某步骤抛异常，数据可能不完整 | status='failed' + error_msg |
| **已删除** | 通过 DocumentManager 删除 | 记录被移除（恢复为未入库状态） |

### 状态流转图

```
               ingest()
未入库 ─────────────────────────────→ 已入库
  ↑                 ↓ 异常            ↓ delete_document()
  │           处理失败               已删除
  │                 ↓ force=True        ↓ （记录删除）
  └─────────────────────────────────────┘
              再次 ingest()
```

### Pipeline 状态写入时机

```python
# 成功时：摄取完成后写入
self._integrity.mark_success(file_hash, abs_path)

# 失败时：PipelineError 捕获后写入
except PipelineError:
    if file_hash:
        self._integrity.mark_failed(file_hash, "pipeline_error")
    raise
```

**失败状态的价值**：Dashboard 可以展示"处理失败的文件列表"，方便用户排查并重新摄取。`force=True` 可强制重试失败文件，无需手动清理数据库记录。

### 状态查询接口

```python
# FileIntegrityChecker 提供的查询接口
checker.should_skip(file_hash)       # 是否已成功入库
checker.list_processed()             # 列出所有处理记录（含失败）
checker.remove_record(file_hash)     # 删除记录（配合 DocumentManager.delete）
checker.remove_record_by_path(path)  # 按路径删除（DocumentManager 使用）
```

### 幂等性的工程价值总结

| 场景 | 无幂等性 | 有幂等性 |
|------|---------|---------|
| 重复运行 ingest | 向量库堆积重复数据 | 自动跳过，零副作用 |
| Pipeline 中途失败 | 部分数据写入，状态不一致 | 失败记录标记，可安全重试 |
| 文档内容更新 | 旧版本和新版本同时存在 | Upsert 覆盖，始终最新 |
| 多进程并发摄取 | 并发写入冲突 | WAL 模式 SQLite 保证安全 |

---

# D10 — Document Manager & Idempotency (English)

> Source files: `src/ingestion/document_manager.py`, `src/libs/loader/file_integrity.py`, `src/ingestion/pipeline.py`, `src/ingestion/storage/`

---

## D10.1 Document Deduplication: Hash Computation and Duplicate Detection

### SHA256 File-Level Deduplication

Triggered as the **first step** of every pipeline run, before any parsing or LLM calls.

```python
class SQLiteIntegrityChecker:
    def compute_sha256(self, path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def should_skip(self, file_hash: str) -> bool:
        # query ingestion_history.db: hash exists AND status='success' → True
```

### Decision Tree

```
pipeline.run(file_path)
    ↓
compute_sha256(file_path)
    ↓
should_skip(hash)?
    YES → return {skipped: True}   (zero cost — no LLM, no vector recomputation)
    NO  → proceed through all 7 stages
    ↓
on success: mark_success(hash, path)
```

### Storage Schema (SQLite WAL Mode)

```sql
-- data/db/ingestion_history.db
CREATE TABLE ingestion_history (
    file_hash     TEXT PRIMARY KEY,
    file_path     TEXT NOT NULL,
    status        TEXT NOT NULL,   -- 'success' | 'failed'
    processed_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    error_msg     TEXT
);
```

**WAL (Write-Ahead Logging)** enables safe concurrent reads and writes — the Dashboard and CLI can operate simultaneously without lock contention.

### Two-Level Deduplication

| Level | Mechanism | Granularity |
|-------|-----------|-------------|
| File-level | SHA256 hash | Coarsest — most efficient, skips everything |
| Chunk-level | `chunk_id = hash(source + index + content[:8])` | Fine-grained — reuses unchanged chunks' vectors even when a file partially changes |

---

## D10.2 Incremental Ingestion: Idempotency Guarantees and Update Strategy

### What Idempotency Means Here

No matter how many times the same document is ingested, the vector store contains exactly **one copy of the latest data** — never accumulated duplicates.

### Three Layers of Idempotency

**Layer 1 — File SHA256**:
```
File content unchanged → same SHA256 → should_skip=True → return immediately, no side effects
```

**Layer 2 — Deterministic Chunk IDs**:
```python
chunk_id = f"{doc_id}_{index:04d}_{content_hash[:8]}"
# same Document + same splitter config → same chunk_id sequence every time
```

**Layer 3 — Upsert Semantics**:
```python
# ChromaDB upsert: existing ID → update; new ID → insert
# Repeated ingestion never increases the record count; only updates content
chroma.upsert(ids=[chunk.id], documents=[chunk.text], metadatas=[chunk.metadata])
```

### Document Update Strategy

When file content changes (different SHA256), use `force=True`:

```bash
python scripts/ingest.py --path guide.pdf --force
```

Flow: `force=True` bypasses `should_skip` → re-split → re-encode → new chunk IDs (content hash changed) → Chroma upsert replaces old vectors → `mark_success(new_hash)`.

---

## D10.3 Collection Management: Metadata Association and Document Lifecycle

### What a Collection Is

A collection is a logical grouping in the knowledge base (analogous to a database table):

```
default/            general documents
  ├── config_guide.pdf
  └── setup_guide.pdf
company-internal/   confidential documents
  └── handbook.pdf
```

Every chunk's `metadata["collection"]` records its collection; queries can pass a `collection` filter to restrict the search scope.

### DocumentManager: Cross-Storage Coordination Layer

```python
class DocumentManager:
    def list_documents(self, collection=None) -> List[DocumentInfo]:
        """List ingested documents with stats (chunk count, image count)"""

    def delete_document(self, source_path: str, collection: str) -> DeleteResult:
        """Coordinate deletion across all four storage backends"""

    def get_collection_stats(self, collection=None) -> CollectionStats:
        """Collection-level statistics (doc count, chunk count, image count)"""
```

### `delete_document`: Four-Step Coordinated Deletion

```python
def delete_document(self, source_path, collection) -> DeleteResult:
    # 1. ChromaDB — batch-delete all chunk vectors by metadata.source
    deleted_chunks = self._chroma.delete_by_metadata({"source": source_path})

    # 2. BM25 index — remove inverted index entries for this document
    self._bm25.remove_document(source_path)

    # 3. ImageStorage — delete associated image files
    deleted_images = self._images.delete_by_source(source_path)

    # 4. FileIntegrity — remove the hash record (makes file re-ingestible)
    self._integrity.remove_record_by_path(source_path)

    return DeleteResult(success=True, deleted_chunks=deleted_chunks, ...)
```

After deletion, the file's SHA256 record is gone — the next `ingest.py` run treats it as a new file and processes it from scratch.

---

## D10.4 Document State Tracking: State Transitions

### State Definitions

| State | Meaning | In `ingestion_history.db` |
|-------|---------|--------------------------|
| **Never ingested** | File has never been processed | No record |
| **Ingested** | Successfully processed; content is in the vector store | `status='success'` |
| **Failed** | A stage threw an exception; data may be incomplete | `status='failed'` + `error_msg` |
| **Deleted** | Removed via `DocumentManager.delete_document` | Record deleted → back to "never ingested" |

### State Transition Diagram

```
                ingest()
Never ingested ──────────────────────────────→ Ingested
      ↑                  ↓ exception               ↓ delete_document()
      │              Failed                      Deleted
      │                  ↓ force=True               ↓ (record removed)
      └───────────────────────────────────────────┘
                  ingest() again
```

### Pipeline State Write Points

```python
# On success — written after all stages complete
self._integrity.mark_success(file_hash, abs_path)

# On failure — written when PipelineError is caught
except PipelineError:
    if file_hash:
        self._integrity.mark_failed(file_hash, "pipeline_error")
    raise
```

The **failed state** lets the Dashboard list "files that failed to ingest" for easy troubleshooting. `force=True` retries a failed file without needing to manually clean up the database.

### Idempotency Engineering Value Summary

| Scenario | Without idempotency | With idempotency |
|----------|--------------------|--------------------|
| Run `ingest` twice | Duplicate vectors accumulate | Auto-skip; zero side effects |
| Pipeline fails mid-way | Partial data written; inconsistent state | Failure marked; safe to retry |
| File content updated | Old and new versions coexist in the vector store | Upsert replaces; always latest |
| Concurrent multi-process ingestion | Write conflicts | SQLite WAL mode guarantees safety |
