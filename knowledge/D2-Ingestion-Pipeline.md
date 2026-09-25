# D2 — Ingestion Pipeline

> 对应源码：`src/ingestion/pipeline.py`, `src/ingestion/chunking/`, `src/ingestion/transform/`, `src/ingestion/embedding/`, `src/ingestion/storage/`, `src/libs/loader/`, `src/libs/splitter/`

---

## D2.1 Pipeline 整体流程：从文档加载到向量存储的阶段设计

### 七步流水线

```
文件路径
  ↓ 1. integrity   SHA256 检查，已存在则跳过（零成本增量）
  ↓ 2. load        PdfLoader → Document（text=Markdown，metadata 含 images[]）
  ↓ 3. split       DocumentChunker → Chunk[]（ID 生成 + 图片引用分发）
  ↓ 4. transform   ChunkRefiner → MetadataEnricher → ImageCaptioner（LLM 增强链）
  ↓ 5. encode      BatchProcessor → Dense Vector + Sparse Vector（双路编码）
  ↓ 6. upsert      VectorUpserter（Chroma） + BM25Indexer（倒排索引）
  ↓ 7. store_images ImageStorage → 文件系统 + SQLite 索引
  ↓ 完成，mark_success(file_hash)
```

### IngestionPipeline 核心接口

```python
class IngestionPipeline:
    def run(self, file_path: str, force: bool = False,
            trace: TraceContext = None) -> dict:
        # 返回：{skipped, chunk_count, image_count, trace_id}

    def run_batch(self, file_paths: List[str], force: bool = False) -> List[dict]:
        # 对多文件依次执行，单个失败不中断整体
```

### 关键设计

- **PipelineError** 携带 `stage` 名，失败时精确知道卡在哪步
- **force=True** 跳过完整性检查，用于强制重新摄取（如更新文档）
- **on_progress 回调**：签名 `(stage_name, current, total)`，Dashboard 用于实时显示进度条

---

## D2.2 Chunking 策略：RecursiveSplitter 的分割逻辑与参数调优

### 为什么用 RecursiveCharacterTextSplitter

PDF → MarkItDown → 标准 Markdown 文本。Recursive 按层级分隔符切分：
```
优先级：\n\n（段落）→ \n（换行）→ 空格 → 单字符
```
在 `chunk_size` 限制内尽量保留完整段落，比定长切分语义更完整。

### DocumentChunker：适配器层的六个增值

`libs/splitter` 只做 `str → List[str]`，DocumentChunker 在此基础上补全业务逻辑：

| 职责 | 实现 |
|------|------|
| Chunk ID 生成 | `{doc_id}_{index:04d}_{content_hash[:8]}`，确定性可重复 |
| 元数据继承 | 从 Document.metadata 复制 source_path、doc_type 等 |
| chunk_index 字段 | 记录序号，支持排序和定位 |
| source_ref 溯源 | `Chunk.doc_id = Document.id` |
| 图片引用分发 | 扫描 `[IMAGE: id]` 占位符，仅把该 chunk 引用的图片子集写入 `chunk.metadata["images"]` |
| 类型转换 | `List[str]` → `List[Chunk]` |

### 图片引用分发的重要性

```python
# DocumentChunker 中的关键逻辑
_IMAGE_PLACEHOLDER_RE = re.compile(r"\[IMAGE:\s*([^\]]+)\]")

referenced_ids = _IMAGE_PLACEHOLDER_RE.findall(text)
if referenced_ids:
    metadata["image_refs"] = referenced_ids
    metadata["images"] = [doc_images[id] for id in referenced_ids if id in doc_images]
```

如果简单地把整个文档的 `images` 列表复制给每个 chunk，ImageCaptioner 就无法知道哪个 chunk 对应哪张图，会对每个 chunk 重复处理所有图片。

### 参数调优

| 参数 | 当前值 | 作用 |
|------|--------|------|
| `chunk_size` | 1000 | 最大字符数，控制 chunk 粒度（太大=检索噪音多，太小=语义不完整） |
| `chunk_overlap` | 200 | 相邻 chunk 重叠字符数，保证跨 chunk 的语义不断裂 |

---

## D2.3 Transform 链：ChunkRefiner / MetadataEnricher / ImageCaptioner 的职责与执行顺序

### Transform 基类设计

```python
class BaseTransform(ABC):
    @abstractmethod
    def transform(self, chunks: List[Chunk], trace: TraceContext = None) -> List[Chunk]:
        pass
```

所有 Transform 设计为**幂等、原子**操作：单个 chunk 失败不影响其他 chunk，失败降级（fallback）不阻塞整条 pipeline。

### 三步 Transform 链的执行顺序

```
chunks
  → ChunkRefiner      规则去噪（页眉页脚/多余空白）+ 可选 LLM 重写
  → MetadataEnricher  注入 title/summary/tags（规则 + 可选 LLM）
  → ImageCaptioner    调用 Vision LLM 生成图片描述，注入 chunk.text
```

顺序有依赖关系：必须先清洗噪声（Refiner）再做语义增强（Enricher），否则 LLM 会基于噪声内容生成低质量摘要。

### ChunkRefiner

- **规则模式**：正则匹配去除页眉页脚、多余空白、HTML 注释、格式标记
- **LLM 模式**（`use_llm: true`）：调用 LLM 基于 `config/prompts/chunk_refinement.txt` 对文本二次加工
- **降级**：LLM 失败时回退规则结果，metadata 标记 `refined_by: "rule"` 或 `"llm"`

### MetadataEnricher

- 为每个 chunk 生成 `title`（精准小标题）、`summary`（内容摘要）、`tags`（主题标签）
- 这些字段写入 `chunk.metadata`，支持后续的混合检索与精确过滤
- 同样支持规则 + 可选 LLM 两种模式

### ImageCaptioner

- 检查 `chunk.metadata.get("image_refs")`，若不为空则调用 Vision LLM
- 生成的 caption 文本**注入到 chunk.text** 正文（替换 `[IMAGE: id]` 占位符）
- 注入正文而非 metadata 的原因：让描述被 Embedding 覆盖，直接可被检索到
- 禁用/失败降级：保留 `image_refs`，标记 `has_unprocessed_images: true`

---

## D2.4 Embedding 编码：Dense/Sparse 双编码与 BatchProcessor 批处理

### 双路编码目的

| 编码类型 | 方法 | 捕捉能力 | 用于 |
|---------|------|---------|------|
| Dense | `text-embedding-3-small` 等模型 | 语义关联（"苹果公司" ↔ "Apple Inc"） | 余弦相似度检索 |
| Sparse | BM25 词频统计 | 精确关键词匹配（函数名、型号、专有名词） | 倒排索引检索 |

两路编码结果都写入 `chunk.metadata`（dense_vector / sparse_vector），然后一起 upsert 到存储层。

### BatchProcessor 分批策略

```python
class BatchProcessor:
    def __init__(self, settings, batch_size=32):
        self._dense = DenseEncoder(settings)
        self._sparse = SparseEncoder(settings)
        self._batch_size = batch_size

    def encode_all(self, chunks, trace=None) -> List[Chunk]:
        for batch in chunks[0::batch_size]:
            batch = self._dense.encode(batch, trace)
            batch = self._sparse.encode(batch, trace)
            results.extend(batch)
        return results
```

- **batch_size=32** 是 API 限流和内存之间的经验平衡点
- 同一 batch 先 Dense 再 Sparse，顺序稳定，输出与输入一一对应，上层不需要做 ID 匹配
- 每批耗时通过 `trace.record_stage("batch_encode", ...)` 记录

### SparseEncoder 原理

对 chunk 文本做分词（去停用词），统计词频（TF），输出 `{term: tf}` 字典，作为 BM25Indexer 的输入。

---

## D2.5 存储层：VectorUpserter / BM25Indexer / ImageStorage 三类存储协同

### 四类存储各司其职

| 存储 | 目录 | 存储内容 | 用于 |
|------|------|---------|------|
| Chroma（向量库） | `data/db/chroma/` | Dense Vector + Chunk 文本 + Metadata | Dense 召回 |
| BM25 索引 | `data/db/bm25/postings.json` | 倒排索引 + IDF 统计 | Sparse 召回 |
| ImageStorage | `data/images/{collection}/` + `data/db/image_index.db` | 原始图片文件 + image_id→path 映射 | 多模态返回 |
| FileIntegrity | `data/db/ingestion_history.db` | SHA256 → status | 增量摄取跳过 |

### BM25Indexer 核心算法

```
IDF(term) = log((N - df + 0.5) / (df + 0.5) + 1)   # +1 防止负 IDF

BM25 查询打分：
score(q, d) = Σ IDF(t) * tf(t,d)*(k1+1) / (tf(t,d) + k1*(1-b+b*|d|/avgdl))
其中 k1=1.5, b=0.75
```

参数语义：
- `k1=1.5`：词频饱和速度（越大则高频词优势越弱）
- `b=0.75`：长文档归一化强度（1.0=完全归一化，0=不归一化）

### VectorUpserter 幂等性

```python
# chunk_id 由内容哈希生成，相同内容 → 相同 ID → Upsert 不产生重复
chunk_id = f"{doc_id}_{index:04d}_{content_hash[:8]}"
vector_store.upsert(records)  # Upsert 语义：已存在则更新，不存在则插入
```

幂等性保证：同一文档重复摄取，向量库中只有一份最新数据，不会堆积重复向量。

---

# D2 — Ingestion Pipeline (English)

> Source files: `src/ingestion/pipeline.py`, `src/ingestion/chunking/`, `src/ingestion/transform/`, `src/ingestion/embedding/`, `src/ingestion/storage/`, `src/libs/loader/`, `src/libs/splitter/`

---

## D2.1 Pipeline Overall Flow: Stage Design from Document Loading to Vector Storage

### Seven-Step Pipeline

```
File path
  ↓ 1. integrity    SHA256 check — skip if already ingested (zero-cost increment)
  ↓ 2. load         PdfLoader → Document (text=Markdown, metadata includes images[])
  ↓ 3. split        DocumentChunker → Chunk[] (ID generation + image ref distribution)
  ↓ 4. transform    ChunkRefiner → MetadataEnricher → ImageCaptioner (LLM augmentation chain)
  ↓ 5. encode       BatchProcessor → Dense Vector + Sparse Vector (dual-path encoding)
  ↓ 6. upsert       VectorUpserter (Chroma) + BM25Indexer (inverted index)
  ↓ 7. store_images ImageStorage → filesystem + SQLite index
  ↓ Done — mark_success(file_hash)
```

### Key Design Points

- `PipelineError` carries a `stage` name so you know exactly which step failed.
- `force=True` skips the integrity check to force re-ingestion (e.g. when updating a document).
- `on_progress` callback: signature `(stage_name, current, total)` — used by the Dashboard to show a real-time progress bar.

---

## D2.2 Chunking Strategy: RecursiveSplitter Logic and Parameter Tuning

**Why Markdown as intermediate format**: `RecursiveCharacterTextSplitter` splits by hierarchical delimiters (`\n\n` → `\n` → space → char), preserving complete paragraphs within the `chunk_size` limit — semantically much better than fixed-length splitting.

### DocumentChunker: Six Value-Adds Over `libs/splitter`

| Responsibility | Implementation |
|----------------|----------------|
| Chunk ID generation | `{doc_id}_{index:04d}_{content_hash[:8]}` — deterministic, repeatable |
| Metadata inheritance | Copies `source_path`, `doc_type`, etc. from `Document.metadata` |
| `chunk_index` field | Records position for sorting and lookup |
| `source_ref` traceability | `Chunk.doc_id = Document.id` |
| Image ref distribution | Scans `[IMAGE: id]` placeholders; writes only the subset of images referenced by that chunk into `chunk.metadata["images"]` |
| Type conversion | `List[str]` → `List[Chunk]` |

### Parameter Tuning

| Param | Default | Effect |
|-------|---------|--------|
| `chunk_size` | 1000 | Max chars per chunk (too large = retrieval noise; too small = broken semantics) |
| `chunk_overlap` | 200 | Overlap chars between adjacent chunks to avoid cross-chunk context breaks |

---

## D2.3 Transform Chain: ChunkRefiner / MetadataEnricher / ImageCaptioner

All transforms extend `BaseTransform` — atomic, idempotent; a single chunk failure does not block the rest.

**Execution order matters**: clean noise first (Refiner), then do semantic enrichment (Enricher) — otherwise LLM generates low-quality summaries from noisy text.

| Transform | Mode | Fallback |
|-----------|------|----------|
| `ChunkRefiner` | Rule-based de-noise + optional LLM rewrite | LLM failure → rule result; marks `refined_by: "rule"` |
| `MetadataEnricher` | Generates `title`/`summary`/`tags` | Rule-based fallback |
| `ImageCaptioner` | Calls Vision LLM for each `image_refs` item; injects caption into `chunk.text` | Marks `has_unprocessed_images: True`; never blocks pipeline |

**Captions are injected into `chunk.text` (not just metadata)** so they are covered by the embedding and become directly searchable.

---

## D2.4 Embedding: Dense/Sparse Dual Encoding and BatchProcessor

| Encoding | Method | Captures |
|----------|--------|----------|
| Dense | Embedding model (BGE-M3, text-embedding-3…) | Semantic associations |
| Sparse | BM25 term-frequency statistics | Exact keyword matches |

`BatchProcessor` slices the chunk list into batches of 32, runs Dense then Sparse on each batch in order, and extends results sequentially — output order always matches input order, no ID matching needed.

---

## D2.5 Storage Layer: VectorUpserter / BM25Indexer / ImageStorage Coordination

| Storage | Location | Content | Used for |
|---------|----------|---------|----------|
| Chroma (vector DB) | `data/db/chroma/` | Dense vector + chunk text + metadata | Dense retrieval |
| BM25 index | `data/db/bm25/postings.json` | Inverted index + IDF stats | Sparse retrieval |
| ImageStorage | `data/images/` + `data/db/image_index.db` | Raw image files + id→path mapping | Multimodal response |
| FileIntegrity | `data/db/ingestion_history.db` | SHA256 → status | Incremental skip |

**BM25 IDF formula**: `log((N - df + 0.5) / (df + 0.5) + 1)` (Robertson-Walker variant; +1 prevents negative IDF)

**VectorUpserter idempotency**: `chunk_id = hash(source + index + content)` — same content always produces the same ID, and Chroma's `upsert` semantics ensure no duplicates accumulate across repeated runs.
