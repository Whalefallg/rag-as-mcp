# RAG as MCP

一个面向生产的 Agentic Hybrid RAG 引擎与 MCP Knowledge Service：在可靠的 Dense + BM25 + RRF 基础上提供自适应查询规划、证据驱动的纠正检索、有界执行、评估与链路追踪。

**中文** · [English](#english)

---

# 中文

## 项目简介

RAG as MCP 将 Retrieval-Augmented Generation 的核心链路拆分为可替换、可测试的工程组件，并通过 **MCP stdio server** 提供标准化知识检索接口。

项目重点不只是“让 RAG 跑起来”，而是围绕以下工程问题建立清晰边界：

- 文档如何稳定摄取并按 collection 隔离
- Dense 与 Sparse retrieval 如何融合
- rerank 如何作为可选阶段插拔
- 查询链路如何端到端追踪
- 检索质量如何离线评估和回归
- MCP 服务如何被其他 Agent 系统稳定调用
- 容器如何通过 CI/CD 可重复构建、验证与发布

项目可独立运行，也可作为 [`Incident Lifecycle Copilot`](https://github.com/Whalefallg/incident-lifecycle-copilot) 的外部知识检索后端。

## 核心能力

- **Adaptive Query Planning**：`auto` 路由让简单查询走低延迟 classic path，复杂查询进入 agentic path。
- **Multi-Query / Query Decomposition**：复杂问题拆解后复用既有 HybridSearch，并用 RRF 去重融合。
- **Evidence-Guided Corrective Retrieval**：证据不足时改写并在预算内重试。
- **Bounded Agent Execution**：显式 iteration、subquery 和 candidate budget，终止后返回最佳证据。
- **Agentic Trace & Offline Evaluation**：记录计划、评分、改写、终止与降级，并比较 classic 和 agentic。
- **Document Ingestion Pipeline**：PDF 加载、切分、转换、Embedding 与持久化。
- **Hybrid Retrieval**：Dense retrieval + BM25 sparse retrieval。
- **RRF Fusion**：使用 Reciprocal Rank Fusion 合并不同召回通道。
- **Pluggable Reranking**：支持可选 Cross-Encoder / LLM rerank。
- **Collection Isolation**：Dense、BM25、ImageStorage 与 FileIntegrity 数据按 collection 隔离。
- **MCP Serving**：通过 stdio JSON-RPC 暴露知识库查询工具。
- **Observability**：单一 `TraceContext` 贯穿 retrieval、fusion、rerank、multimodal 与 response 阶段。
- **Offline Evaluation**：支持本地评估与可选 Ragas 评估依赖。
- **Engineering Validation**：Unit / Integration / E2E 分层测试，Python 3.10 / 3.12 CI。
- **Container Delivery**：Git tag 驱动 Docker build、MCP smoke test 与 GHCR 发布。

## 架构

```text
MCP Query -> Adaptive Router -> Classic --------------------+
                         \-> Agentic -> Analyze -> Plan      |
                                      -> Decompose ----------+
                                                             v
                                           Dense + BM25 -> RRF
                                                             |
                                                    Evidence Grader
                                                     /          \
                                                 enough       rewrite
                                                     \          /
                                                      Final Rerank
                                                           |
                                                      MCP Response
```

```text
                         Ingestion
                            |
                            v
PDF -> Loader -> Chunker -> Transform -> Embedding -> ChromaDB
                               |
                               +------------------> BM25 Index
                               |
                               +------------------> File / Image Metadata


                           Query
                            |
                            v
             +--------------+--------------+
             |                             |
             v                             v
      Dense Retrieval                BM25 Retrieval
             |                             |
             +--------------+--------------+
                            |
                            v
                         RRF Fusion
                            |
                            v
                     Optional Reranker
                            |
                            v
                       MCP Response
```

核心模块：

```text
src/ingestion/            文档摄取与存储协调
src/core/query_engine/    查询、混合检索、融合与重排序
src/core/agentic/         自适应分析、计划、评分、改写与有界编排
src/mcp_server/           MCP 协议与 tools
src/observability/        Trace、Dashboard 与评估
```

## 查询链路

典型查询流程：

```text
MCP tools/call
    |
    v
query_knowledge_hub (mode=classic | agentic | auto)
    |
    v
Query Engine
    |
    +---- Dense Retriever
    |
    +---- BM25 Retriever
    |
    v
RRF Fusion
    |
    v
Optional Rerank
    |
    v
Citation / MCP Response
```

混合检索的目标不是简单增加召回通道，而是把语义召回与关键词精确匹配放在同一可评估管线中：

- Dense retrieval 负责语义相似内容
- BM25 对术语、错误码、服务名等 lexical signal 更敏感
- RRF 在不依赖两个检索器分数可比性的情况下融合排序
- reranker 作为独立阶段进一步优化候选顺序

Agentic 模式不是无界 autonomous agent，而是一个 **bounded adaptive retrieval loop**：它有明确的 iteration budget、fallback、termination reason 和 trace。任何智能模块失败都降级到 deterministic 或 classic retrieval，不牺牲服务可用性。

## MCP 工具

当前 MCP Server 暴露以下主要工具：

| Tool | Purpose |
|---|---|
| `query_knowledge_hub` | 对指定 collection 执行知识检索 |
| `list_collections` | 列出可用 collection |
| `get_document_summary` | 获取文档摘要信息 |

Server 使用 **stdout 传输 JSON-RPC**，运行日志写入 **stderr**，避免污染 MCP transport。

启动：

```bash
python main.py
```

支持 stdio MCP 的客户端可使用类似配置：

```json
{
  "mcpServers": {
    "rag-as-mcp": {
      "command": "/absolute/path/to/rag-as-mcp/.venv/bin/python",
      "args": ["/absolute/path/to/rag-as-mcp/main.py"],
      "cwd": "/absolute/path/to/rag-as-mcp",
      "env": {
        "MCP_SETTINGS_PATH": "/absolute/path/to/rag-as-mcp/config/settings.yaml"
      }
    }
  }
}
```

## Correctness 设计

### Collection 隔离

Dense index、BM25 index、ImageStorage 和 FileIntegrity 都按 collection 隔离，避免不同知识域之间发生隐式数据污染。

### 可重试文档更新

文档更新采用“**先写新版本，再清理 stale 数据**”的收敛策略，使中途失败的更新可以安全重试，而不要求跨多个存储实现分布式事务。

### 显式 best-effort 删除

`DocumentManager.delete_document()` 协调多个存储执行删除，但不会伪装成跨存储 ACID transaction。返回结果会明确暴露各 store 的成功与失败状态。

### 端到端 TraceContext

一次 query 创建单一 `TraceContext`，贯穿：

```text
Dense
  -> Sparse
  -> Fusion
  -> Rerank
  -> Multimodal
  -> Response
```

失败路径同样记录 trace，便于定位“召回失败”“fusion 异常”“rerank 异常”或响应阶段问题。

更完整的 correctness 语义见 [`knowledge/CORRECTNESS-HARDENING.md`](knowledge/CORRECTNESS-HARDENING.md)。

## 环境要求

- Python 3.10+
- 在线 LLM / Embedding provider 的 API Key（按需）
- Ollama（仅当选择本地 Ollama provider 时需要）

## 安装

```bash
git clone https://github.com/Whalefallg/rag-as-mcp.git
cd rag-as-mcp

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

cp config/settings.example.yaml config/settings.yaml
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
```

可选依赖：

```bash
python -m pip install -e ".[dashboard]"   # Streamlit Dashboard
python -m pip install -e ".[rerank]"      # Cross-Encoder reranking
python -m pip install -e ".[evaluation]"  # Ragas evaluation
python -m pip install -e ".[all]"         # All optional features
```

## 配置

`config/settings.example.yaml` 是可提交模板。

本地配置：

```text
config/settings.yaml
```

已被 Git 忽略，凭证建议通过环境变量注入：

```bash
export RAG_LLM_API_KEY="your-llm-key"
export RAG_EMBEDDING_API_KEY="your-embedding-key"
export RAG_VISION_API_KEY="your-vision-key"
```

可选：

```bash
export RAG_LLM_AZURE_ENDPOINT="https://example.openai.azure.com/"
export RAG_EMBEDDING_BASE_URL="https://api.example.com/v1"
export MCP_SETTINGS_PATH="/absolute/path/to/settings.yaml"
```

不要将真实密钥写入示例配置或提交到 Git。

## 运行完整流程

生成演示 PDF：

```bash
python scripts/gen_sample_pdfs.py
```

摄取文档：

```bash
python scripts/ingest.py \
  --path data/documents/default \
  --collection default
```

命令行查询：

```bash
python scripts/query.py \
  --query "RRF 如何融合检索结果？" \
  --collection default
```

启动 MCP Server：

```bash
python main.py
```

启动 Dashboard：

```bash
python scripts/start_dashboard.py
```

运行本地检索评估：

```bash
python scripts/evaluate.py --evaluator local
```

## 测试与 CI

项目将测试分为 Unit、Integration 和 E2E 三层：

```bash
python -m pytest tests/unit -q
python -m pytest tests/integration -q
python -m pytest tests/e2e -q
python -m pytest -q
```

CI 策略：

```text
Unit          Python 3.10 + 3.12
Integration   Python 3.12
E2E           Python 3.12
```

`PytestCollectionWarning` 被升级为错误，避免测试 helper 因命名或收集问题被静默忽略。

核心算法与协议测试可以在不调用付费外部模型服务的情况下运行。

## Docker 与持续交付

本地构建：

```bash
docker build -t rag-as-mcp:local .
```

MCP 容器 smoke test：

```bash
python scripts/smoke_mcp_container.py rag-as-mcp:local
```

发布由 Git tag 驱动：

```bash
git tag v0.1.0
git push origin v0.1.0
```

`Release Container` workflow 执行：

```text
Full pytest regression
        |
        v
Build Docker image
        |
        v
MCP stdio smoke test
        |
        v
Push version tag + latest to GHCR
```

发布镜像使用：

```text
ghcr.io/whalefallg/rag-as-mcp:<version>
ghcr.io/whalefallg/rag-as-mcp:latest
```

容器默认可以使用示例配置启动 MCP 协议；实际环境可通过 `MCP_SETTINGS_PATH` 与 volume mount 注入配置和数据。

## 当前限制

- Loader 当前主要面向 PDF，其他文档格式尚未形成完整实现。
- 当前已注册 VectorStore backend 主要为 Chroma。
- 多进程并发写入与分布式部署不是当前设计范围。
- `DocumentManager.delete_document()` 是 best-effort coordination，而不是跨存储 ACID transaction。
- 外部模型的稳定性、限流、价格和输出质量不由本项目控制。
- 测试与评估结果用于验证实现正确性和回归，不等价于生产容量或 SLA 结论。

## 仓库结构

```text
config/          配置模板
data/            本地文档与持久化数据
knowledge/       Correctness / design notes
scripts/         ingest、query、evaluation、dashboard、container smoke test
src/             核心实现
tests/           unit / integration / e2e tests
```

## Roadmap

- 建立基于固定已摄取语料的可重复检索 benchmark
- 扩展更多文档 Loader
- 增加新的 VectorStore 与 Fusion backend
- 完善 MCP client 端到端集成示例
- 持续增强 retrieval evaluation 与 observability

## License

MIT License。详见 [LICENSE](LICENSE)。

---

# English

## Overview

RAG as MCP is a production-oriented Agentic Hybrid RAG engine exposed through MCP. It adds adaptive planning, evidence-guided corrective retrieval, bounded execution, evaluation, and tracing on top of dense + BM25 + RRF retrieval.

The project focuses on the engineering boundaries required to make a RAG pipeline understandable and testable:

- deterministic document ingestion and collection isolation
- hybrid semantic and lexical retrieval
- pluggable fusion and reranking stages
- end-to-end query tracing
- offline retrieval evaluation
- a stable MCP interface for downstream agents
- repeatable testing, container validation, and release automation

It can run as a standalone retrieval service or as the external knowledge backend for [`Incident Lifecycle Copilot`](https://github.com/Whalefallg/incident-lifecycle-copilot).

## Highlights

- **Adaptive query planning** with `classic`, `agentic`, and low-latency `auto` routing.
- **Multi-query and query decomposition** over the existing HybridSearch primitive.
- **Evidence-guided corrective retrieval** with deterministic offline grading and optional LLM assistance.
- **Bounded agent execution** with iteration, subquery, and candidate budgets.
- **Agentic tracing and offline evaluation** including classic-vs-agentic comparison.
- **Document ingestion pipeline** — PDF loading, chunking, transformation, embeddings, and persistence.
- **Hybrid retrieval** — dense retrieval combined with BM25 sparse retrieval.
- **RRF fusion** — Reciprocal Rank Fusion combines independent rankers without requiring directly comparable scores.
- **Pluggable reranking** — optional Cross-Encoder or LLM-based reranking.
- **Collection isolation** — dense, BM25, image, and file-integrity data are isolated by collection.
- **MCP serving** — knowledge tools are exposed through stdio JSON-RPC.
- **Observability** — one `TraceContext` spans retrieval, fusion, reranking, multimodal processing, and response generation.
- **Offline evaluation** — local evaluation with optional Ragas support.
- **Layered testing** — unit, integration, and E2E suites across Python 3.10 and 3.12 CI.
- **Container delivery** — tag-driven Docker build, MCP smoke testing, and GHCR publishing.

## Architecture

```text
MCP Query -> Adaptive Router -> Classic --------------------+
                         \-> Agentic -> Analyze -> Plan      |
                                      -> Decompose ----------+
                                                             v
                                           Dense + BM25 -> RRF
                                                             |
                                                    Evidence Grader
                                                     /          \
                                                 enough       rewrite
                                                     \          /
                                                      Final Rerank
                                                           |
                                                      MCP Response
```

```text
                         Ingestion
                            |
                            v
PDF -> Loader -> Chunker -> Transform -> Embedding -> ChromaDB
                               |
                               +------------------> BM25 Index
                               |
                               +------------------> File / Image Metadata


                           Query
                            |
                            v
             +--------------+--------------+
             |                             |
             v                             v
      Dense Retrieval                BM25 Retrieval
             |                             |
             +--------------+--------------+
                            |
                            v
                         RRF Fusion
                            |
                            v
                     Optional Reranker
                            |
                            v
                       MCP Response
```

Core modules:

```text
src/ingestion/            ingestion and storage coordination
src/core/query_engine/    query processing, hybrid retrieval, fusion, reranking
src/mcp_server/           MCP protocol and tools
src/observability/        traces, dashboard, and evaluation
```

## Query Pipeline

A typical MCP query follows this path:

```text
MCP tools/call
    |
    v
query_knowledge_hub
    |
    v
Query Engine
    |
    +---- Dense Retriever
    |
    +---- BM25 Retriever
    |
    v
RRF Fusion
    |
    v
Optional Rerank
    |
    v
Citation / MCP Response
```

Each retrieval stage addresses a different signal:

- dense retrieval captures semantic similarity
- BM25 is effective for lexical signals such as service names, identifiers, and error codes
- RRF merges rankings without assuming the underlying scores are calibrated
- the optional reranker refines the final candidate order

## MCP Tools

The server currently exposes the following primary tools:

| Tool | Purpose |
|---|---|
| `query_knowledge_hub` | Query a selected knowledge collection |
| `list_collections` | List available collections |
| `get_document_summary` | Return document summary information |

The server writes JSON-RPC messages to **stdout** and operational logs to **stderr**, keeping the MCP transport clean.

Start the server with:

```bash
python main.py
```

Example stdio MCP client configuration:

```json
{
  "mcpServers": {
    "rag-as-mcp": {
      "command": "/absolute/path/to/rag-as-mcp/.venv/bin/python",
      "args": ["/absolute/path/to/rag-as-mcp/main.py"],
      "cwd": "/absolute/path/to/rag-as-mcp",
      "env": {
        "MCP_SETTINGS_PATH": "/absolute/path/to/rag-as-mcp/config/settings.yaml"
      }
    }
  }
}
```

## Correctness Model

### Collection Isolation

Dense indexes, BM25 indexes, image storage, and file-integrity state are separated by collection to avoid implicit cross-domain contamination.

### Retryable Document Updates

Document replacement follows a convergence-oriented strategy: write the new version first, then remove stale state. This keeps updates retryable without pretending that all storage layers participate in one distributed transaction.

### Explicit Best-Effort Deletion

`DocumentManager.delete_document()` coordinates deletion across multiple stores, but it is intentionally not presented as an ACID transaction. The result reports which stores succeeded and which failed.

### End-to-End TraceContext

A query creates one `TraceContext` that spans:

```text
Dense
  -> Sparse
  -> Fusion
  -> Rerank
  -> Multimodal
  -> Response
```

Failure paths are traced as well, making it easier to distinguish retrieval, fusion, reranking, and response-stage failures.

See [`knowledge/CORRECTNESS-HARDENING.md`](knowledge/CORRECTNESS-HARDENING.md) for the detailed correctness semantics.

## Requirements

- Python 3.10+
- API credentials for online LLM or embedding providers when enabled
- Ollama only when using a local Ollama-backed provider

## Installation

```bash
git clone https://github.com/Whalefallg/rag-as-mcp.git
cd rag-as-mcp

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

cp config/settings.example.yaml config/settings.yaml
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Optional extras:

```bash
python -m pip install -e ".[dashboard]"   # Streamlit dashboard
python -m pip install -e ".[rerank]"      # Cross-Encoder reranking
python -m pip install -e ".[evaluation]"  # Ragas evaluation
python -m pip install -e ".[all]"         # All optional features
```

## Configuration

`config/settings.example.yaml` is the committed configuration template.

Create a local configuration at:

```text
config/settings.yaml
```

Credentials should be injected through environment variables:

```bash
export RAG_LLM_API_KEY="your-llm-key"
export RAG_EMBEDDING_API_KEY="your-embedding-key"
export RAG_VISION_API_KEY="your-vision-key"
```

Optional settings:

```bash
export RAG_LLM_AZURE_ENDPOINT="https://example.openai.azure.com/"
export RAG_EMBEDDING_BASE_URL="https://api.example.com/v1"
export MCP_SETTINGS_PATH="/absolute/path/to/settings.yaml"
```

Do not commit real credentials.

## End-to-End Usage

Generate the bundled sample PDFs:

```bash
python scripts/gen_sample_pdfs.py
```

Ingest documents:

```bash
python scripts/ingest.py \
  --path data/documents/default \
  --collection default
```

Run a CLI query:

```bash
python scripts/query.py \
  --query "How does RRF combine retrieval results?" \
  --collection default
```

Start the MCP server:

```bash
python main.py
```

Start the dashboard:

```bash
python scripts/start_dashboard.py
```

Run local retrieval evaluation:

```bash
python scripts/evaluate.py --evaluator local
```

## Testing and CI

Tests are separated into unit, integration, and end-to-end layers:

```bash
python -m pytest tests/unit -q
python -m pytest tests/integration -q
python -m pytest tests/e2e -q
python -m pytest -q
```

Current CI matrix:

```text
Unit          Python 3.10 + 3.12
Integration   Python 3.12
E2E           Python 3.12
```

`PytestCollectionWarning` is promoted to an error so accidentally uncollected tests do not fail silently.

Core algorithm and protocol tests are designed to run without paid external model calls.

## Docker and Continuous Delivery

Build locally:

```bash
docker build -t rag-as-mcp:local .
```

Run the MCP container smoke test:

```bash
python scripts/smoke_mcp_container.py rag-as-mcp:local
```

Releases are driven by Git tags:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The `Release Container` workflow performs:

```text
Full pytest regression
        |
        v
Build Docker image
        |
        v
MCP stdio smoke test
        |
        v
Push version tag + latest to GHCR
```

Published images use:

```text
ghcr.io/whalefallg/rag-as-mcp:<version>
ghcr.io/whalefallg/rag-as-mcp:latest
```

The container can boot the MCP protocol with the example configuration; production usage can inject local settings and data through `MCP_SETTINGS_PATH` and volume mounts.

## Current Limitations

- The loader is currently focused on PDF documents.
- Chroma is the primary registered VectorStore backend.
- Multi-process write coordination and distributed deployment are outside the current scope.
- `DocumentManager.delete_document()` provides best-effort coordination rather than cross-store ACID semantics.
- External model availability, rate limits, cost, and output quality are outside this project's control.
- Test and evaluation results validate implementation behavior and regression; they should not be interpreted as production capacity or SLA claims.

## Repository Structure

```text
config/          configuration templates
data/            local documents and persisted data
knowledge/       correctness and design notes
scripts/         ingestion, query, evaluation, dashboard, and container utilities
src/             core implementation
tests/           unit, integration, and E2E coverage
```

## Roadmap

- Add reproducible retrieval benchmarks against a fixed ingested corpus
- Support additional document loaders
- Add new VectorStore and fusion backends
- Expand end-to-end MCP client examples
- Continue improving retrieval evaluation and observability

## License

Licensed under the [MIT License](LICENSE).
