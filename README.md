# RAG AS MCP

RAG AS MCP 是一个个人工程实践项目：把文档摄取、混合检索、重排序、评估与链路追踪拆分为可替换组件，并通过 MCP 暴露知识库查询能力。

项目用于验证模块化 RAG 的设计与实现，不以生产环境开箱即用为目标。

## 核心能力

- PDF 摄取、文本切分、Embedding 与向量持久化
- Dense 检索与 BM25 稀疏检索，并使用 RRF 融合结果
- 可选 Cross-Encoder / LLM 重排序、离线评估和调用链追踪
- 通过 MCP 提供 `query_knowledge_hub`、`list_collections` 和 `get_document_summary` 工具
- Streamlit Dashboard 用于查看数据、查询链路和评估结果

## 数据流

```text
PDF -> Loader -> Chunker -> Transform -> Embedding -> ChromaDB
                               |                     |
                               +-> BM25 index        |
                                                     v
Query -> Dense retrieval + Sparse retrieval -> RRF -> Rerank -> MCP response
```

核心实现位于：

- `src/ingestion/`：文档摄取与存储
- `src/core/query_engine/`：查询处理、混合检索与重排序
- `src/mcp_server/`：MCP 协议与工具
- `src/observability/`：Trace、Dashboard 与评估

## 环境要求

- Python 3.10+
- 使用在线 LLM 或 Embedding 时，需要对应服务的 API Key
- 使用 Ollama 时，需要本机已启动 Ollama 服务并准备相应模型

## 安装

```bash
git clone <your-repository-url> rag-as-mcp
cd rag-as-mcp
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
cp config/settings.example.yaml config/settings.yaml
```

Windows PowerShell 激活虚拟环境：

```powershell
.venv\Scripts\Activate.ps1
```

按需安装额外能力：

```bash
python -m pip install -e ".[dashboard]"  # Streamlit Dashboard
python -m pip install -e ".[rerank]"     # Cross-Encoder 重排序
python -m pip install -e ".[evaluation]" # Ragas 评估
python -m pip install -e ".[all]"        # 全部可选能力
```

## 配置

`config/settings.example.yaml` 是可提交的配置模板。复制后的 `config/settings.yaml` 仅用于本地运行，已被 Git 忽略。

凭证推荐通过环境变量提供，环境变量会覆盖 YAML 中的空值：

```bash
export RAG_LLM_API_KEY="your-llm-key"
export RAG_EMBEDDING_API_KEY="your-embedding-key"
export RAG_VISION_API_KEY="your-vision-key"
```

可选配置：

```bash
export RAG_LLM_AZURE_ENDPOINT="https://example.openai.azure.com/"
export RAG_EMBEDDING_BASE_URL="https://api.example.com/v1"
export MCP_SETTINGS_PATH="/absolute/path/to/settings.yaml"
```

不要把真实密钥写入示例配置或提交到 Git。

## 无付费 API 的最小验证

核心算法和协议测试不访问外部模型服务。测试按目录分层：

```bash
python -m pytest tests/unit -q
python -m pytest tests/integration -q
python -m pytest tests/e2e -q
python -m pytest -q
```

CI 在 Python 3.10 和 3.12 上运行 unit tests，并在 Python 3.12 上独立运行
integration / e2e tests。`PytestCollectionWarning` 会被视为错误，避免测试 helper
因命名问题被静默跳过。

## 运行完整流程

先在 `config/settings.yaml` 中选择 Embedding、LLM 和 Vector Store，并通过环境变量提供凭证。

生成项目自带的演示 PDF：

```bash
python scripts/gen_sample_pdfs.py
```

摄取文档：

```bash
python scripts/ingest.py --path data/documents/default --collection default
```

命令行查询：

```bash
python scripts/query.py --query "RRF 如何融合检索结果？" --collection default
```

启动 MCP Server：

```bash
python main.py
```

安装 Dashboard 依赖后启动界面：

```bash
python scripts/start_dashboard.py
```

运行本地检索评估：

```bash
python scripts/evaluate.py --evaluator local
```

## MCP 客户端配置

以下示例适用于支持 stdio MCP Server 的客户端。请替换项目绝对路径和 Python 解释器路径。

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

Server 使用 stdout 传输 JSON-RPC 消息，运行日志写入 stderr。

## 当前运行契约与已知限制

- 当前 Loader 主要面向 PDF，其他文档格式尚未实现。
- 当前已注册的 VectorStore backend 是 Chroma；未注册 backend 会在配置校验阶段直接拒绝，而不是延迟到运行时失败。
- Dense / BM25 / ImageStorage / FileIntegrity 都按 collection 隔离；文档更新采用“先写新版本、再清 stale”的可重试收敛策略。
- `DocumentManager.delete_document()` 是 best-effort 协调删除，不是跨四类存储的 ACID 事务；返回结果会明确列出成功和失败的 store。
- Query tool 创建单一 `TraceContext` 并贯穿 Dense / Sparse / Fusion / Rerank / Multimodal / Response，失败路径也会收集 trace。
- 外部模型服务的输出、速率限制和费用不由本项目控制。
- 测试覆盖核心组件与协议行为，但不代表生产环境容量或性能结论；多进程写入和分布式部署不在当前范围内。

当前 correctness hardening 的具体语义见 `knowledge/CORRECTNESS-HARDENING.md`。

## 后续计划

- 增加基于真实已摄取数据的可重复检索基准和实验记录
- 补充更多文档 Loader
- 扩展新的 VectorStore / Fusion 实现，并通过 registry 暴露能力
- 为 MCP 客户端增加完整的端到端示例

## License

[MIT](LICENSE)
