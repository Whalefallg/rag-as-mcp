"""
生成 Golden Test Set 对应的样本 PDF 文档。
纯 Python 标准库实现，无需 reportlab / fpdf 等第三方依赖。
"""
import os
import struct
import zlib
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "documents" / "default"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── 极简 PDF 生成器 ───────────────────────────────────────────────────────────

def _pdf(pages: list[str]) -> bytes:
    """
    生成包含多页纯文本的最小合法 PDF（PDF 1.4）。
    每页一段文字，字体 Helvetica，自动换行。
    """
    def encode_text(s: str) -> str:
        # PDF 字符串转义
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    def wrap(text: str, width: int = 80) -> list[str]:
        lines, cur = [], ""
        for word in text.split():
            if len(cur) + len(word) + 1 > width:
                lines.append(cur)
                cur = word
            else:
                cur = (cur + " " + word).strip()
        if cur:
            lines.append(cur)
        return lines

    body = b"%PDF-1.4\n"
    offsets = []

    # Object 1: catalog (placeholder, filled after pages)
    obj_starts = {}

    objects = {}

    # Build page content streams
    content_ids = []
    page_ids = []
    base = 3  # obj 1=catalog, 2=pages, then content+page pairs

    for i, text in enumerate(pages):
        lines = wrap(text)
        stream_txt = "BT\n/F1 11 Tf\n50 780 Td\n14 TL\n"
        for ln in lines:
            stream_txt += f"({encode_text(ln)}) Tj T*\n"
        stream_txt += "ET\n"
        stream_bytes = stream_txt.encode("latin-1", errors="replace")

        content_id = base + i * 2        # 3, 5, 7, ...
        page_id    = base + i * 2 + 1    # 4, 6, 8, ...
        content_ids.append(content_id)
        page_ids.append(page_id)
        objects[content_id] = (
            f"{content_id} 0 obj\n"
            f"<< /Length {len(stream_bytes)} >>\n"
            f"stream\n"
        ).encode() + stream_bytes + b"\nendstream\nendobj\n"
        objects[page_id] = (
            f"{page_id} 0 obj\n"
            f"<< /Type /Page /Parent 2 0 R\n"
            f"   /MediaBox [0 0 595 842]\n"
            f"   /Contents {content_id} 0 R\n"
            f"   /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >>\n"
            f">>\nendobj\n"
        ).encode()

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    catalog_obj = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    pages_obj = (
        f"2 0 obj\n"
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>\n"
        f"endobj\n"
    ).encode()

    # Assemble
    out = b"%PDF-1.4\n"
    xref = {}
    xref[1] = len(out); out += catalog_obj
    xref[2] = len(out); out += pages_obj
    for oid in sorted(objects):
        xref[oid] = len(out)
        out += objects[oid]

    # xref table
    total = max(xref) + 1
    xref_pos = len(out)
    out += f"xref\n0 {total}\n".encode()
    out += b"0000000000 65535 f \n"
    for i in range(1, total):
        out += f"{xref.get(i, 0):010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {total} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    ).encode()
    return out


# ── 文档内容 ──────────────────────────────────────────────────────────────────

DOCS = {
    "config_guide.pdf": [
        "Azure OpenAI 配置指南",
        (
            "本文档介绍如何在 RAG AS MCP 中配置 Azure OpenAI 端点。"
            " 打开 config/settings.yaml，将 llm.provider 设置为 azure。"
            " 在 llm.azure_endpoint 字段填写你的 Azure OpenAI 资源 URL，"
            " 格式为 https://<resource-name>.openai.azure.com/。"
            " 在 llm.api_key 字段填写 Azure 门户中的 API Key。"
            " llm.model 字段填写你在 Azure 中创建的部署名称（不是模型名）。"
            " endpoint 和 deployment 必须与 Azure 资源配置一致。"
            " 同理，embedding.provider 也可设置为 azure，并填写对应的 embedding 部署名。"
        ),
    ],
    "azure_setup.pdf": [
        "Azure OpenAI 快速入门",
        (
            "步骤一：登录 Azure 门户，创建 Azure OpenAI 资源。"
            " 步骤二：在资源中部署模型，例如 gpt-4o 和 text-embedding-3-small。"
            " 步骤三：复制端点 URL 和 API Key。"
            " 步骤四：在 settings.yaml 中配置如下：llm.provider: azure，"
            " llm.model: 你的部署名，llm.azure_endpoint: 你的端点 URL，"
            " llm.api_key: 你的密钥。"
            " 如果出现 AuthenticationError，检查 api_key、endpoint 和 deployment 是否匹配。"
        ),
    ],
    "bm25_guide.pdf": [
        "BM25 算法详解",
        (
            "BM25（Best Match 25）是信息检索领域最经典的稀疏检索算法。"
            " 它在 TF-IDF 的基础上引入了两个关键参数：k1 和 b。"
            " k1 参数控制词频（Term Frequency）的饱和速度。"
            " k1 越大，词频对得分的影响越持久；k1 越小，词频很快就会饱和。"
            " 典型取值范围是 1.2 到 2.0。"
            " b 参数控制文档长度归一化的强度。"
            " b=1 表示完全按文档长度归一化；b=0 表示不做归一化。"
            " 典型取值为 0.75。"
            " BM25 为专有名词和精确关键词提供词法信号，与 Dense Retrieval 形成互补。"
        ),
    ],
    "rag_concepts.pdf": [
        "RAG 核心概念",
        (
            "RAG（Retrieval-Augmented Generation）检索增强生成是一种将知识检索与语言生成结合的技术。"
            " 核心流程分为离线摄取和在线查询两个阶段。"
            " 离线摄取：将文档解析、切分为 Chunk，生成向量存入数据库，同时建立稀疏索引。"
            " 在线查询：系统根据 classic、auto 或 agentic mode 规划查询，检索相关 Chunk，"
            " 可选执行 Rerank，并通过 MCP 返回带引用的证据。"
            " 混合检索结合 BM25 稀疏检索和 Dense Embedding 语义检索，通过 RRF 融合两路结果，"
            " 同时利用精确关键词和语义相似度信号；实际质量应通过 evaluation set 测量。"
        ),
    ],
    "hybrid_search.pdf": [
        "混合检索与 RRF 融合",
        (
            "混合检索（Hybrid Search）同时利用稠密向量检索（Dense）和稀疏关键词检索（Sparse）。"
            " Dense 检索使用 Embedding 模型将文本转为高维向量，通过余弦相似度找到语义相近的文档。"
            " Sparse 检索使用 BM25 等算法，通过关键词匹配找到包含精确词语的文档。"
            " RRF（Reciprocal Rank Fusion）是一种常用的融合算法。"
            " 公式为：score(d) = sum(1 / (k + rank_i(d)))，其中 k 通常取 60。"
            " RRF 不依赖各路分数的绝对值，只看排名，因此对不同分布的分数具有鲁棒性。"
            " 融合后可选 Cross-Encoder 或 LLM 进行精排（Rerank），进一步提升 Top-K 质量。"
        ),
    ],
    "fusion_methods.pdf": [
        "检索融合方法对比",
        (
            "常见的多路检索融合方法包括：RRF、加权求和（Weighted Sum）和 CombSUM。"
            " RRF（Reciprocal Rank Fusion）：基于排名的融合，对分数分布不敏感，鲁棒性强，推荐默认使用。"
            " 加权求和（Weighted Sum）：对各路分数乘以权重后求和，需要各路分数归一化到同一量纲。"
            " CombSUM：直接对分数求和，实现简单但对量纲差异敏感。"
            " 当前实现通过 retrieval.fusion_algorithm: rrf 选择已注册的 RRF 融合器。"
            " 未注册的融合算法会在配置校验阶段被拒绝。"
        ),
    ],
    "setup_guide.pdf": [
        "项目安装与启动指南",
        (
            "安装步骤如下。"
            " 第一步：克隆仓库，进入项目目录。"
            " 第二步：创建并激活虚拟环境：python -m venv .venv && source .venv/bin/activate。"
            " 第三步：安装依赖：pip install -e '.[dev]'。"
            " 如需 Dashboard，安装项目可选依赖：pip install -e '.[dashboard]'。"
            " Cross-Encoder 和评估依赖分别通过 '.[rerank]' 与 '.[evaluation]' 安装。"
            " 第四步：配置 config/settings.yaml，填写 API Key 和 Provider。"
            " 第五步：摄取文档：python scripts/ingest.py --path data/documents/。"
            " 第六步：查询：python scripts/query.py --query '你的问题'。"
            " 第七步：启动 Dashboard：python scripts/start_dashboard.py。"
        ),
    ],
    "evaluation_guide.pdf": [
        "RAG 评估指标详解",
        (
            "RAG 系统的评估分为检索质量评估和生成质量评估两大类。"
            " 检索质量指标包括 Hit Rate、MRR 和 Precision@K。"
            " Hit Rate（命中率）：检索结果中至少包含一个正确文档的比例。"
            " MRR（Mean Reciprocal Rank）：正确文档排名的倒数均值，衡量正确答案排名靠前的程度。"
            " Precision@K：Top-K 结果中正确文档的比例。"
            " 生成质量指标（需要 Ragas）包括 Faithfulness 和 Answer Relevancy。"
            " Faithfulness（忠实度）：答案是否完全基于检索到的上下文，不捏造内容。"
            " Answer Relevancy（答案相关性）：答案是否准确回答了用户的问题。"
            " Context Precision（上下文精准度）：检索到的上下文中有用内容的比例。"
        ),
    ],
    "ragas_docs.pdf": [
        "Ragas 评估框架使用说明",
        (
            "Ragas 是可选的 RAG 评估框架，部分指标需要参考答案或 LLM judge。"
            " 安装项目评估依赖：pip install -e '.[evaluation]'。"
            " Faithfulness 指标衡量生成答案是否忠实于检索到的上下文，避免模型幻觉（Hallucination）。"
            " 计算方式：将答案分解为若干陈述，检查每个陈述是否可以从上下文中推导出来。"
            " 取值范围 0 到 1，越高越好。"
            " Answer Relevancy 衡量答案与问题的相关程度，通过反向生成问题并计算余弦相似度实现。"
            " Context Recall 衡量检索到的上下文覆盖了标准答案多少内容。"
            " 本项目默认提供 deterministic local retrieval evaluation；"
            " scripts/evaluate.py 可选择 local、ragas 或 composite evaluator。"
        ),
    ],
    "chunking_guide.pdf": [
        "文档切分策略指南",
        (
            "Chunk（文本块）是 RAG 系统的基本检索单元，切分策略直接影响检索质量。"
            " 常见方法包括固定长度、递归和语义切分。"
            " 当前项目注册的是 RecursiveSplitter：优先按段落等分隔符切分，再逐级细分。"
            " 配置中未注册的 splitter method 会在启动校验时被拒绝。"
            " Chunk Overlap（重叠）的作用：在相邻 Chunk 之间保留一部分重叠文本，"
            " 确保跨 Chunk 的信息不会因切分边界而丢失，避免语义断裂。"
            " 典型配置：chunk_size=1000，chunk_overlap=200（约 20% 重叠）。"
            " chunk_size 越小，检索粒度越细，但也会增加向量数量和存储开销。"
        ),
    ],
    "mcp_guide.pdf": [
        "MCP 协议集成指南",
        (
            "MCP（Model Context Protocol）是 Anthropic 提出的开放协议，"
            " 用于连接 AI 助手与外部工具和数据源。"
            " 本项目实现了 MCP Stdio Transport，通过标准输入输出与 MCP Client 通信。"
            " 在 VS Code 中配置 Copilot：在 .vscode/mcp.json 中添加服务器配置，"
            " command 填写 Python 解释器路径，args 填写 ['-m', 'src.mcp_server.server']，"
            " cwd 填写项目根目录。"
            " 配置完成后，AI 助手可调用以下工具：query_knowledge_hub 执行 classic、auto 或 agentic 检索，"
            " list_collections 列出知识库集合，get_document_summary 获取文档摘要。"
            " tools/call 调用格式：发送 JSON-RPC 请求，method 为 tools/call，"
            " params 包含 name（工具名）和 arguments（参数字典）。"
        ),
    ],
    "multimodal_guide.pdf": [
        "多模态文档处理指南",
        (
            "多模态处理允许 RAG 系统理解 PDF 中的图片内容，实现图文联合检索。"
            " 本项目采用 Image-to-Text 策略：提取 PDF 页面中的图片，"
            " 调用 Vision LLM（如 gpt-4o）生成图片描述文字，"
            " 将 caption 保存到 Chunk metadata 的 images 记录；Loader 使用 [IMAGE: id] 占位符关联图片。"
            " ImageCaptioner 是负责此流程的组件，在 settings.yaml 中通过"
            " ingestion.image_captioner.enabled: true 开启。"
            " 开启后，摄取含图文档时会自动调用 Vision LLM 生成图片描述。"
            " 查询时，MultimodalAssembler 根据检索到的 Chunk 关联对应图片，"
            " 在 MCP 响应中同时返回文本内容和图片（ImageContent 格式）。"
        ),
    ],
    "observability_guide.pdf": [
        "可观测性与追踪系统指南",
        (
            "全链路可观测性是调试和优化 RAG 系统的关键能力。"
            " 本项目通过 TraceContext 和 TraceCollector 记录查询与摄取阶段。"
            " 组件可使用 record_stage() 写入已有耗时，或使用 span() 上下文管理器自动计时。"
            " Agentic trace 还记录 route、plan、grade、rewrite、termination 和 degraded 状态。"
            " TraceCollector 负责将完整的 TraceContext 序列化为 JSON Lines 格式，"
            " 追加写入 logs/traces.jsonl 文件。"
            " Dashboard 的摄取追踪和查询追踪页面读取此文件，可视化每次操作的各阶段耗时分布，"
            " 帮助快速定位性能瓶颈（如 embedding 耗时过长、rerank 延迟偏高等）。"
        ),
    ],
    "architecture_guide.pdf": [
        "系统架构与存储层设计",
        (
            "RAG AS MCP 由 ingestion、HybridSearch、bounded Agentic orchestration、MCP 和 observability 模块组成。"
            " 存储层由四个独立组件构成，分别负责不同类型的数据持久化。"
            " ChromaDB：存储 Dense Embedding 向量，支持高效相似度查询。"
            " BM25Index：存储倒排索引，支持关键词精确匹配检索。"
            " ImageStorage：存储从 PDF 中提取的图片文件，按 source_path 组织目录。"
            " FileIntegrity（SQLite）：按 SHA256 与 collection 存储摄取记录，实现幂等处理。"
            " DocumentManager 是跨四个存储的协调层。"
            " delete_document 操作依次执行：删除 ChromaDB 中对应向量，"
            " 从 BM25 索引移除文档，删除 ImageStorage 中关联图片，"
            " 最后清除 FileIntegrity 中的文件记录。"
            " 删除是 explicit best-effort coordination，不是跨存储原子事务；"
            " 返回结果会明确记录成功和失败的存储，失败后可重试或重新摄取。"
        ),
    ],
}


# ── 生成文件 ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for filename, (title, body) in DOCS.items():
        content = f"{title}\n\n{body}"
        pdf_bytes = _pdf([content])
        out_path = OUTPUT_DIR / filename
        out_path.write_bytes(pdf_bytes)
        print(f"  生成: {out_path.name}  ({len(pdf_bytes):,} bytes)")

    print(f"\n共生成 {len(DOCS)} 个 PDF → {OUTPUT_DIR}")
    print("\n下一步，运行摄取命令：")
    print("  python scripts/ingest.py --path data/documents/default/ --collection default")
