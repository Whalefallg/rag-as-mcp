#!/usr/bin/env python3
"""
query.py — 在线查询 CLI 入口 (scripts/query.py)
================================================
为什么需要这个脚本：
  MCP Server（阶段 E）是生产环境的查询入口，但开发阶段需要一个轻量的命令行工具
  来快速验证检索效果——不用启动完整的 MCP Server，直接在终端跑一条命令
  就能看到"这个 query 能不能召回到我期望的文档"。
  这也是调试 Hybrid Search 和 Reranker 效果的标准工作流。

用法：
    python scripts/query.py --query "如何配置 Azure OpenAI？"
    python scripts/query.py --query "BM25 算法原理" --top-k 5 --verbose
    python scripts/query.py --query "test" --collection my_kb --no-rerank

参数说明:
  --query       : 必填，查询文本
  --top-k       : 返回结果数量（默认 10）
  --collection  : 限定检索集合（默认 "default"）
  --config      : settings.yaml 路径（默认 "config/settings.yaml"）
  --no-rerank   : 跳过 Reranker 精排阶段
  --verbose     : 显示 Dense/Sparse 各路中间召回结果
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _parse_args():
    parser = argparse.ArgumentParser(
        description="RAG AS MCP — 在线查询工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--query", required=True, help="查询文本")
    parser.add_argument("--top-k", type=int, default=10, help="返回结果数量")
    parser.add_argument("--collection", default="default", help="限定检索集合")
    parser.add_argument("--config", default="config/settings.yaml", help="配置文件路径")
    parser.add_argument("--no-rerank", action="store_true", help="跳过 Reranker 精排")
    parser.add_argument("--verbose", action="store_true", help="显示各阶段中间结果")
    return parser.parse_args()


def _print_results(results, title: str = "检索结果") -> None:
    print(f"\n{'─' * 60}")
    print(f"{title}（{len(results)} 条）")
    print('─' * 60)
    if not results:
        print("  (空)")
        return
    for i, r in enumerate(results, 1):
        source_path = r.metadata.get("source_path", r.metadata.get("source", "unknown"))
        summary = r.text[:120].replace("\n", " ").strip()
        if len(r.text) > 120:
            summary += "..."
        print(f"[{i:2d}] score={r.score:.4f}  from={Path(source_path).name}")
        print(f"     {summary}")


def main():
    args = _parse_args()

    # 加载配置
    try:
        from src.core.settings import load_settings
        settings = load_settings(args.config)
    except FileNotFoundError:
        print(f"[error] 配置文件不存在: {args.config}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[error] 配置加载失败: {e}", file=sys.stderr)
        sys.exit(1)

    # 初始化检索组件
    from src.core.query_engine.query_processor import QueryProcessor
    from src.core.query_engine.dense_retriever import DenseRetriever
    from src.core.query_engine.sparse_retriever import SparseRetriever
    from src.core.query_engine.fusion import RRFusion
    from src.core.query_engine.hybrid_search import HybridSearch
    from src.core.query_engine.reranker import CoreReranker
    from src.core.trace.trace_context import TraceContext

    trace = TraceContext(trace_type="query")
    trace.set_metadata("user_query", args.query)
    trace.set_metadata("collection", args.collection)
    trace.set_metadata("top_k", args.top_k)
    query_proc = QueryProcessor()
    dense = DenseRetriever(settings)
    sparse = SparseRetriever(settings)
    fusion = RRFusion(k=60)
    hybrid = HybridSearch(
        settings,
        query_processor=query_proc,
        dense_retriever=dense,
        sparse_retriever=sparse,
        fusion=fusion,
    )

    print(f"\nQuery: {args.query}")
    print(f"Collection: {args.collection}  top-k: {args.top_k}")

    # Verbose：显示 QueryProcessor 的中间结果
    if args.verbose:
        processed = query_proc.process(args.query)
        print(f"\n[QueryProcessor] keywords = {processed.keywords}")
        print(f"[QueryProcessor] filters  = {processed.filters}")

    # 执行混合检索
    try:
        results = hybrid.search(
            query=args.query,
            top_k=args.top_k,
            collection=args.collection,
            trace=trace,
        )
    except Exception as e:
        print(f"[error] 检索失败: {e}", file=sys.stderr)
        sys.exit(1)

    if not results:
        print("\n未找到相关文档，请先运行 ingest.py 摄取数据。")
        sys.exit(0)

    # Verbose：显示各路中间结果
    if args.verbose:
        try:
            processed = query_proc.process(args.query)
            dense_r = dense.retrieve(processed.original, top_k=settings.retrieval.top_k_dense)
            sparse_r = sparse.retrieve(processed.keywords, top_k=settings.retrieval.top_k_sparse)
            _print_results(dense_r, "Dense 召回结果")
            _print_results(sparse_r, "Sparse 召回结果")
            _print_results(results, "Fusion 融合结果")
        except Exception:
            pass

    # Reranker 精排
    if not args.no_rerank:
        try:
            reranker = CoreReranker(settings)
            results = reranker.rerank(args.query, results, top_k=args.top_k, trace=trace)
            if args.verbose:
                _print_results(results, "Rerank 精排结果")
        except Exception as e:
            print(f"[warn] Reranker 失败，使用 Fusion 结果: {e}", file=sys.stderr)

    # 最终输出
    _print_results(results, "最终结果")

    # 持久化 trace → logs/traces.jsonl（供 Dashboard Query 追踪页读取）
    from src.core.trace.trace_collector import global_collector
    global_collector.collect(trace)

    print(f"\n{'═' * 60}")
    print(f"查询完成  trace_id={trace.trace_id}")


if __name__ == "__main__":
    main()
