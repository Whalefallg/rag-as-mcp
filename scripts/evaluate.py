"""
evaluate.py (scripts/evaluate.py)
===================================
运行方式：
  python scripts/evaluate.py
  python scripts/evaluate.py --test-set tests/fixtures/golden_test_set.json
  python scripts/evaluate.py --collection default --evaluator local
  python scripts/evaluate.py --output reports/eval_result.json
"""
import argparse
import json
import sys
from pathlib import Path

# 确保项目根目录在 path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 RAG 评估并输出指标报告")
    parser.add_argument(
        "--test-set",
        default="tests/fixtures/golden_test_set.json",
        help="golden test set 路径（default: tests/fixtures/golden_test_set.json）",
    )
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="settings.yaml 路径",
    )
    parser.add_argument(
        "--collection",
        default="default",
        help="目标知识库集合名称",
    )
    parser.add_argument(
        "--evaluator",
        default="local",
        choices=["local", "ragas", "composite"],
        help="评估器类型（local=离线，ragas=需要API，composite=组合）",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="将完整报告以 JSON 格式输出到文件（可选）",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="展示每条用例的详细结果",
    )
    args = parser.parse_args()

    try:
        from src.core.settings import load_settings
        settings = load_settings(args.config)
    except Exception as exc:
        print(f"[ERROR] 无法加载配置文件 {args.config}：{exc}")
        return 1

    evaluator = _build_evaluator(args.evaluator, settings)
    if evaluator is None:
        return 1

    from src.observability.evaluation.eval_runner import EvalRunner
    runner = EvalRunner(
        settings=settings,
        evaluator=evaluator,
        collection=args.collection,
    )

    print(f"[INFO] 加载测试集：{args.test_set}")
    print(f"[INFO] 评估器：{args.evaluator}  集合：{args.collection}\n")

    try:
        report = runner.run(args.test_set)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1
    except Exception as exc:
        print(f"[ERROR] 评估运行失败：{exc}")
        return 1

    report.print_summary()

    if args.verbose:
        print("== 逐条结果 ==")
        for r in report.per_query:
            status = "✅" if r.error is None else "❌"
            metrics_str = "  ".join(f"{k}={v:.3f}" for k, v in r.metrics.items()
                                    if not k.startswith("_"))
            print(f"{status} [{r.query[:50]}]  {metrics_str}")
            if r.error:
                print(f"   → 错误：{r.error}")

    if args.output:
        _save_report(report, args.output)

    # CI 退出码：有失败用例时返回 1（便于流水线检测）
    return 1 if report.failed_cases > 0 else 0


def _build_evaluator(name: str, settings):
    if name == "local":
        from src.observability.evaluation.local_retrieval_evaluator import LocalRetrievalEvaluator
        return LocalRetrievalEvaluator(k=settings.retrieval.top_k_final)
    elif name == "ragas":
        try:
            from src.observability.evaluation.ragas_evaluator import RagasEvaluator
            return RagasEvaluator()
        except ImportError as exc:
            print(f"[ERROR] {exc}")
            return None
    elif name == "composite":
        from src.observability.evaluation.local_retrieval_evaluator import LocalRetrievalEvaluator
        evaluators = [LocalRetrievalEvaluator(k=settings.retrieval.top_k_final)]
        try:
            from src.observability.evaluation.ragas_evaluator import RagasEvaluator
            evaluators.append(RagasEvaluator())
        except ImportError:
            print("[WARN] Ragas 未安装，composite 模式仅使用 local evaluator。")
        from src.observability.evaluation.composite_evaluator import CompositeEvaluator
        return CompositeEvaluator(evaluators)
    return None


def _save_report(report, output_path: str) -> None:
    data = {
        "summary": report.summary,
        "total_cases": report.total_cases,
        "failed_cases": report.failed_cases,
        "elapsed_ms": report.elapsed_ms,
        "test_set_path": report.test_set_path,
        "per_query": [
            {
                "query": r.query,
                "metrics": r.metrics,
                "retrieved_chunk_ids": r.retrieved_chunk_ids,
                "retrieved_sources": r.retrieved_sources,
                "expected_chunk_ids": r.expected_chunk_ids,
                "expected_sources": r.expected_sources,
                "error": r.error,
            }
            for r in report.per_query
        ],
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] 报告已保存：{output_path}")


if __name__ == "__main__":
    sys.exit(main())
