#!/usr/bin/env python3
"""
ingest.py — 离线文档摄取 CLI 入口 (scripts/ingest.py)
=======================================================
用法：
    python scripts/ingest.py --path /path/to/doc.pdf --collection my_kb
    python scripts/ingest.py --path /path/to/dir/ --collection my_kb --force

参数说明:
  --path        : 待摄取的文件路径或目录路径（目录下递归扫描 .pdf 文件）
  --collection  : 目标知识库 collection 名称（默认 "default"）
  --config      : settings.yaml 路径（默认 "config/settings.yaml"）
  --force       : 强制重新摄取（跳过 SHA256 完整性检查）
  --batch-size  : 编码批次大小（默认 32）
"""
import argparse
import sys
import os
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent.parent))


def _collect_files(path: str) -> list:
    """收集待摄取的文件路径列表（支持单文件和目录递归扫描）"""
    p = Path(path)
    if p.is_file():
        return [str(p.resolve())]
    elif p.is_dir():
        files = sorted(p.rglob("*.pdf"))
        if not files:
            print(f"[warn] 目录 {path} 下未找到 .pdf 文件", file=sys.stderr)
        return [str(f.resolve()) for f in files]
    else:
        print(f"[error] 路径不存在: {path}", file=sys.stderr)
        sys.exit(1)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="RAG AS MCP — 离线文档摄取工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--path", required=True, help="文件路径或目录路径")
    parser.add_argument("--collection", default="default", help="知识库 collection 名称")
    parser.add_argument("--config", default="config/settings.yaml", help="配置文件路径")
    parser.add_argument("--force", action="store_true", help="强制重新摄取（跳过 SHA256 检查）")
    parser.add_argument("--batch-size", type=int, default=32, help="编码批次大小")
    return parser.parse_args()


def _progress_handler(progress):
    """进度回调：打印到 stdout"""
    bar = f"[{progress.stage:12s}] {progress.current}/{progress.total}"
    msg = f"  {progress.message}" if progress.message else ""
    print(f"{bar}{msg}")


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

    # 收集文件
    files = _collect_files(args.path)
    print(f"找到 {len(files)} 个文件，collection: {args.collection}")

    # 初始化 Pipeline
    from src.ingestion.pipeline import IngestionPipeline, PipelineError
    pipeline = IngestionPipeline(
        settings=settings,
        collection=args.collection,
        progress_callback=_progress_handler,
    )

    # 批量执行
    success, skipped, failed = 0, 0, 0
    for file_path in files:
        print(f"\n{'─' * 60}")
        print(f"处理: {file_path}")
        try:
            result = pipeline.run(file_path, force=args.force)
            if result.get("skipped"):
                print(f"  ↳ 已跳过（{result.get('reason', 'already_ingested')}）")
                skipped += 1
            else:
                print(
                    f"  ↳ 完成 — chunks: {result['chunk_count']}, "
                    f"images: {result['image_count']}, "
                    f"trace: {result['trace_id']}"
                )
                success += 1
        except PipelineError as e:
            print(f"  ↳ [失败] 阶段 {e.stage}: {e.cause}", file=sys.stderr)
            failed += 1
        except Exception as e:
            print(f"  ↳ [未知错误] {e}", file=sys.stderr)
            failed += 1

    # 汇总
    print(f"\n{'═' * 60}")
    print(f"摄取完成：成功 {success}，跳过 {skipped}，失败 {failed}，共 {len(files)} 个文件")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
