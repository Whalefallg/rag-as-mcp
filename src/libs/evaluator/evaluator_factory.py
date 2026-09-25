"""
Evaluator 工厂 (src/libs/evaluator/evaluator_factory.py)
=========================================================
为什么需要这个文件：
  与其他工厂对称，按名称创建 Evaluator 实例。
  CompositeEvaluator 可以同时创建多个 evaluator（如 Ragas + Local）
  并行执行，结果汇总到 Dashboard。

  关键点：
    evaluator 实现类依赖 @register_evaluator 自注册，因此工厂模块加载时
    必须主动导入内置实现模块，触发注册逻辑。
"""
from typing import Dict, Type, Optional

from src.libs.evaluator.base_evaluator import BaseEvaluator
from src.core.settings import Settings

_EVALUATOR_REGISTRY: Dict[str, Type[BaseEvaluator]] = {}


def register_evaluator(backend: str):
    """装饰器：将 Evaluator 实现类注册到工厂，用法：@register_evaluator("ragas")"""
    def decorator(cls: Type[BaseEvaluator]):
        _EVALUATOR_REGISTRY[backend] = cls
        return cls
    return decorator


# 导入内置 evaluator 实现，触发自动注册。
# 阶段 H 增加了 local / ragas 两类 evaluator。
try:
    from src.observability.evaluation import local_retrieval_evaluator, ragas_evaluator  # noqa: F401,E402
except Exception:
    # ragas 可能未安装；local evaluator 应始终可用。
    try:
        from src.observability.evaluation import local_retrieval_evaluator  # noqa: F401,E402
    except Exception:
        pass


def create_evaluator(backend: str, settings: Optional[Settings] = None) -> BaseEvaluator:
    """
    按名称创建对应的 Evaluator 实例。

    Args:
        backend: 后端名称，例如 "ragas" / "local"。
        settings: 全局配置（当前未使用，保留扩展位）。
    Raises:
        ValueError: backend 未注册时抛出。
    """
    if backend not in _EVALUATOR_REGISTRY:
        supported = ", ".join(_EVALUATOR_REGISTRY.keys())
        raise ValueError(f"不支持的 Evaluator backend: '{backend}'。已注册的后端: {supported}")

    return _EVALUATOR_REGISTRY[backend]()


def get_supported_backends() -> list:
    """返回当前已注册的所有 Evaluator 后端名称。"""
    return list(_EVALUATOR_REGISTRY.keys())
