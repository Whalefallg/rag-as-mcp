"""
Embedding 抽象层 (src/libs/embedding/base_embedding.py)
=======================================================
为什么需要这个文件：
  Ingestion（写入向量）和 Retrieval（查询向量）都需要 Embedding，
  且必须用同一个模型——否则向量空间不一致，相似度计算失效。
  BaseEmbedding 定义统一接口，让两侧都通过工厂创建，
  只要 settings.yaml 中 embedding.provider 一致，向量空间就保证相同。

本文件定义文本向量化的统一抽象接口。

类说明:
  - BaseEmbedding : Embedding 抽象基类。所有具体 Provider（OpenAI/Azure/Ollama）
                    都必须继承它并实现 embed() 方法。
                    embed() 接受一个字符串列表，返回对应的浮点数向量列表。
                    在 Ingestion 阶段用于把 chunk 转成向量存入数据库，
                    在 Retrieval 阶段用于把用户 query 转成向量做相似度搜索。
                    _validate_texts() 是公共校验逻辑，子类在 embed() 开头调用即可。
"""
from abc import ABC, abstractmethod
from typing import List


class BaseEmbedding(ABC):
    """Embedding 抽象基类"""

    def __init__(self, model: str, **kwargs):
        self.model = model
        self.config = kwargs

    @abstractmethod
    def embed(self, texts: List[str], trace=None) -> List[List[float]]:
        """
        批量将文本转为向量。

        Args:
            texts: 待向量化的文本列表，例如 ["苹果是什么", "香蕉的营养"]。
            trace: 追踪上下文（可选，Phase F 阶段使用）。
        Returns:
            与输入等长的向量列表，每个向量是一个浮点数列表。
        Raises:
            ValueError: 输入格式错误。
            RuntimeError: API 调用失败。
        """
        pass

    def _validate_texts(self, texts: List[str]) -> None:
        """校验文本列表，子类在 embed() 开头调用。"""
        if not texts:
            raise ValueError("文本列表不能为空")
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(f"文本[{i}]类型错误，期望 str，实际 {type(text).__name__}")
