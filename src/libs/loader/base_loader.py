"""Loader 抽象层 (src/libs/loader/base_loader.py)"""
from abc import ABC, abstractmethod
from src.core.types import Document


class BaseLoader(ABC):
    """文档加载器抽象基类"""

    @abstractmethod
    def load(self, path: str) -> Document:
        """
        加载文件并解析为 Document 对象。

        Args:
            path: 文件的绝对路径。
        Returns:
            Document: 包含规范化文本和元数据的文档对象。
        Raises:
            FileNotFoundError: 文件不存在。
            ValueError: 文件格式不支持或解析失败。
        """
        pass
