"""
Loader 抽象层 (src/libs/loader/base_loader.py)
==============================================
为什么需要这个文件：
  Loader 是 Ingestion 的第一步，不同文档格式（PDF/Word/HTML）需要不同解析逻辑。
  BaseLoader 定义统一接口，Pipeline 不需要知道具体格式，
  新增格式只需实现 BaseLoader，无需修改 Pipeline。

本文件定义文档加载的统一抽象接口。

类说明:
  - BaseLoader : Loader 抽象基类。所有具体格式加载器（PDF/Markdown/DOCX）
                 都必须继承它并实现 load() 方法。
                 load() 接收文件路径，返回统一的 Document 对象。
                 上层 Pipeline 只依赖这个接口，新增格式只需添加新实现。
"""
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
