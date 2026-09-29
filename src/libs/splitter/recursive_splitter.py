"""Recursive Splitter 实现 (src/libs/splitter/recursive_splitter.py)"""
from typing import List

from src.libs.splitter.base_splitter import BaseSplitter
from src.libs.splitter.splitter_factory import register_splitter

# 针对 Markdown 文档优化的分隔符优先级列表
MARKDOWN_SEPARATORS = [
    "\n## ",   # H2 标题
    "\n### ",  # H3 标题
    "\n#### ", # H4 标题
    "\n\n",    # 段落空行
    "\n",      # 单行换行
    ". ",      # 句号
    ", ",      # 逗号
    " ",       # 空格
    "",        # 单字符（最后兜底）
]


@register_splitter("recursive")
class RecursiveSplitter(BaseSplitter):
    """递归字符切分器，对 Markdown 结构有天然适配性"""

    def __init__(self, chunk_size: int, chunk_overlap: int, **kwargs):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self._splitter = None  # 懒加载，避免没装 langchain 就报错

    def _get_splitter(self):
        """懒加载 LangChain splitter 实例"""
        if self._splitter is None:
            try:
                from langchain_text_splitters import RecursiveCharacterTextSplitter
            except ImportError:
                try:
                    from langchain.text_splitter import RecursiveCharacterTextSplitter
                except ImportError:
                    raise RuntimeError(
                        "请先安装 langchain 依赖：pip install langchain-text-splitters"
                    )
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=MARKDOWN_SEPARATORS,
                length_function=len,
            )
        return self._splitter

    def split_text(self, text: str, trace=None) -> List[str]:
        if not text or not text.strip():
            return []
        splitter = self._get_splitter()
        return splitter.split_text(text)
