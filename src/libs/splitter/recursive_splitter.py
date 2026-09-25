"""
Recursive Splitter 实现 (src/libs/splitter/recursive_splitter.py)
==================================================================
为什么需要这个文件：
  「递归」切分优先按大分隔符（Markdown 标题、段落）切，
  某段还是太长再用小分隔符（句子、字符）继续切，最大程度保留语义完整性。
  相比 Fixed 切分，它不会在句子中间断开；相比 Semantic 切分，无需额外模型调用。
  是工业界 RAG 系统最常用的默认切分策略。

本文件实现基于 LangChain RecursiveCharacterTextSplitter 的文本切分策略。

类说明:
  - RecursiveSplitter : 继承 BaseSplitter，封装 LangChain 的递归字符切分器。
                        "递归"的含义：优先按大分隔符（Markdown 标题、段落）切分，
                        如果某段还是太长，再用更小的分隔符（句子、字符）继续切，
                        直到每段都不超过 chunk_size。这样能最大限度保留语义完整性。
                        separators 列表专门为 Markdown 格式优化，按优先级排列：
                        标题 → 段落 → 句子 → 逗号 → 空格 → 单字符。
                        通过 @register_splitter("recursive") 自动注册到 SplitterFactory，
                        settings.yaml 中设置 splitter.method: recursive 时工厂自动创建此实例。
"""
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
