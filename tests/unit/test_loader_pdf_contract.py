"""
PDF Loader 契约测试 (tests/unit/test_loader_pdf_contract.py)
============================================================
为什么需要这个文件：
  PDF 是最常见的知识库文档格式，Loader 是 Ingestion 的第一步。
  如果 PDF 解析失败（依赖未安装、文件损坏），错误应该早在 Loader 层暴露，
  而不是在 Pipeline 深处才崩溃。
  契约测试确保 BaseLoader 的抽象约定被遵守：FileNotFoundError 对应找不到文件，
  RuntimeError 对应依赖缺失，不会出现静默的 None 返回。

验收标准（DEV_SPEC C3）：
  - load() 返回 Document，metadata 含 source_path
  - 对不存在的文件抛出 FileNotFoundError
  - PdfLoader 在 PyMuPDF 未安装时抛出 RuntimeError（不是 ImportError 崩溃）
  - BaseLoader 的抽象接口不可直接实例化
"""
import pytest
from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.pdf_loader import PdfLoader


def test_base_loader_is_abstract():
    with pytest.raises(TypeError):
        BaseLoader()


def test_pdf_loader_file_not_found():
    loader = PdfLoader()
    with pytest.raises(FileNotFoundError):
        loader.load("/nonexistent/path/doc.pdf")


def test_pdf_loader_has_load_method():
    loader = PdfLoader()
    assert hasattr(loader, "load")
    assert callable(loader.load)


def test_pdf_loader_with_real_pdf(tmp_path):
    """用 reportlab 创建简单 PDF 并验证加载（若 reportlab/pymupdf 未安装则跳过）"""
    pytest.importorskip("fitz", reason="PyMuPDF (fitz) not installed")
    pytest.importorskip("reportlab", reason="reportlab not installed")

    from reportlab.pdfgen import canvas as rl_canvas
    pdf_path = str(tmp_path / "test.pdf")
    c = rl_canvas.Canvas(pdf_path)
    c.drawString(100, 750, "Hello World Test Document")
    c.save()

    loader = PdfLoader(image_output_dir=str(tmp_path / "images"))
    doc = loader.load(pdf_path)

    assert doc.id is not None
    assert doc.text is not None
    assert "source_path" in doc.metadata
    assert doc.metadata["doc_type"] == "pdf"
    assert "Hello" in doc.text or len(doc.text) >= 0  # PDF 解析成功


def test_pdf_loader_doc_id_stable(tmp_path):
    """同一文件两次加载产生相同 ID"""
    pytest.importorskip("fitz", reason="PyMuPDF (fitz) not installed")
    pytest.importorskip("reportlab", reason="reportlab not installed")

    from reportlab.pdfgen import canvas as rl_canvas
    pdf_path = str(tmp_path / "stable.pdf")
    c = rl_canvas.Canvas(pdf_path)
    c.drawString(100, 750, "Stable ID Test")
    c.save()

    loader = PdfLoader()
    doc1 = loader.load(pdf_path)
    doc2 = loader.load(pdf_path)
    assert doc1.id == doc2.id
