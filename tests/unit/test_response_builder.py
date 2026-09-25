"""
ResponseBuilder 单元测试 (tests/unit/test_response_builder.py)
==============================================================
为什么需要这个文件：
  ResponseBuilder 是 MCP 响应格式的核心组装器。
  它的输出直接影响 Copilot/Claude 展示给用户的内容质量，
  因此需要严格验证：结果存在时 Markdown 格式正确、引用标注存在、
  无结果时返回友好提示、图片内容被正确追加。
"""
import pytest
from src.core.types import RetrievalResult
from src.core.response.response_builder import ResponseBuilder
from src.core.response.citation_generator import CitationGenerator
from src.core.response.multimodal_assembler import MultimodalAssembler


def _make_result(chunk_id="c1", text="这是测试内容", score=0.9, **meta):
    m = {"source_path": "test.pdf", "page": 1}
    m.update(meta)
    return RetrievalResult(chunk_id=chunk_id, text=text, score=score, metadata=m)


class TestResponseBuilder:
    def test_no_results_returns_friendly_message(self):
        builder = ResponseBuilder()
        content = builder.build([], "查询")
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert "未找到" in content[0]["text"]

    def test_single_result_has_citation_number(self):
        builder = ResponseBuilder()
        results = [_make_result()]
        content = builder.build(results, "查询")
        text = content[0]["text"]
        assert "[1]" in text
        assert "test.pdf" in text

    def test_multiple_results_numbered_sequentially(self):
        builder = ResponseBuilder()
        results = [_make_result(f"c{i}", f"内容{i}", 0.9 - i * 0.1) for i in range(3)]
        content = builder.build(results, "查询")
        text = content[0]["text"]
        assert "[1]" in text
        assert "[2]" in text
        assert "[3]" in text

    def test_long_text_is_truncated(self):
        builder = ResponseBuilder(max_text_length=50)
        results = [_make_result(text="A" * 200)]
        content = builder.build(results, "查询")
        text = content[0]["text"]
        assert "…" in text

    def test_short_text_not_truncated(self):
        builder = ResponseBuilder(max_text_length=500)
        results = [_make_result(text="短文本")]
        content = builder.build(results, "查询")
        assert "短文本" in content[0]["text"]
        assert "…" not in content[0]["text"]

    def test_image_content_appended_after_text(self):
        builder = ResponseBuilder()
        results = [_make_result()]
        image_contents = [{"type": "image", "data": "abc123", "mimeType": "image/png"}]
        content = builder.build(results, "查询", image_contents)
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image"
        assert content[1]["data"] == "abc123"

    def test_content_has_text_type(self):
        builder = ResponseBuilder()
        content = builder.build([_make_result()], "查询")
        assert content[0]["type"] == "text"

    def test_score_shown_in_output(self):
        builder = ResponseBuilder()
        content = builder.build([_make_result(score=0.876)], "查询")
        assert "0.876" in content[0]["text"]

    def test_build_with_structured_returns_citations(self):
        builder = ResponseBuilder()
        result = builder.build_with_structured([_make_result()], "查询")
        assert "content" in result
        assert "structuredContent" in result
        citations = result["structuredContent"]["citations"]
        assert len(citations) == 1
        assert citations[0]["index"] == 1


class TestCitationGenerator:
    def test_citation_index_starts_at_1(self):
        gen = CitationGenerator()
        results = [_make_result(f"c{i}") for i in range(3)]
        citations = gen.generate(results)
        assert [c["index"] for c in citations] == [1, 2, 3]

    def test_citation_contains_required_fields(self):
        gen = CitationGenerator()
        citations = gen.generate([_make_result(chunk_id="c1", score=0.9)])
        c = citations[0]
        assert c["chunk_id"] == "c1"
        assert c["source_path"] == "test.pdf"
        assert c["source"] == "test.pdf"
        assert c["score"] == 0.9
        assert "snippet" in c

    def test_snippet_truncated_to_150(self):
        gen = CitationGenerator()
        citations = gen.generate([_make_result(text="X" * 300)])
        assert len(citations[0]["snippet"]) <= 152  # 150 + "…"

    def test_optional_fields_included_when_present(self):
        gen = CitationGenerator()
        result = _make_result(page=5, tags=["tag1"], title="标题", summary="摘要")
        citations = gen.generate([result])
        c = citations[0]
        assert c["page"] == 5
        assert c["tags"] == ["tag1"]
        assert c["title"] == "标题"

    def test_optional_fields_absent_when_missing(self):
        gen = CitationGenerator()
        result = RetrievalResult(chunk_id="c1", text="text", score=0.5, metadata={})
        citations = gen.generate([result])
        assert "page" not in citations[0]
        assert "tags" not in citations[0]

    def test_empty_results_returns_empty_list(self):
        gen = CitationGenerator()
        assert gen.generate([]) == []


class TestMultimodalAssembler:
    def test_no_storage_returns_empty(self):
        assembler = MultimodalAssembler(image_storage=None)
        result = _make_result(image_refs=["img1"])
        assert assembler.assemble([result]) == []

    def test_no_image_refs_returns_empty(self):
        storage = _make_mock_storage({"img1": "/tmp/img1.png"})
        assembler = MultimodalAssembler(image_storage=storage)
        result = _make_result()  # 无 image_refs
        assert assembler.assemble([result]) == []

    def test_missing_file_skipped_gracefully(self):
        storage = _make_mock_storage({})  # 查不到路径
        assembler = MultimodalAssembler(image_storage=storage)
        result = _make_result(image_refs=["img1"])
        output = assembler.assemble([result])
        assert output == []

    def test_max_images_limits_output(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = {}
            for i in range(5):
                p = os.path.join(tmpdir, f"img{i}.png")
                # 写入最小有效 PNG（1x1 白色像素）
                open(p, "wb").write(
                    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
                    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
                    b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
                    b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
                )
                paths[f"img{i}"] = p

            storage = _make_mock_storage(paths)
            assembler = MultimodalAssembler(image_storage=storage)
            result = _make_result(image_refs=[f"img{i}" for i in range(5)])
            output = assembler.assemble([result], max_images=2)
            assert len(output) == 2

    def test_deduplicates_same_image_across_results(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            p = os.path.join(tmpdir, "img.png")
            open(p, "wb").write(
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
                b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
                b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
                b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            storage = _make_mock_storage({"img1": p})
            assembler = MultimodalAssembler(image_storage=storage)
            r1 = _make_result("c1", image_refs=["img1"])
            r2 = _make_result("c2", image_refs=["img1"])
            output = assembler.assemble([r1, r2])
            assert len(output) == 1  # 相同图片只返回一次


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_mock_storage(path_map: dict):
    """创建一个简单的 mock ImageStorage"""
    class MockStorage:
        def get_path(self, image_id):
            return path_map.get(image_id)
    return MockStorage()
