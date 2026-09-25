"""
RagasEvaluator 单元测试 (tests/unit/test_ragas_evaluator.py)
=============================================================
验收标准 (DEV_SPEC H1)：
  - Ragas 未安装时构造 RagasEvaluator 抛出 ImportError，含安装提示
  - Ragas 已安装时，mock _run_ragas 后 evaluate() 返回含所需指标的 dict
  - 注册到工厂：create_evaluator("ragas") 不报 ValueError
  - 无 answer 时指标只含 context_precision
  - RuntimeError 封装：_run_ragas 抛异常时被包装成 RuntimeError
"""
import pytest
from unittest.mock import patch, MagicMock


class TestRagasEvaluatorNotInstalled:
    def test_import_error_when_ragas_missing(self):
        """Ragas 未安装时构造器抛 ImportError，含安装提示"""
        with patch.dict("sys.modules", {"ragas": None}):
            from importlib import reload
            import src.observability.evaluation.ragas_evaluator as mod
            with pytest.raises(ImportError) as exc_info:
                mod.RagasEvaluator()
            assert "pip install ragas" in str(exc_info.value)

    def test_import_error_message_contains_package_name(self):
        with patch.dict("sys.modules", {"ragas": None}):
            import src.observability.evaluation.ragas_evaluator as mod
            with pytest.raises(ImportError) as exc_info:
                mod.RagasEvaluator()
            assert "ragas" in str(exc_info.value).lower()


class TestRagasEvaluatorWithMock:
    """Ragas 已安装（或 mock 已安装）时的行为测试"""

    def _make_evaluator(self):
        """patch _check_ragas 跳过安装检查"""
        with patch(
            "src.observability.evaluation.ragas_evaluator.RagasEvaluator._check_ragas"
        ):
            from src.observability.evaluation.ragas_evaluator import RagasEvaluator
            ev = RagasEvaluator.__new__(RagasEvaluator)
            ev._llm = None
            ev._embeddings = None
            ev.config = {}
            return ev

    def test_evaluate_returns_dict_with_faithfulness(self):
        ev = self._make_evaluator()
        mock_result = {
            "faithfulness": 0.9,
            "answer_relevancy": 0.85,
            "context_precision": 0.78,
        }
        with patch.object(ev, "_run_ragas", return_value=mock_result):
            result = ev.evaluate(
                query="test",
                retrieved_chunks=[{"id": "c1", "text": "content"}],
                generated_answer="answer",
                ground_truth={"answer": "gt"},
            )
        assert "faithfulness" in result
        assert "answer_relevancy" in result
        assert result["faithfulness"] == 0.9

    def test_evaluate_wraps_runtime_error(self):
        ev = self._make_evaluator()
        with patch.object(ev, "_run_ragas", side_effect=ValueError("api error")):
            with pytest.raises(RuntimeError) as exc_info:
                ev.evaluate(query="q", retrieved_chunks=[])
            assert "Ragas 评估失败" in str(exc_info.value)

    def test_evaluate_reraises_import_error(self):
        """ImportError 不被包装，直接透传"""
        ev = self._make_evaluator()
        with patch.object(ev, "_run_ragas", side_effect=ImportError("no ragas")):
            with pytest.raises(ImportError):
                ev.evaluate(query="q", retrieved_chunks=[])

    def test_registered_in_factory(self):
        """工厂注册验证：create_evaluator("ragas") 应可找到注册项"""
        from src.libs.evaluator.evaluator_factory import _EVALUATOR_REGISTRY
        import src.observability.evaluation.ragas_evaluator  # 触发注册
        assert "ragas" in _EVALUATOR_REGISTRY
