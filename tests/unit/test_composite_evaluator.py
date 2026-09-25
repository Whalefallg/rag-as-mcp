"""
CompositeEvaluator 单元测试 (tests/unit/test_composite_evaluator.py)
=====================================================================
验收标准 (DEV_SPEC H2)：
  - 两个 evaluator 时返回的 metrics 包含两者的指标
  - 任一 evaluator 失败时 _errors 字段有记录，其余指标正常返回
  - 空 evaluators 列表时 __init__ 抛 ValueError
  - 并行执行（结果顺序不影响合并）
  - evaluators 属性返回副本
"""
import time
import threading
import pytest
from unittest.mock import MagicMock
from src.observability.evaluation.composite_evaluator import CompositeEvaluator
from src.libs.evaluator.base_evaluator import BaseEvaluator


def _make_ev(name: str, result: dict, delay: float = 0.0, fail: bool = False):
    """构造一个返回固定结果的 mock evaluator"""
    ev = MagicMock(spec=BaseEvaluator)
    ev.__class__.__name__ = name

    def _evaluate(**kwargs):
        if delay:
            time.sleep(delay)
        if fail:
            raise RuntimeError(f"{name} failed")
        return result

    ev.evaluate.side_effect = _evaluate
    return ev


class TestCompositeEvaluatorInit:
    def test_empty_evaluators_raises(self):
        with pytest.raises(ValueError, match="至少需要一个"):
            CompositeEvaluator([])

    def test_single_evaluator_ok(self):
        ev = _make_ev("A", {"hit_rate": 0.9})
        comp = CompositeEvaluator([ev])
        assert len(comp.evaluators) == 1

    def test_evaluators_property_returns_copy(self):
        ev = _make_ev("A", {})
        comp = CompositeEvaluator([ev])
        lst = comp.evaluators
        lst.append("intruder")
        assert len(comp.evaluators) == 1


class TestCompositeEvaluatorMerge:
    def test_two_evaluators_merged_metrics(self):
        ev1 = _make_ev("Retrieval", {"hit_rate": 0.9, "mrr": 0.7})
        ev2 = _make_ev("Semantic", {"faithfulness": 0.85, "answer_relevancy": 0.8})
        comp = CompositeEvaluator([ev1, ev2])
        result = comp.evaluate(query="q", retrieved_chunks=[])
        assert "hit_rate" in result
        assert "faithfulness" in result
        assert result["hit_rate"] == 0.9
        assert result["faithfulness"] == 0.85

    def test_key_conflict_later_evaluator_wins(self):
        ev1 = _make_ev("A", {"shared_key": 0.5})
        ev2 = _make_ev("B", {"shared_key": 0.9})
        comp = CompositeEvaluator([ev1, ev2])
        result = comp.evaluate(query="q", retrieved_chunks=[])
        # 后完成的覆盖前者（并发结果，不保证顺序，但不能崩）
        assert "shared_key" in result

    def test_no_errors_key_when_all_succeed(self):
        ev1 = _make_ev("A", {"hit_rate": 1.0})
        comp = CompositeEvaluator([ev1])
        result = comp.evaluate(query="q", retrieved_chunks=[])
        assert "_errors" not in result

    def test_partial_failure_records_errors(self):
        ev_ok = _make_ev("OK", {"mrr": 0.8})
        ev_fail = _make_ev("Fail", {}, fail=True)
        comp = CompositeEvaluator([ev_ok, ev_fail])
        result = comp.evaluate(query="q", retrieved_chunks=[])
        assert "_errors" in result
        assert len(result["_errors"]) == 1
        assert "mrr" in result   # 成功的 evaluator 的指标仍然存在

    def test_all_fail_returns_errors_only(self):
        ev1 = _make_ev("A", {}, fail=True)
        ev2 = _make_ev("B", {}, fail=True)
        comp = CompositeEvaluator([ev1, ev2])
        result = comp.evaluate(query="q", retrieved_chunks=[])
        assert "_errors" in result
        # mock objects share __class__.__name__ ("MagicMock"); at least 1 error key
        assert len(result["_errors"]) >= 1

    def test_passes_all_args_to_evaluators(self):
        ev = _make_ev("A", {"hit_rate": 1.0})
        comp = CompositeEvaluator([ev])
        comp.evaluate(
            query="my query",
            retrieved_chunks=[{"id": "c1"}],
            generated_answer="answer",
            ground_truth={"answer": "gt"},
        )
        call_kwargs = ev.evaluate.call_args[1]
        assert call_kwargs["query"] == "my query"
        assert call_kwargs["generated_answer"] == "answer"

    def test_parallel_execution_overlaps_evaluators(self):
        """用 Barrier 验证两个 evaluator 确实并发进入执行区，不依赖 wall-clock 阈值。"""
        barrier = threading.Barrier(2)

        def _make_barrier_ev(name: str, result: dict):
            ev = MagicMock(spec=BaseEvaluator)
            ev.__class__.__name__ = name

            def _evaluate(**kwargs):
                # 若是串行执行，第一个任务会在这里超时/打破 barrier；
                # 两个 worker 并发时二者会互相释放。
                barrier.wait(timeout=1.0)
                return result

            ev.evaluate.side_effect = _evaluate
            return ev

        ev1 = _make_barrier_ev("A", {"a": 1.0})
        ev2 = _make_barrier_ev("B", {"b": 1.0})
        comp = CompositeEvaluator([ev1, ev2], max_workers=2)

        result = comp.evaluate(query="q", retrieved_chunks=[])

        assert result["a"] == 1.0
        assert result["b"] == 1.0
        assert "_errors" not in result
