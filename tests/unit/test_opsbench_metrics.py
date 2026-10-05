import math
from pathlib import Path
import pytest
from src.observability.evaluation.opsbench import extract_kb_ids, validate_dataset
from src.observability.evaluation.retrieval_metrics import ndcg_at, percentile

ROOT=Path(__file__).resolve().parents[2]/"benchmarks/opsbench_v1"

def test_dataset_contract():
    assert validate_dataset(ROOT)["queries"] == 80

def test_extracts_multiple_stable_ids():
    assert extract_kb_ids("[KB_ID: a#x] text [KB_ID: b#y]") == ["a#x","b#y"]

def test_ndcg_perfect_ordering():
    assert ndcg_at(["a","b","c"],{"a":3,"b":2,"c":1},10) == pytest.approx(1)

def test_ndcg_reversed_ordering_hand_calculated():
    expected=((2**1-1)/1+(2**2-1)/math.log2(3)+(2**3-1)/2)/((2**3-1)/1+(2**2-1)/math.log2(3)+(2**1-1)/2)
    assert ndcg_at(["c","b","a"],{"a":3,"b":2,"c":1},10) == pytest.approx(expected)

def test_ndcg_no_relevant_results():
    assert ndcg_at(["x"],{"a":3},10) == 0
    assert ndcg_at(["x"],{},10) == 0

def test_percentile_linear_interpolation():
    assert percentile([0,10,20,30],.5)==15
    assert percentile([0,10,20,30],.95)==pytest.approx(28.5)
