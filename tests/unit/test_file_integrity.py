"""FileIntegrity 单元测试 (tests/unit/test_file_integrity.py)"""
import os
import tempfile
import pytest
from src.libs.loader.file_integrity import SQLiteIntegrityChecker


@pytest.fixture
def checker(tmp_path):
    db = str(tmp_path / "test_history.db")
    return SQLiteIntegrityChecker(db_path=db)


@pytest.fixture
def sample_file(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("hello world content", encoding="utf-8")
    return str(f)


def test_compute_sha256_consistent(checker, sample_file):
    h1 = checker.compute_sha256(sample_file)
    h2 = checker.compute_sha256(sample_file)
    assert h1 == h2
    assert len(h1) == 64  # SHA256 hex


def test_compute_sha256_different_files(checker, tmp_path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("content A")
    f2.write_text("content B")
    assert checker.compute_sha256(str(f1)) != checker.compute_sha256(str(f2))


def test_should_skip_new_hash_returns_false(checker):
    assert checker.should_skip("nonexistent_hash_abc") is False


def test_mark_success_then_should_skip_true(checker, sample_file):
    h = checker.compute_sha256(sample_file)
    assert checker.should_skip(h) is False
    checker.mark_success(h, sample_file)
    assert checker.should_skip(h) is True


def test_mark_failed_then_should_skip_false(checker):
    h = "fake_hash_failed_001"
    checker.mark_failed(h, error_msg="test error")
    assert checker.should_skip(h) is False  # failed 状态允许重试


def test_mark_success_after_failed(checker, sample_file):
    h = checker.compute_sha256(sample_file)
    checker.mark_failed(h, "first attempt failed")
    checker.mark_success(h, sample_file)
    assert checker.should_skip(h) is True


def test_get_record_returns_correct_status(checker, sample_file):
    h = checker.compute_sha256(sample_file)
    checker.mark_success(h, sample_file)
    record = checker.get_record(h)
    assert record is not None
    assert record["status"] == "success"
    assert record["file_path"] == sample_file


def test_db_file_created(tmp_path):
    db_path = str(tmp_path / "subdir" / "history.db")
    checker = SQLiteIntegrityChecker(db_path=db_path)
    assert os.path.exists(db_path)


def test_idempotent_mark_success(checker, sample_file):
    h = checker.compute_sha256(sample_file)
    checker.mark_success(h, sample_file)
    checker.mark_success(h, sample_file)  # 重复调用不报错
    assert checker.should_skip(h) is True
