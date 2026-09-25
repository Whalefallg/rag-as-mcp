"""
ImageStorage 单元测试 (tests/unit/test_image_storage.py)
========================================================
为什么需要这个文件：
  多模态 RAG 的图片需要持久化存储（不能只放内存），并且在删除文档时要能级联清理。
  ImageStorage 管理图片文件 + SQLite 索引的双层结构，
  这里的测试验证两层都正确写入、读取、删除，
  确保图片的生命周期与文档保持同步，不留"孤儿图片"占用磁盘空间。

验收标准（DEV_SPEC C13）：
  - save() 后文件存在于磁盘
  - get_path() 返回正确路径，不存在返回 None
  - delete_by_doc() 删除文件和索引记录，返回正确数量
  - 数据库持久化验证
"""
import pytest
from pathlib import Path
from src.ingestion.storage.image_storage import ImageStorage


@pytest.fixture
def storage(tmp_path):
    return ImageStorage(
        storage_root=str(tmp_path / "images"),
        db_path=str(tmp_path / "image_index.db"),
    )


FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50


def test_save_creates_file(storage, tmp_path):
    path = storage.save("img001", FAKE_PNG, collection="test")
    assert Path(path).exists()


def test_save_returns_correct_path(storage, tmp_path):
    path = storage.save("img002", FAKE_PNG, collection="col1")
    assert "img002.png" in path
    assert "col1" in path


def test_get_path_returns_saved_path(storage):
    storage.save("img003", FAKE_PNG, collection="test")
    result = storage.get_path("img003")
    assert result is not None
    assert "img003.png" in result


def test_get_path_nonexistent_returns_none(storage):
    assert storage.get_path("nonexistent_img") is None


def test_list_by_collection(storage):
    storage.save("img_a1", FAKE_PNG, collection="collA")
    storage.save("img_a2", FAKE_PNG, collection="collA")
    storage.save("img_b1", FAKE_PNG, collection="collB")

    results = storage.list_by_collection("collA")
    assert len(results) == 2
    ids = {r["image_id"] for r in results}
    assert ids == {"img_a1", "img_a2"}


def test_delete_by_doc_removes_files(storage, tmp_path):
    storage.save("img_d1", FAKE_PNG, collection="test", doc_hash="doc123")
    storage.save("img_d2", FAKE_PNG, collection="test", doc_hash="doc123")
    storage.save("img_d3", FAKE_PNG, collection="test", doc_hash="other")

    count = storage.delete_by_doc("doc123")
    assert count == 2

    # 文件已删除
    assert storage.get_path("img_d1") is None
    assert storage.get_path("img_d2") is None
    # 其他文档的图片未受影响
    assert storage.get_path("img_d3") is not None


def test_save_idempotent(storage):
    """同一 image_id 重复 save 不报错，路径更新"""
    storage.save("dup_img", FAKE_PNG, collection="test")
    path2 = storage.save("dup_img", b"\x89PNG\r\n\x1a\n" + b"\xff" * 50, collection="test")
    result = storage.get_path("dup_img")
    assert result == path2


def test_db_persists_across_instances(tmp_path):
    """数据库持久化：新建实例能读到之前保存的记录"""
    db = str(tmp_path / "persist.db")
    root = str(tmp_path / "imgs")

    s1 = ImageStorage(storage_root=root, db_path=db)
    s1.save("persist_img", FAKE_PNG, collection="test")

    s2 = ImageStorage(storage_root=root, db_path=db)
    assert s2.get_path("persist_img") is not None
