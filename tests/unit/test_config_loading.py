import yaml
import pytest

from src.core.settings import load_settings


def _base_config():
    return {
        "llm": {"provider": "openai", "model": "gpt-4o"},
        "embedding": {"provider": "openai", "model": "text-embedding-3-small"},
        "vector_store": {"backend": "chroma", "persist_path": "/tmp/chroma"},
        "splitter": {"method": "recursive", "chunk_size": 512, "chunk_overlap": 64},
        "retrieval": {"sparse_backend": "bm25", "fusion_algorithm": "rrf", "top_k_dense": 20, "top_k_sparse": 20, "top_k_final": 10},
        "rerank": {"backend": "none"},
    }


def _write(path, data):
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_old_config_without_agentic_section_uses_defaults(tmp_path):
    path = tmp_path / "old.yaml"
    _write(path, _base_config())
    settings = load_settings(str(path))
    assert settings.agentic.max_iterations == 2
    assert settings.agentic.default_mode == "auto"


def test_new_agentic_config_loads(tmp_path):
    data = _base_config()
    data["agentic"] = {"max_iterations": 3, "max_subqueries": 2, "grader": {"min_confidence": .6}}
    path = tmp_path / "new.yaml"
    _write(path, data)
    settings = load_settings(str(path))
    assert settings.agentic.max_iterations == 3
    assert settings.agentic.grader.min_confidence == .6


@pytest.mark.parametrize("agentic", [{"max_iterations": 0}, {"grader": {"min_confidence": 1.2}}])
def test_invalid_agentic_config_rejected(tmp_path, agentic):
    data = _base_config()
    data["agentic"] = agentic
    path = tmp_path / "invalid.yaml"
    _write(path, data)
    with pytest.raises(ValueError):
        load_settings(str(path))
