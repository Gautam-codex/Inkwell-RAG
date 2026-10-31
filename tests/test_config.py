"""Phase 1 smoke tests: settings load with sensible defaults."""

from src.config import Settings


def test_defaults_are_offline(monkeypatch, tmp_path):
    # Ignore any real .env so the test only checks the code defaults.
    monkeypatch.delenv("PROVIDER", raising=False)
    s = Settings(_env_file=None, data_dir=tmp_path)
    assert s.provider == "ollama"
    assert s.active_llm_model == "llama3.2"
    assert s.active_embedding_model == "BAAI/bge-small-en-v1.5"
    assert s.provider_problem() is None


def test_openai_without_key_is_reported(tmp_path):
    s = Settings(_env_file=None, data_dir=tmp_path, provider="openai", openai_api_key=None)
    assert "OPENAI_API_KEY" in s.provider_problem()


def test_ensure_dirs_creates_folders(tmp_path):
    s = Settings(_env_file=None, data_dir=tmp_path)
    s.ensure_dirs()
    assert s.uploads_dir.is_dir() and s.chroma_dir.is_dir() and s.images_dir.is_dir()
