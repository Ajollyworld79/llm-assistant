import pytest

from backend.app.core.config import ConfigError, Settings


def test_defaults_match_production_shape():
    s = Settings.from_env({})
    assert s.demo is True
    assert s.embedding_dim == 768 and s.embedding_provider == "gemini"
    assert s.rate_limit_per_min == 10 and s.daily_token_budget == 50000


def test_lowercase_gemini_key_is_accepted():
    assert Settings.from_env({"gemini_api_key": "abc"}).gemini_api_key == "abc"


def test_exact_case_wins_over_lowercase():
    assert Settings.from_env({"GEMINI_API_KEY": "upper", "gemini_api_key": "lower"}).gemini_api_key == "upper"


def test_env_values_are_parsed():
    s = Settings.from_env(
        {
            "DEMO": "false",
            "EMBEDDING_DIM": "384",
            "RATE_LIMIT_PER_MIN": "3",
            "CORS_ALLOW_ORIGINS": "https://a.example, https://b.example",
            "TEMPERATURE": "0.5",
            "QDRANT_FORCE_RECREATE": "true",
        }
    )
    assert not s.demo and s.embedding_dim == 384 and s.rate_limit_per_min == 3
    assert s.cors_allow_origins == ("https://a.example", "https://b.example")
    assert s.temperature == 0.5


@pytest.mark.parametrize("env", [{"EMBEDDING_DIM": "x"}, {"TEMPERATURE": "hot"}])
def test_bad_numbers_raise(env):
    with pytest.raises(ConfigError):
        Settings.from_env(env)


def test_production_requires_encryption_key_and_gemini():
    with pytest.raises(ConfigError, match="PAYLOAD_ENCRYPTION_KEY"):
        Settings.from_env({"DEMO": "false"}).validate()
    with pytest.raises(ConfigError, match="EMBEDDING_PROVIDER"):
        Settings.from_env(
            {"DEMO": "false", "EMBEDDING_PROVIDER": "fastembed", "PAYLOAD_ENCRYPTION_KEY": "x"}
        ).validate()
    Settings.from_env({"DEMO": "true"}).validate()


def test_public_view_has_no_secrets():
    view = Settings(admin_token="a", gemini_api_key="g").public_view
    assert "admin_token" not in view and "gemini_api_key" not in view and "qdrant_path" not in view


def test_loads_dotenv_file_when_asked(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("MAX_SEARCH_LIMIT=7\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("backend.app.core.config.find_dotenv", lambda usecwd=False: str(env_file))
    monkeypatch.delenv("MAX_SEARCH_LIMIT", raising=False)
    assert Settings.from_env().max_search_limit == 7
