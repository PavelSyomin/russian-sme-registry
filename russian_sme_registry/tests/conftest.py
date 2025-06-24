import pytest

from russian_sme_registry.main import app_config, default_config


@pytest.fixture(autouse=True)
def reset_settings(monkeypatch):
    for k, v in default_config.items():
        monkeypatch.setitem(app_config, k, v)