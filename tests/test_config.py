from pathlib import Path

import pytest

from src.utils.config import ConfigError, load_config, require_sections


def test_load_smoke_config() -> None:
    config = load_config(Path("configs/environment/smoke.yaml"))
    require_sections(config, "experiment", "model", "generation", "environment")
    assert config["model"]["name_or_path"] == "Qwen/Qwen3-0.6B-Base"
    assert config["experiment"]["seed"] == 42


def test_missing_section_is_rejected() -> None:
    with pytest.raises(ConfigError, match="Missing mapping"):
        require_sections({"experiment": {}}, "experiment", "model")

