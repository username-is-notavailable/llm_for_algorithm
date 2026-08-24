from pathlib import Path

import yaml

from scripts import render_m9_eval_config


def test_renderer_selects_posttrained_reference_without_base_name(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "rendered.yaml"
    monkeypatch.setattr(
        "sys.argv",
        [
            "render_m9_eval_config.py",
            "--input",
            "configs/eval/agent_qwen3_1_7b_base_smoke_v1.yaml",
            "--model-size",
            "1.7b-post",
            "--output",
            str(output),
            "--max-new-tokens",
            "8192",
            "--max-total-generation-tokens",
            "32768",
        ],
    )
    assert render_m9_eval_config.main() == 0
    config = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert config["model"]["name_or_path"] == "Qwen/Qwen3-1.7B"
    assert config["model"]["revision"] == "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    assert config["experiment"]["name"] == "m9-agent-qwen3-1.7b-posttrained-smoke-v1-long8k"
    assert config["generation"]["max_new_tokens"] == 8192
    assert config["agent"]["max_total_generation_tokens"] == 32768
