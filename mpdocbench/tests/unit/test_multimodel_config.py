from src.core import config
from src.llm.structured_output import build_request_config


def test_model_for_includes_vision():
    m = config.model_for("validator")
    assert set(m) == {"backend", "model", "vision", "quantizations"}
    assert isinstance(m["vision"], bool)


def test_region_analyzer_routed_to_gemma():
    assert config.model_for("region_analyzer") == {
        "backend": "openrouter", "model": "google/gemma-4-31b-it", "vision": True,
        "quantizations": ["bf16", "fp16"]}


def test_critic_corrector_roles_on_deepseek_with_fp8_pin():
    for role in ("super_validator", "final_decider"):
        m = config.model_for(role)
        assert m == {"backend": "openrouter", "model": "deepseek/deepseek-v4.1-flash", "vision": True,
                     "quantizations": ["fp8"]}


def test_provider_for_adds_per_model_precision_pin(monkeypatch):
    monkeypatch.setattr(config, "USE_QUANTIZATIONS", True)
    monkeypatch.setattr(config, "OPENROUTER_PROVIDER_BASE", {"allow_fallbacks": True})
    prov = config.provider_for(config.model_for("super_validator"))
    assert prov == {"allow_fallbacks": True, "quantizations": ["fp8"]}
    monkeypatch.setattr(config, "USE_QUANTIZATIONS", False)
    assert config.provider_for(config.model_for("super_validator")) == {"allow_fallbacks": True}


def test_extractor_models_is_a_list_with_vision_flags():
    models = config.extractor_models()
    assert len(models) >= 1
    for m in models:
        assert "model" in m and "backend" in m and isinstance(m["vision"], bool)
    labels = {config.model_label(m["model"]) for m in models}
    assert len(labels) == len(models)


def test_model_label_is_filesystem_safe():
    assert config.model_label("deepseek/deepseek-v4-flash-0731") == "deepseek_deepseek-v4-flash-0731"
    assert config.model_label("qwen/qwen3.7-flash") == "qwen_qwen3.7-flash"
    assert "/" not in config.model_label("a/b:c") and ":" not in config.model_label("a/b:c")


def test_build_request_config_honours_override():
    cfg = build_request_config(
        "extractor",
        model_override={"backend": "openrouter", "model": "x/y", "vision": False},
    )
    assert cfg.backend == "openrouter" and cfg.model == "x/y"
