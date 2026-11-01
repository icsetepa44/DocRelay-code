from src.core import config
from src.llm.structured_output import build_request_config


def test_model_for_includes_vision_and_quantizations():
    m = config.model_for("validator")
    assert set(m) == {"backend", "model", "vision", "quantizations"}
    assert isinstance(m["vision"], bool)


def test_per_role_quantizations_resolve_from_models_yaml():
    # gemma-backed roles inherit the full-precision default (bf16/fp16 hosts exist)
    assert config.model_for("validator")["quantizations"] == ["bf16", "fp16"]
    # deepseek critic roles pin fp8 — their best available precision (no bf16/fp16 host on OpenRouter)
    for role in ("super_validator", "final_decider"):
        assert config.model_for(role)["quantizations"] == ["fp8"]


def test_provider_for_pins_per_model_quantizations_when_enabled(monkeypatch):
    monkeypatch.setattr(config, "USE_QUANTIZATIONS", True)
    assert config.provider_for(config.model_for("validator"))["quantizations"] == ["bf16", "fp16"]
    assert config.provider_for(config.model_for("super_validator"))["quantizations"] == ["fp8"]


def test_provider_for_omits_quantizations_when_disabled(monkeypatch):
    monkeypatch.setattr(config, "USE_QUANTIZATIONS", False)
    prov = config.provider_for(config.model_for("super_validator")) or {}
    assert "quantizations" not in prov


def test_extractor_models_is_a_list_with_vision_flags():
    models = config.extractor_models()
    assert len(models) >= 1
    for m in models:
        assert "model" in m and "backend" in m and isinstance(m["vision"], bool)
    # Don't hardcode specific model ids — models.yaml is the source of truth and the extractor
    # comparison set changes between runs. Assert the structural invariant that matters instead:
    # each model yields a distinct, filesystem-safe label (used to separate per-model run dirs).
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
