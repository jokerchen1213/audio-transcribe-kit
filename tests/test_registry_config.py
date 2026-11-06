from pathlib import Path
import pytest
from backends import ASR_BACKENDS, DIARIZERS, get_asr, get_diarizer, list_backends
from backends.base import ASRResult, Word
from config import load_config, resolve

def test_registry_has_real_backends():
    assert ASR_BACKENDS["qwen3-mlx"].endswith(":Qwen3MLX")
    assert DIARIZERS["sherpa"].endswith(":SherpaDiarizer")
    assert DIARIZERS["nemotron"].endswith(":NemotronDiarizer")
    assert list_backends() == {"asr": ["qwen3-mlx"], "diarizer": ["nemotron", "sherpa"]}

def test_get_backend_by_module_path(monkeypatch):
    monkeypatch.setitem(ASR_BACKENDS, "fake-asr", "tests.fake_backends:FakeASR")
    monkeypatch.setitem(DIARIZERS, "fake-diar", "tests.fake_backends:FakeDiarizer")
    asr = get_asr("fake-asr", {"x": 1})
    assert asr.cfg == {"x": 1}
    res = asr.transcribe(Path("a.wav"), context="张三")
    assert isinstance(res, ASRResult) and isinstance(res.segments[0].words[0], Word)
    assert get_diarizer("fake-diar", {}).diarize(Path("a.wav"))[1].speaker == 1

def test_unknown_backend_lists_registered():
    with pytest.raises(KeyError) as e:
        get_asr("nope", {})
    assert "qwen3-mlx" in str(e.value)

def test_load_config_and_resolve(tmp_path: Path):
    toml = tmp_path / "b.toml"
    toml.write_text('[default]\nasr="qwen3-mlx"\ndiarizer="sherpa"\n[asr.qwen3-mlx]\nmodel="m"\n[diarizer.sherpa]\nthreshold=0.5\n', encoding="utf-8")
    cfg = load_config(toml)
    assert resolve(cfg, None, None) == ("qwen3-mlx", {"model": "m"}, "sherpa", {"threshold": 0.5})
    assert resolve(cfg, None, "none")[2] is None
    assert resolve(cfg, "qwen3-mlx", None)[0] == "qwen3-mlx"

def test_default_toml_ships_with_tool():
    cfg = load_config()
    assert cfg["default"]["asr"] == "qwen3-mlx" and cfg["default"]["diarizer"] == "nemotron"   # 10-22 起本地默认
    assert "sherpa" in cfg["diarizer"] and cfg["default"]["runner"] == "win"
    assert "funasr" not in cfg.get("asr", {})
