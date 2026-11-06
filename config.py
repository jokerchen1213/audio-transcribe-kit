"""读取 backends.toml。"""
from __future__ import annotations
import tomllib
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent / "backends.toml"

def load_config(path: Path | None = None) -> dict:
    with open(path or DEFAULT_PATH, "rb") as f:
        return tomllib.load(f)

def resolve(cfg: dict, asr: str | None, diarizer: str | None):
    """返回 (asr_name, asr_cfg, diarizer_name|None, diarizer_cfg)。diarizer=='none' 关闭分离。"""
    asr_name = asr or cfg["default"]["asr"]
    diar_name = diarizer or cfg["default"].get("diarizer")
    if diar_name == "none":
        diar_name = None
    asr_cfg = cfg.get("asr", {}).get(asr_name, {})
    diar_cfg = cfg.get("diarizer", {}).get(diar_name, {}) if diar_name else {}
    return asr_name, asr_cfg, diar_name, diar_cfg
