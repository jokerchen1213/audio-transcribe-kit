"""注册表：名称 → "module:Class"。新增后端 = 新文件 + 此处一行 + backends.toml 一节。"""
from __future__ import annotations
import importlib

ASR_BACKENDS: dict[str, str] = {"qwen3-mlx": "backends.qwen3_mlx:Qwen3MLX"}
DIARIZERS: dict[str, str] = {"sherpa": "backends.sherpa_diar:SherpaDiarizer",
                             "nemotron": "backends.nemotron_diar:NemotronDiarizer"}

def _load(table: dict[str, str], name: str, cfg: dict):
    if name not in table:
        raise KeyError(f"未注册的后端 '{name}'；可用：{', '.join(sorted(table))}")
    mod, cls = table[name].split(":")
    return getattr(importlib.import_module(mod), cls)(cfg)

def get_asr(name: str, cfg: dict):
    return _load(ASR_BACKENDS, name, cfg)

def get_diarizer(name: str, cfg: dict):
    return _load(DIARIZERS, name, cfg)

def list_backends() -> dict[str, list[str]]:
    return {"asr": sorted(ASR_BACKENDS), "diarizer": sorted(DIARIZERS)}
