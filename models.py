"""sherpa 分离模型下载与缓存（GitHub Releases 直链，免账号）。"""
from __future__ import annotations
import tarfile, urllib.request
from pathlib import Path

DEFAULT_CACHE = Path.home() / ".cache" / "cc-transcribe" / "sherpa"

def _fetch(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
    except Exception as e:
        tmp.unlink(missing_ok=True)   # 残缺 .part 不留盘，避免下次被误认为完整文件
        raise RuntimeError(f"下载失败：{url}\n可手动下载后放到 {dst}\n原始错误：{e}") from e
    tmp.replace(dst)

def ensure_sherpa_models(seg_url: str, emb_url: str, cache_dir: Path | None = None) -> tuple[Path, Path]:
    cache = Path(cache_dir) if cache_dir else DEFAULT_CACHE
    seg_dir = cache / "sherpa-onnx-pyannote-segmentation-3-0"
    seg_model = seg_dir / "model.onnx"
    if not seg_model.exists():
        tar = cache / "seg.tar.bz2"
        _fetch(seg_url, tar)
        with tarfile.open(tar, "r:bz2") as t:
            t.extractall(cache, filter="data")
        tar.unlink(missing_ok=True)
        if not seg_model.exists():
            raise RuntimeError(f"解压后未找到 {seg_model}")
    emb_model = cache / Path(emb_url).name
    if not emb_model.exists():
        _fetch(emb_url, emb_model)
    return seg_model, emb_model
