"""ffmpeg / ffprobe 封装：时长、转 16 kHz 单声道、超长切片。原件只读。"""
from __future__ import annotations
import shutil, subprocess
from pathlib import Path

AUDIO_EXTS = frozenset({".m4a", ".mp3", ".mp4", ".wav", ".amr", ".wma", ".aac", ".flac", ".ogg", ".opus", ".mov", ".webm"})

class AudioError(Exception):
    pass

def _need(tool: str) -> str:
    p = shutil.which(tool)
    if not p:
        raise AudioError(f"缺少 {tool}：brew install ffmpeg")
    return p

def _run(cmd: list[str], what: str) -> subprocess.CompletedProcess[str]:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise AudioError(f"{what}：{r.stderr.strip()[:200]}")
    return r

def probe_duration(path: Path) -> float:
    r = _run([_need("ffprobe"), "-v", "error", "-show_entries", "format=duration",
              "-of", "default=nw=1:nk=1", str(path)], f"ffprobe 读取失败：{path.name}")
    try:
        return float(r.stdout.strip())
    except ValueError:
        raise AudioError(f"ffprobe 未返回有效时长：{path.name}")

def to_wav16k(src: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (src.stem + ".16k.wav")
    try:
        _run([_need("ffmpeg"), "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
              "-c:a", "pcm_s16le", str(out)], f"ffmpeg 转换失败：{src.name}")
        if not out.exists():
            raise AudioError(f"ffmpeg 转换失败：{src.name}：未生成输出文件")
    except AudioError:
        out.unlink(missing_ok=True)
        raise
    return out

def split_wav(wav: Path, chunk_s: float, out_dir: Path) -> list[tuple[Path, float]]:
    if chunk_s <= 0:
        raise AudioError(f"chunk_s 必须为正数，当前 {chunk_s}")
    out_dir.mkdir(parents=True, exist_ok=True)
    total = probe_duration(wav)
    parts: list[tuple[Path, float]] = []
    written: list[Path] = []
    offset, i = 0.0, 0
    try:
        while offset < total - 0.05:
            p = out_dir / f"{wav.stem}.part{i:03d}.wav"
            written.append(p)
            _run([_need("ffmpeg"), "-v", "error", "-y", "-ss", f"{offset:.3f}", "-t", f"{chunk_s:.3f}",
                  "-i", str(wav), "-c", "copy", str(p)], f"ffmpeg 切片失败：{wav.name}#{i}")
            parts.append((p, offset)); offset += chunk_s; i += 1
    except AudioError:
        for part_path in written:
            part_path.unlink(missing_ok=True)
        raise
    return parts
