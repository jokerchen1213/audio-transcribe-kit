"""后端协议与数据类。协议只定最小公分母：文本 + 时间戳 + 可选说话人。"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float

@dataclass
class Segment:
    text: str
    start: float
    end: float
    words: list[Word] = field(default_factory=list)

@dataclass(frozen=True)
class SpeakerTurn:
    speaker: int   # 0 起；writer 输出时 +1
    start: float
    end: float

@dataclass
class ASRResult:
    segments: list[Segment]
    language: str | None
    truncated: bool
    finish_reason: str | None
    model_label: str   # 写进 front-matter 的 "后端/模型" 标签

class ASRBackend(Protocol):
    name: str
    def transcribe(self, wav16k: Path, *, context: str | None = None) -> ASRResult: ...

class Diarizer(Protocol):
    name: str
    def diarize(self, wav16k: Path, *, num_speakers: int | None = None,
                min_speakers: int | None = None, max_speakers: int | None = None) -> list[SpeakerTurn]: ...
