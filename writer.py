"""Meta + Paragraph → Markdown（front-matter → 汇总占位 → 转录稿）。工具永不写汇总正文。"""
from __future__ import annotations
from dataclasses import dataclass
from merge import Paragraph

SUMMARY_PLACEHOLDER = "<!-- SUMMARY: pending -->"
H_SUMMARY = "## 结构化汇总"
H_TRANSCRIPT = "## 转录稿"

@dataclass
class Meta:
    source: str
    duration_s: float
    recorded_at: str
    transcribed_at: str
    asr_label: str
    diarizer_label: str | None
    speakers: int
    speakers_hint: str
    quality: str            # ok / suspect / diarization-failed
    context: str
    chunks: int
    speaker_map: dict[str, str] | None = None
    # 以下为 Windows 路线 / 第二意见才有的可选字段：None 时 render 不输出，保证 Mac 本地路线逐字节不变。
    engine_host: str | None = None          # "win"
    asr_revision: str | None = None
    diarizer_revision: str | None = None
    speakers_dropped: int | None = None     # 累计发声 < 5 秒被丢弃的残标签个数
    second_opinion: str | None = None       # 如 "moss@704aa4a"；未跑为 None
    disagreements: int | None = None        # 第二意见「文字分歧」条数
    suspected_gaps: int | None = None       # 第二意见「Qwen3 疑似漏字」条数

def format_ts(sec: float) -> str:
    s = int(sec); return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"

def _q(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'

def render(meta: Meta, paragraphs: list[Paragraph]) -> str:
    fm = ["---", f"source: {_q(meta.source)}", f"duration: {format_ts(meta.duration_s)}",
          f"recorded_at: {meta.recorded_at}", f"transcribed_at: {meta.transcribed_at}",
          f"asr: {meta.asr_label}", f"diarizer: {meta.diarizer_label or 'none'}",
          f"speakers: {meta.speakers}", f"speakers_hint: {meta.speakers_hint}", f"quality: {meta.quality}",
          f"context: {_q(meta.context)}", f"chunks: {meta.chunks}"]
    for key, value in (("engine_host", meta.engine_host), ("asr_revision", meta.asr_revision),
                       ("diarizer_revision", meta.diarizer_revision), ("speakers_dropped", meta.speakers_dropped),
                       ("second_opinion", meta.second_opinion), ("disagreements", meta.disagreements),
                       ("suspected_gaps", meta.suspected_gaps)):
        if value is not None:
            fm.append(f"{key}: {value}")
    if meta.speaker_map:
        fm.append("speaker_map:")
        fm += [f"  {_q(k)}: {_q(v)}" for k, v in meta.speaker_map.items()]
    fm.append("---")
    body = [H_SUMMARY, "", SUMMARY_PLACEHOLDER, "", H_TRANSCRIPT, ""]
    for p in paragraphs:
        label = f"**说话人{p.speaker + 1}**：" if p.speaker is not None else ""
        body += [f"<!-- T {format_ts(p.start)} -->", f"{label}{p.text}", ""]
    return "\n".join(fm) + "\n\n" + "\n".join(body).rstrip() + "\n"
