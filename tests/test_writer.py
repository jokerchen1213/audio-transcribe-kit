import yaml

from pathlib import Path
from merge import Paragraph
from writer import H_SUMMARY, H_TRANSCRIPT, SUMMARY_PLACEHOLDER, Meta, format_ts, render

FIX = Path(__file__).resolve().parent / "fixtures" / "win"

def _front_matter(md: str) -> dict:
    _, fm_text, _ = md.split("---", 2)
    return yaml.safe_load(fm_text)

def _meta(**kw):
    base = dict(source="a b.m4a", duration_s=515.5, recorded_at="2024-10-21", transcribed_at="2026-09-21",
                asr_label="qwen3-mlx / moona3k/mlx-qwen3-asr-1.7b-8bit", diarizer_label="sherpa / pyannote-seg-3.0 + eres2net-zh",
                speakers=2, speakers_hint="2", quality="ok", context="王五 赵六", chunks=1, speaker_map=None)
    base.update(kw); return Meta(**base)

def test_format_ts():
    assert format_ts(0) == "00:00:00" and format_ts(3725.4) == "01:02:05"

def test_render_structure_and_order():
    md = render(_meta(), [Paragraph(0, 0.0, 1.0, "你好"), Paragraph(1, 42.2, 43.0, "再见"), Paragraph(None, 60, 61, "无人")])
    assert md.startswith("---\nsource: \"a b.m4a\"\nduration: 00:08:35\n")
    assert "quality: ok" in md and "context: \"王五 赵六\"" in md and "speakers: 2" in md
    i_sum, i_ph, i_tr = md.index(H_SUMMARY), md.index(SUMMARY_PLACEHOLDER), md.index(H_TRANSCRIPT)
    assert i_sum < i_ph < i_tr
    body = md[i_tr:]
    assert "<!-- T 00:00:00 -->\n**说话人1**：你好" in body
    assert "<!-- T 00:00:42 -->\n**说话人2**：再见" in body
    assert "<!-- T 00:01:00 -->\n无人" in body        # 无说话人不打标签

def test_render_without_diarizer_and_with_map():
    md = render(_meta(diarizer_label=None, speakers=0, speaker_map={"1": "陈律师"}), [])
    assert "diarizer: none" in md and "speaker_map:" in md and "陈律师" in md
    assert _front_matter(md)["speaker_map"] == {"1": "陈律师"}

def test_front_matter_escapes_backslash_and_quotes():
    md = render(_meta(source=r"note\zoom.m4a", context='王五\\赵六 "领导"'), [])
    fm = _front_matter(md)
    assert fm["source"] == r"note\zoom.m4a"
    assert fm["context"] == '王五\\赵六 "领导"'

# ---- Windows 路线新增字段：为 None 时必须与改动前逐字节相同（V4 回归）----
# 黄金样本由**基线** writer.py 生成（tests/fixtures/win/render_*.golden.md），
# 不是用本文件同一份 render 现算的，才谈得上“新旧输出比对”。
def test_render_new_fields_none_matches_baseline_golden():
    meta = Meta(source="通话 虚构样本.m4a", duration_s=515.5, recorded_at="2024-10-21",
                transcribed_at="2026-10-22", asr_label="qwen3-mlx / moona3k/mlx-qwen3-asr-1.7b-8bit",
                diarizer_label="sherpa / pyannote-seg-3.0 + eres2net-zh", speakers=2,
                speakers_hint="2", quality="ok", context="甲方 乙方", chunks=1)
    paras = [Paragraph(0, 0.0, 1.25, "你好，这里是虚构测试文本"), Paragraph(1, 42.2, 43.0, "收到 OK"),
             Paragraph(None, 60.0, 61.0, "无说话人段落")]
    assert render(meta, paras) == (FIX / "render_plain.golden.md").read_text(encoding="utf-8")

def test_render_speaker_map_none_fields_matches_baseline_golden():
    meta = Meta(source="note\\zoom 录音.wav", duration_s=0.0, recorded_at="2026-01-02",
                transcribed_at="2026-10-22", asr_label="fake-asr/1.0", diarizer_label=None,
                speakers=1, speakers_hint="auto", quality="suspect", context="",
                chunks=2, speaker_map={"1": "张三"})
    assert render(meta, []) == (FIX / "render_map.golden.md").read_text(encoding="utf-8")

def test_render_windows_fields_only_when_set():
    kw = dict(source="a.m4a", duration_s=1.0, recorded_at="2026-01-01", transcribed_at="2026-10-22",
              asr_label="qwen3 / Qwen3-ASR-1.7B-hf@bcd2b5b", diarizer_label="nemotron@f667ed7",
              speakers=3, speakers_hint="auto", quality="ok", context="", chunks=2, engine_host="win",
              asr_revision="bcd2b5b7f32b480ab5790554cfa8347f246a14f3",
              diarizer_revision="f667ed73aee57d40cc39428eb768b4fd87a0a29e", speakers_dropped=1,
              second_opinion="moss@704aa4a", disagreements=2, suspected_gaps=1,
              speaker_map={"1": "张三"})
    md = render(Meta(**kw), [])
    fm = _front_matter(md)
    assert fm["engine_host"] == "win"
    assert fm["asr_revision"] == kw["asr_revision"]
    assert fm["diarizer_revision"] == kw["diarizer_revision"]
    assert fm["speakers_dropped"] == 1 and fm["second_opinion"] == "moss@704aa4a"
    assert fm["disagreements"] == 2 and fm["suspected_gaps"] == 1
    # 顺序：chunks → 新字段 → speaker_map
    order = ["chunks:", "engine_host:", "asr_revision:", "diarizer_revision:", "speakers_dropped:",
             "second_opinion:", "disagreements:", "suspected_gaps:", "speaker_map:"]
    idx = [md.index(k) for k in order]
    assert idx == sorted(idx)
    assert "second_opinion" not in render(_meta(), [])   # 未设置时一行都不多
