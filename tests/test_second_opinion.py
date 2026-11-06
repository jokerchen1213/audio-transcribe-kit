"""second_opinion 纯函数与报告落盘（全合成数据，不碰真实录音）。"""
import json
from pathlib import Path

import second_opinion
from backends.base import Word


def _words(text: str, *, start: float = 0.0, step: float = 0.5) -> list[Word]:
    return [Word(ch, start + i * step, start + i * step + step * 0.8) for i, ch in enumerate(text)]


def test_edit_distance_and_similarity():
    assert second_opinion.edit_distance("", "abc") == 3
    assert second_opinion.edit_distance("abc", "") == 3
    assert second_opinion.edit_distance("你好", "你好") == 0
    assert second_opinion.edit_distance("今天天气", "今天天汽") == 1
    assert second_opinion.similarity("", "") == 1.0
    assert second_opinion.similarity("abcd", "abce") == 0.75


def test_compare_flags_text_disagreement():
    qwen = _words("今天天气不错我们开会吧")            # 11 字，0–5.5 s
    moss = [{"text": "明天要去某地出差办事。", "start": 0.0, "end": 5.5}]   # 12 字，读法完全不同
    res = second_opinion.compare(qwen, moss, [{"start": 0.0, "end": 6.0, "last_ts": 6.0}])
    assert res.disagreements == 1 and res.suspected_gaps == 0
    assert res.entries[0]["kind"] == "文字分歧"
    assert res.entries[0]["moss"] == "明天要去某地出差办事"


def test_compare_flags_suspected_gap_not_double_counted():
    # MOSS 20 字，Qwen3 只认到 3 字（< 30%）→ 疑似漏字，且不再重复计「文字分歧」
    qwen = _words("今天天气好")
    moss = [{"text": "今天天气不错我们下午三点开会讨论一下这个方案", "start": 0.0, "end": 8.0}]
    res = second_opinion.compare(qwen, moss, [{"start": 0.0, "end": 8.0, "last_ts": 8.0}])
    assert res.suspected_gaps == 1 and res.disagreements == 0
    assert res.entries[0]["kind"] == "Qwen3 疑似漏字"


def test_compare_short_segment_and_gap_region_skipped():
    qwen = _words("完全对不上的十个字内容")
    moss = [
        {"text": "好的。", "start": 0.0, "end": 1.0},                         # 去标点 2 字 < 6 → 不比
        {"text": "这句话在缺口区里不该被比对", "start": 12.0, "end": 16.0},   # 块 last_ts=10 之后 → 不比
    ]
    res = second_opinion.compare(qwen, moss, [{"start": 0.0, "end": 20.0, "last_ts": 10.0}])
    assert res.entries == [] and res.disagreements == 0 and res.suspected_gaps == 0
    assert res.moss_gap_s == 10.0      # 缺口秒数 = 20 − 10


def test_compare_uses_char_midpoint_and_ignores_punctuation():
    # Qwen3 字中点落在 MOSS 段内才计入；标点差异不算分歧
    qwen = _words("今天天气不错")     # 0.0–4.0
    moss = [{"text": "今天天气不错。", "start": 0.0, "end": 4.0}]
    res = second_opinion.compare(qwen, moss, [{"start": 0.0, "end": 4.0, "last_ts": 4.0}])
    assert res.entries == [] and res.disagreements == 0


def test_write_report_skips_when_no_entries(tmp_path):
    md = tmp_path / "_md" / "录音.md"
    res = second_opinion.Result()
    assert second_opinion.write_report(md, res, "moss@704aa4a", "录音.m4a") is None
    assert not (tmp_path / "_md" / "录音.分歧.md").exists()


def test_write_report_content_and_atomic(tmp_path):
    md = tmp_path / "_md" / "通话 张三.m4a.md"
    md.parent.mkdir(parents=True)
    md.write_text("主稿\n", encoding="utf-8")
    res = second_opinion.Result(entries=[
        {"start": 62.0, "kind": "文字分歧", "qwen": "今天天气不错", "moss": "今天天汽不错"},
        {"start": 3661.0, "kind": "Qwen3 疑似漏字", "qwen": "好的", "moss": "这句话MOSS读到了但Qwen3没有读到"},
    ], disagreements=1, suspected_gaps=1, moss_gap_s=12.3)
    out = second_opinion.write_report(md, res, "moss@704aa4a", "通话 张三.m4a")
    assert out == md.parent / "通话 张三.m4a.分歧.md"
    text = out.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    fm = text.split("---")[1]
    assert 'source: "通话 张三.m4a"' in fm
    assert "second_opinion: moss@704aa4a" in fm
    assert "disagreements: 1" in fm and "suspected_gaps: 1" in fm and "moss_gap_s: 12.3" in fm
    assert "MOSS 只作第二意见；引用原文或质证前请回听原声核对。" in text
    assert "| 时间 | 类型 | Qwen3 文字 | MOSS 文字 |" in text
    assert "| 00:01:02 | 文字分歧 | 今天天气不错 | 今天天汽不错 |" in text
    assert "| 01:01:01 | Qwen3 疑似漏字 | 好的 |" in text
    assert not list(md.parent.glob("*.tmp"))
    assert md.read_text(encoding="utf-8") == "主稿\n"      # 主稿不动


def test_report_cell_escapes_pipe(tmp_path):
    md = tmp_path / "a.md"
    res = second_opinion.Result(entries=[{"start": 0, "kind": "文字分歧", "qwen": "甲|乙", "moss": "甲 乙"}],
                                disagreements=1)
    out = second_opinion.write_report(md, res, "moss@704aa4a", "a.m4a")
    assert "甲\\|乙" in out.read_text(encoding="utf-8")
