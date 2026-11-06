"""标点回填纯函数测试：离线，不触 mlx_qwen3_asr。"""
import sys
from backends.base import Word
from backends.qwen3_mlx import _reattach_punct

def _w(texts):
    return [Word(t, float(i), float(i) + 0.5) for i, t in enumerate(texts)]

def test_punct_attached_to_preceding_word_and_trailing():
    words = _w(["你", "好", "世", "界", "OK", "了"])
    out = _reattach_punct("你好，世界。OK了！", words)
    assert [w.text for w in out] == ["你", "好，", "世", "界。", "OK", "了！"]
    assert [(w.start, w.end) for w in out] == [(w.start, w.end) for w in words]

def test_leading_quote_prepended_to_first_word():
    out = _reattach_punct("“你好”", _w(["你", "好"]))
    assert [w.text for w in out] == ["“你", "好”"]

def test_alignment_failure_returns_input_unchanged():
    words = _w(["你", "错", "好"])
    assert _reattach_punct("你好。", words) is words

def test_empty_words_returns_empty():
    assert _reattach_punct("你好。", []) == []

def test_module_import_stays_lazy():
    assert "mlx_qwen3_asr" not in sys.modules

# ---- 非 CJK 串会被对齐器先 clean_token 再整体成一个 token（'3.5%'→'35'）；回填须按原文字符还原 ----
def test_percent_inside_token():
    out = _reattach_punct("利率3.5%。", _w(["利", "率", "35"]))
    assert [w.text for w in out] == ["利", "率", "3.5%。"]

def test_thousands_separator_inside_token():
    out = _reattach_punct("每月10,000元。", _w(["每", "月", "10000", "元"]))
    assert [w.text for w in out] == ["每", "月", "10,000", "元。"]

def test_colon_time_inside_token():
    out = _reattach_punct("下午12:30开会。", _w(["下", "午", "1230", "开", "会"]))
    assert [w.text for w in out] == ["下", "午", "12:30", "开", "会。"]

def test_hyphenated_english_token():
    out = _reattach_punct("state-of-the-art模型", _w(["stateoftheart", "模", "型"]))
    assert [w.text for w in out] == ["state-of-the-art", "模", "型"]

def test_repeated_characters():
    out = _reattach_punct("的的。", _w(["的", "的"]))
    assert [w.text for w in out] == ["的", "的。"]

def test_english_with_spaces():
    out = _reattach_punct("hello, world!", _w(["hello", "world"]))
    assert [w.text for w in out] == ["hello,", "world!"]
