from backends.base import SpeakerTurn, Word
from merge import Paragraph, assign_speakers, build_paragraphs, join_words, shift_words

W = lambda t, s, e: Word(t, s, e)

def test_assign_by_midpoint_and_nearest_gap():
    words = [W("你好", 0.0, 0.5), W("世界", 0.6, 1.0), W("再见", 3.0, 3.4), W("远", 9.0, 9.2)]
    turns = [SpeakerTurn(0, 0.0, 1.2), SpeakerTurn(1, 2.8, 3.5)]
    got = [s for _, s in assign_speakers(words, turns, gap=1.0)]
    assert got == [0, 0, 1, None]   # "远" 距最近 turn 5.5s > gap → None

def test_paragraphs_split_on_speaker_pause_and_length():
    labeled = [(W("a", 0, 0.2), 0), (W("b", 0.3, 0.5), 0), (W("c", 3.0, 3.2), 0), (W("d", 3.3, 3.5), 1)]
    ps = build_paragraphs(labeled, pause=2.0, max_chars=300)
    assert [(p.speaker, p.text) for p in ps] == [(0, "a b"), (0, "c"), (1, "d")]
    assert ps[1].start == 3.0 and ps[0].end == 0.5
    long = [(W("字", i * 0.1, i * 0.1 + 0.05), 0) for i in range(700)]
    assert [len(p.text) for p in build_paragraphs(long, pause=2.0, max_chars=300)] == [300, 300, 100]

def test_join_words_cjk_vs_ascii():
    assert join_words(["你好", "世界", "OK", "了", "A", "B"]) == "你好世界OK了A B"

def test_shift_words():
    assert shift_words([W("x", 1.0, 1.5)], 30.0)[0] == W("x", 31.0, 31.5)

def test_paragraphs_without_speakers():
    labeled = [(W("a", 0, 0.2), None), (W("b", 5.0, 5.2), None)]
    ps = build_paragraphs(labeled)
    assert [p.speaker for p in ps] == [None, None] and len(ps) == 2

def test_paragraphs_ascii_words_never_exceed_max_chars():
    # ASCII 词相邻会被 join_words 插入空格，切段估算必须与实际拼接同源，
    # 否则段落长度会因未计入空格而突破 max_chars（回归 I1）。
    labeled = [(W(chr(97 + i % 26), i * 0.1, i * 0.1 + 0.05), 0) for i in range(400)]
    ps = build_paragraphs(labeled, pause=2.0, max_chars=300)
    assert all(len(p.text) <= 300 for p in ps)
    assert sum(len(p.text.replace(" ", "")) for p in ps) == 400  # 无字符丢失

# ---- 首战后修正 1：热词回吐过滤 ----
def test_drop_context_echo_filters_hotword_echo():
    from merge import drop_context_echo
    ctx = "张三 李四 Tom 财务核对"
    texts = ["张三李四Tom财务核对",                       # (a) 恰等于 key
             "李四Tom财务核对",                             # (b) key 子串且 ≥ 8 字
             "张三李四Tom财务核对张三李四Tom财务核对",  # (c) key*2
             "李四说这个账要核对"]                            # 实质内容，保留
    paras = [Paragraph(None, i * 10.0, i * 10.0 + 1, t) for i, t in enumerate(texts)]
    assert [p.text for p in drop_context_echo(paras, ctx)] == ["李四说这个账要核对"]
    extra = [Paragraph(0, 0, 1, "张三，李四 Tom 财务核对。"),       # 空白 / 标点不影响判定
             Paragraph(0, 1, 2, "张三李四Tom财务核对张三李"),      # key*2 的前缀
             Paragraph(0, 2, 3, "财务核对"),                             # 子串但 < 8 字，保留
             Paragraph(0, 3, 4, "Tom 说财务核对完了")]                 # 含热词但非回吐，保留
    assert [p.text for p in drop_context_echo(extra, ctx)] == ["财务核对", "Tom 说财务核对完了"]

def test_drop_context_echo_short_or_empty_context_is_noop():
    from merge import drop_context_echo
    paras = [Paragraph(None, 0, 1, "张三李四"), Paragraph(0, 1, 2, "张三 李四 Ga")]
    assert drop_context_echo(paras, "张三 李四") == paras       # key 5 字 < 8 → 不过滤
    assert drop_context_echo(paras, "张三 李四 Ga") == paras     # key 7 字 < 8 → 不过滤
    assert drop_context_echo(paras, "") == paras
    assert drop_context_echo([], "张三 李四 Tom 财务核对") == []
