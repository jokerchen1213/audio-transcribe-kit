"""后端无关的拼接层：词按时间中点归说话人段，再按说话人 / 停顿 / 长度成段。"""
from __future__ import annotations
import unicodedata
from dataclasses import dataclass
from backends.base import SpeakerTurn, Word

@dataclass
class Paragraph:
    speaker: int | None   # 0 起；None = 无说话人信息
    start: float
    end: float
    text: str

def _is_ascii_word(t: str) -> bool:
    return t.isascii() and any(c.isalnum() for c in t)

def join_words(words: list[str]) -> str:
    out = ""
    for i, w in enumerate(words):
        if i and _is_ascii_word(words[i - 1]) and _is_ascii_word(w):
            out += " "
        out += w
    return out

def shift_words(words: list[Word], offset: float) -> list[Word]:
    return [Word(w.text, w.start + offset, w.end + offset) for w in words]

def assign_speakers(words: list[Word], turns: list[SpeakerTurn], gap: float = 1.0) -> list[tuple[Word, int | None]]:
    turns = sorted(turns, key=lambda t: t.start)
    out = []
    for w in words:
        mid = (w.start + w.end) / 2
        hit = next((t for t in turns if t.start <= mid <= t.end), None)
        if hit is None and turns:
            near = min(turns, key=lambda t: min(abs(t.start - mid), abs(t.end - mid)))
            if min(abs(near.start - mid), abs(near.end - mid)) <= gap:
                hit = near
        out.append((w, hit.speaker if hit else None))
    return out

def build_paragraphs(labeled: list[tuple[Word, int | None]], pause: float = 2.0, max_chars: int = 300) -> list[Paragraph]:
    paras: list[Paragraph] = []
    cur: list[Word] = []
    cur_spk: int | None = None
    cur_len = 0   # 与 join_words 同源的累计长度（含词间空格），段落长度只此一处计算
    def flush():
        if cur:
            paras.append(Paragraph(cur_spk, cur[0].start, cur[-1].end, join_words([w.text for w in cur])))
    for w, spk in labeled:
        sep = 1 if cur and _is_ascii_word(cur[-1].text) and _is_ascii_word(w.text) else 0
        new = (not cur) or spk != cur_spk or (w.start - cur[-1].end) > pause \
              or cur_len + sep + len(w.text) > max_chars
        if new and cur:
            flush(); cur = []; cur_len = 0
        cur_spk = spk if (new or not cur) else cur_spk
        sep = 1 if cur and _is_ascii_word(cur[-1].text) and _is_ascii_word(w.text) else 0
        cur.append(w)
        cur_len += sep + len(w.text)
    flush()
    return paras

def _squash(s: str) -> str:
    """去掉全部空白与标点（Unicode P* 类），只留可比对的实质字符。"""
    return "".join(c for c in s if not c.isspace() and not unicodedata.category(c).startswith("P"))

def drop_context_echo(paragraphs: list[Paragraph], context: str, min_len: int = 8) -> list[Paragraph]:
    """剔除 ASR 在嘈杂 / 重叠片段把 `--context` 热词串原样回吐成的段落（后端无关、纯函数）。
    key = context 去空白标点；key < min_len 字或为空时原样返回。段落文本去空白标点后满足其一即删：
    (a) 等于 key；(b) 是 key 的子串且 ≥ min_len 字；(c) 由 key 重复拼接而成（恰为 key*n 或其前缀，n ≥ 1）。
    含其他实质内容的段落一律保留。"""
    key = _squash(context)
    if len(key) < min_len:
        return list(paragraphs)
    def is_echo(text: str) -> bool:
        t = _squash(text)
        if len(t) < min_len:
            return False
        if t in key:                       # (a) 与 (b)
            return True
        n = -(-len(t) // len(key))         # (c)：t 是 key*n 的前缀（n = ceil(len(t)/len(key))）
        return t.startswith(key) and (key * n).startswith(t)
    return [p for p in paragraphs if not is_echo(p.text)]
