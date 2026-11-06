"""second_opinion.py — MOSS 第二意见：逐段比对 Qwen3（Windows 路线）与 MOSS 的文字分歧。

定位（B1 卡 · 律师 2026-10-22 裁定方案 A）：Qwen3 出文字，MOSS 只作第二意见，
**绝不进主转录稿正文**；本模块只在 `_md/` 里另写一份 `<同名>.分歧.md` 供人工回听核
对，主 `_md` 的正文不动，只在 front-matter 记两个计数。

比对口径（B1 卡 §四）：
- 对每个 MOSS 段，去标点后 ≥ 6 字、且不落在任何块 `last_ts` 之后的缺口区（MOSS 没转
  到的尾段，不算它读错）才比对；
- 取字时间中点落在该段 `[start, end]` 内的 Qwen3 字，去标点后与 MOSS 段比较；
- 相似度 = 1 − 编辑距离 / max(两者长度)；
  * 相似度 < 0.85 → 「文字分歧」；
  * MOSS 段 ≥ 10 字而 Qwen3 对应字数 < MOSS 的 30% → 「Qwen3 疑似漏字」（单独计数，
    这是更具体的诊断，优先于「文字分歧」，不重复计数）。
- 编辑距离纯 Python 实现（段短），不引入新依赖。
"""
from __future__ import annotations

import os
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

MIN_CHARS = 6          # MOSS 段去标点后至少 6 字才比对
GAP_MIN_MOSS_CHARS = 10  # 「疑似漏字」要求 MOSS 段 ≥ 10 字
GAP_MAX_QWEN_RATIO = 0.30  # 且 Qwen3 对应字数 < MOSS 的 30%
SIMILARITY_MIN = 0.85    # 相似度低于此值算「文字分歧」


def _squash(s: str) -> str:
    """去掉全部空白与标点（Unicode P* 类），只留可比对的实质字符（与 merge._squash 同口径）。"""
    return "".join(c for c in s if not c.isspace() and not unicodedata.category(c).startswith("P"))


def edit_distance(a: str, b: str) -> int:
    """Levenshtein 距离（两行滚动数组；段短，纯 Python 足够）。"""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def similarity(a: str, b: str) -> float:
    """1 − 编辑距离 / max(len)，两者皆空时为 1.0。"""
    longest = max(len(a), len(b))
    if longest == 0:
        return 1.0
    return 1.0 - edit_distance(a, b) / longest


@dataclass
class Result:
    entries: list[dict] = field(default_factory=list)   # {start, kind, qwen, moss}
    disagreements: int = 0
    suspected_gaps: int = 0
    moss_gap_s: float = 0.0

    def __bool__(self) -> bool:
        return bool(self.entries)


def _gap_regions(chunks: list[dict]) -> tuple[list[tuple[float, float]], float]:
    """每个 MOSS 块 `last_ts` 到块尾的缺口区，及其总秒数。"""
    regions: list[tuple[float, float]] = []
    total = 0.0
    for ch in chunks or []:
        try:
            end = float(ch.get("end", 0.0))
            last = float(ch.get("last_ts", 0.0))
        except (TypeError, ValueError):
            continue
        if end > last:
            regions.append((last, end))
            total += end - last
    return regions, round(total, 1)


def compare(words: list, segments: list[dict], chunks: list[dict]) -> Result:
    """Qwen3 字流 vs MOSS 段。`words` 只需有 .text/.start/.end（backends.base.Word）。"""
    regions, gap_s = _gap_regions(chunks)
    res = Result(moss_gap_s=gap_s)

    def in_gap(start: float) -> bool:
        return any(a - 1e-6 <= start < b for a, b in regions)

    for seg in segments or []:
        moss_text = _squash(str(seg.get("text") or ""))
        if len(moss_text) < MIN_CHARS:
            continue
        try:
            start, end = float(seg["start"]), float(seg["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if in_gap(start):
            continue   # MOSS 自己没转到的尾段：不拿它当「第二意见」
        chars = [w.text for w in words if start <= (w.start + w.end) / 2.0 <= end]
        qwen_text = _squash("".join(chars))
        if (len(moss_text) >= GAP_MIN_MOSS_CHARS
                and len(qwen_text) < GAP_MAX_QWEN_RATIO * len(moss_text)):
            res.entries.append({"start": start, "kind": "Qwen3 疑似漏字", "qwen": qwen_text, "moss": moss_text})
            res.suspected_gaps += 1
            continue
        if similarity(qwen_text, moss_text) < SIMILARITY_MIN:
            res.entries.append({"start": start, "kind": "文字分歧", "qwen": qwen_text, "moss": moss_text})
            res.disagreements += 1
    return res


# ---- 报告落盘（原子写入，与主 _md 同目录）-----------------------------------

def _q(s: str) -> str:
    """YAML 双引号转义（与 writer._q 同口径）。"""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _cell(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def _write_atomic(target: Path, text: str) -> None:
    """先写 .tmp 再 os.replace（与 cli._write_atomic 同法）：中断不留半成品。"""
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def report_path(md_path: Path) -> Path:
    md_path = Path(md_path)
    return md_path.with_name(md_path.stem + ".分歧.md")


def write_report(md_path: Path, res: Result, label: str, source: str) -> Path | None:
    """有条目才写 `_md/<同名>.分歧.md`；返回写出的路径，无条目返回 None。"""
    if not res.entries:
        return None
    target = report_path(md_path)
    lines = ["---", f"source: {_q(source)}", f"second_opinion: {label}",
             f"disagreements: {res.disagreements}", f"suspected_gaps: {res.suspected_gaps}",
             f"moss_gap_s: {res.moss_gap_s:.1f}", "---", "",
             "> MOSS 只作第二意见；引用原文或质证前请回听原声核对。", "",
             "| 时间 | 类型 | Qwen3 文字 | MOSS 文字 |", "|---|---|---|---|"]
    from writer import format_ts   # 局部导入：writer 不依赖本模块，避免顶层循环依赖
    for e in res.entries:
        lines.append(f"| {format_ts(e['start'])} | {e['kind']} | {_cell(e['qwen'])} | {_cell(e['moss'])} |")
    _write_atomic(target, "\n".join(lines) + "\n")
    return target
