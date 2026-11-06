"""_转换索引.md 维护（六列格式与 cc-convert 逐字一致，幂等 upsert）+ 去重三闸。

索引文件与 cc-convert 共享同一份 `_转换索引.md`：upsert 对任何未被精确识别为
本工具六列数据行的既有行（cc-convert 遗留的五列旧行、标题/备注行、非精确匹配
的分隔行等）一律逐字保留、保持原有相对顺序，绝不静默丢弃。
"""
from __future__ import annotations
import os
from pathlib import Path

COLUMNS = ["原件", "页数", "MD", "状态", "引擎", "日期"]
HEADER = "| " + " | ".join(COLUMNS) + " |"
SEP = "|" + "|".join(["---"] * len(COLUMNS)) + "|"

def md_target(src: Path) -> Path:
    """默认 `_md/<stem>.md`；若该文件已存在且文件头不含本工具写的 `source: "<原件名>"`（即 cc-convert 等
    跨格式同名产物），改用 `_md/<stem>.<ext>.md` 消歧（与 cc-convert `_md_path` 同一约定）。"""
    cand = src.parent / "_md" / (src.stem + ".md")
    if cand.exists():
        with cand.open(encoding="utf-8", errors="ignore") as f:
            head = f.read(200)
        if f'source: "{src.name}"' not in head:
            return src.parent / "_md" / f"{src.stem}.{src.suffix.lstrip('.')}.md"
    return cand

def needs_transcribe(src: Path, force: bool) -> tuple[bool, str]:
    md = md_target(src)
    if not md.exists():
        return True, "new"
    if force:
        return True, "forced"
    if src.stat().st_mtime > md.stat().st_mtime:
        return False, "stale-hint"
    return False, "exists"

def _row_cells(line: str) -> list[str] | None:
    """line 是本工具六列数据行则返回其 cells，否则返回 None（含 header/sep 精确匹配、
    列数不符、全空首列、cells 全由 -/:/空格 组成的类分隔行——一律视为外来内容，
    交由 upsert 原样保留，不参与识别）。"""
    s = line.strip()
    if not (s.startswith("|") and s.endswith("|")):
        return None
    if s == HEADER or s == SEP:
        return None
    cells = [c.strip() for c in s.strip("|").split("|")]
    if len(cells) != len(COLUMNS) or not cells[0]:
        return None
    if all(set(c) <= set("-: ") for c in cells):
        return None
    return cells

def upsert(index_path: Path, original: str, pages: str, md_rel: str, status: str, engine: str, date: str) -> None:
    p = Path(index_path)
    new_row = "| " + " | ".join([original, pages, md_rel, status, engine, date]) + " |"

    lines = p.read_text(encoding="utf-8").splitlines() if p.exists() else []
    has_header = any(line.strip() == HEADER for line in lines)

    out: list[str] = [] if has_header else [HEADER, SEP]
    replaced = False
    last_row_idx: int | None = None
    last_sep_idx: int | None = None

    for line in lines:
        s = line.strip()
        if s == SEP:
            out.append(line)
            last_sep_idx = len(out) - 1
            continue
        cells = _row_cells(line)
        if cells is None:
            out.append(line)  # 外来行（header / 五列旧行 / 标题 / 备注等）：原样保留
            continue
        if cells[0] == original:
            if replaced:
                continue  # 既有索引中该 原件 的重复行：折叠为一条，仅保留首次出现处已替换的那行
            out.append(new_row)
            replaced = True
        else:
            out.append(line)
        last_row_idx = len(out) - 1

    if not replaced:
        if last_row_idx is not None:
            out.insert(last_row_idx + 1, new_row)
        elif last_sep_idx is not None:
            out.insert(last_sep_idx + 1, new_row)
        else:
            out.append(new_row)

    text = "\n".join(out) + "\n"
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)
