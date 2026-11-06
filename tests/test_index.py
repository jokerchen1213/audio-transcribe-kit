import os, time
from pathlib import Path
from index import md_target, needs_transcribe, upsert

def test_md_target(tmp_path):
    src = tmp_path / "通话录音 某秘书.m4a"
    assert md_target(src) == tmp_path / "_md" / "通话录音 某秘书.md"

def test_needs_transcribe_three_gates(tmp_path):
    src = tmp_path / "a.m4a"; src.write_bytes(b"x")
    assert needs_transcribe(src, False) == (True, "new")
    md = md_target(src); md.parent.mkdir(); md.write_text('---\nsource: "a.m4a"\n---\nold', encoding="utf-8")   # 本工具自产的文件头
    assert needs_transcribe(src, False) == (False, "exists")
    assert needs_transcribe(src, True) == (True, "forced")
    later = time.time() + 100; os.utime(src, (later, later))
    assert needs_transcribe(src, False) == (False, "stale-hint")

def test_upsert_idempotent_and_format(tmp_path):
    idx = tmp_path / "_转换索引.md"
    upsert(idx, "a.m4a", "00:08:35", "_md/a.md", "✅已转", "qwen3-mlx+sherpa", "2026-09-21")
    upsert(idx, "a.m4a", "00:08:36", "_md/a.md", "✅已转", "qwen3-mlx+sherpa", "2026-09-22")
    upsert(idx, "b.mp3", "00:01:00", "_md/b.md", "✅已转", "qwen3-mlx", "2026-09-22")
    lines = idx.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "| 原件 | 页数 | MD | 状态 | 引擎 | 日期 |" and lines[1] == "|---|---|---|---|---|---|"
    assert len(lines) == 4 and "00:08:36" in lines[2] and lines[3].startswith("| b.mp3 |")

def test_upsert_preserves_foreign_lines(tmp_path):
    idx = tmp_path / "_转换索引.md"
    docx_row = "| 强制执行申请书-王五赵六.docx | —（0节） | _md/强制执行申请书-王五赵六.md | ✅已转 | native-docx | 2026-08-14 |"
    legacy_5col_row = "| 某某日常工作及生活对接时间线—商务.doc | _md/….md | 16 页 | 2026-08-19 | 经 LibreOffice 转 PDF 后 OCR |"
    note_line = "> 注：手动补录"
    idx.write_text(
        "\n".join([
            "| 原件 | 页数 | MD | 状态 | 引擎 | 日期 |",
            "|---|---|---|---|---|---|",
            docx_row,
            legacy_5col_row,
            note_line,
        ]) + "\n",
        encoding="utf-8",
    )

    upsert(idx, "通话录音.m4a", "00:08:35", "_md/通话录音.md", "✅已转", "qwen3-mlx+sherpa", "2026-09-21")
    lines = idx.read_text(encoding="utf-8").splitlines()
    seeded = [docx_row, legacy_5col_row, note_line]
    assert all(s in lines for s in seeded)
    assert [l for l in lines if l in seeded] == seeded
    assert sum(1 for l in lines if l.startswith("| 通话录音.m4a |")) == 1

    upsert(idx, "通话录音.m4a", "00:09:10", "_md/通话录音.md", "✅已转", "qwen3-mlx+sherpa", "2026-09-22")
    lines = idx.read_text(encoding="utf-8").splitlines()
    assert all(s in lines for s in seeded)
    assert [l for l in lines if l in seeded] == seeded
    audio_rows = [l for l in lines if l.startswith("| 通话录音.m4a |")]
    assert len(audio_rows) == 1 and "00:09:10" in audio_rows[0]

def test_upsert_collapses_duplicate_rows(tmp_path):
    idx = tmp_path / "_转换索引.md"
    row1 = "| dup.m4a | 00:01:00 | _md/dup.md | ✅已转 | qwen3-mlx | 2026-01-01 |"
    unrelated_row = "| other.m4a | 00:02:00 | _md/other.md | ✅已转 | qwen3-mlx | 2026-01-03 |"
    row2 = "| dup.m4a | 00:01:05 | _md/dup.md | ✅已转 | qwen3-mlx | 2026-01-02 |"
    idx.write_text(
        "\n".join([
            "| 原件 | 页数 | MD | 状态 | 引擎 | 日期 |",
            "|---|---|---|---|---|---|",
            row1,
            unrelated_row,
            row2,
        ]) + "\n",
        encoding="utf-8",
    )

    upsert(idx, "dup.m4a", "00:01:10", "_md/dup.md", "✅已转", "qwen3-mlx", "2026-09-22")
    lines = idx.read_text(encoding="utf-8").splitlines()
    data_rows = lines[2:]
    assert len(data_rows) == 2
    assert data_rows[0].startswith("| dup.m4a |") and "2026-09-22" in data_rows[0]
    assert data_rows[1] == unrelated_row

def test_md_target_cross_format_disambiguation(tmp_path):
    src = tmp_path / "会议.m4a"; mddir = tmp_path / "_md"; mddir.mkdir()
    assert md_target(src) == mddir / "会议.md"                                   # (a) 无既有文件
    (mddir / "会议.md").write_text('---\nsource: "会议.m4a"\nduration: 00:01:00\n---\n', encoding="utf-8")
    assert md_target(src) == mddir / "会议.md"                                   # (b) 本工具自产 → 同路径（重跑 / --force）
    (mddir / "会议.md").write_text("---\nsource: ../会议.docx\nengine: native-docx\n---\n", encoding="utf-8")
    assert md_target(src) == mddir / "会议.m4a.md"                               # (c) cc-convert 同名产物 → 扩展名消歧
    assert needs_transcribe(src, False) == (True, "new")
